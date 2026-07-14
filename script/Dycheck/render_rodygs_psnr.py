import json
import math
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from omegaconf.dictconfig import DictConfig
from omegaconf import OmegaConf
from PIL import Image

from arguments import ModelParams, OptimizationParams, PipelineParams
from script.Dycheck.render_official_val import (
    apply_sim3_to_c2w,
    build_alignment,
    camera_from_c2w,
    dycheck_c2w_from_camera,
    image_stats,
    load_frame_names,
    load_json,
    tensor_to_uint8_image,
)

RETAIN_RENDER_INTERVAL = 50


def merge_config_into_args(args, config_path):
    cfg = OmegaConf.load(config_path)

    def recursive_merge(key, host):
        value = host[key]
        if isinstance(value, DictConfig):
            for child_key in value.keys():
                recursive_merge(child_key, value)
        else:
            if not hasattr(args, key):
                raise AttributeError(f"Unknown config key: {key}")
            setattr(args, key, value)

    for key in cfg.keys():
        recursive_merge(key, cfg)
    return args


def load_rgb_float(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def psnr_float(pred, gt):
    mse = float(np.mean((pred - gt) ** 2))
    if mse <= 0.0:
        return float("inf")
    return float(20.0 * math.log10(1.0 / math.sqrt(mse)))


class FullImageMetricEvaluator:
    def __init__(self, device):
        try:
            from piqa import LPIPS, SSIM
        except ImportError as exc:
            raise ImportError(
                "RoDyGS full-image SSIM/LPIPS metrics require piqa. "
                "Install it in the Instant4D environment with `pip install piqa`."
            ) from exc

        self.ssim = SSIM().to(device).eval()
        self.lpips = LPIPS(network="alex").to(device).eval()
        self.device = device

    def tensor_from_rgb(self, image):
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device, dtype=torch.float32).clamp(0.0, 1.0)

    def evaluate(self, pred, gt):
        pred_tensor = self.tensor_from_rgb(pred)
        gt_tensor = self.tensor_from_rgb(gt)
        with torch.inference_mode():
            return {
                "ssim": float(self.ssim(gt_tensor, pred_tensor).detach().cpu()),
                "lpips": float(self.lpips(gt_tensor, pred_tensor).detach().cpu()),
            }


def frame_numeric_id(frame_name):
    matches = re.findall(r"\d+", frame_name)
    if not matches:
        return None
    return int(matches[-1])


def source_train_frame_count(source_path):
    transforms_path = source_path / "transforms_train.json"
    if not transforms_path.exists():
        return None
    transforms = load_json(transforms_path)
    return len(transforms.get("frames", []))


def sampled_render_keep_names(frame_names, source_path, interval=RETAIN_RENDER_INTERVAL):
    if not frame_names:
        return set(), []

    train_count = source_train_frame_count(source_path)
    if not train_count:
        train_count = len(frame_names)

    frame_ids = [frame_numeric_id(name) for name in frame_names]
    if any(frame_id is None for frame_id in frame_ids):
        keep = frame_names[::interval]
        return set(keep), keep

    targets = range(0, train_count, interval)
    selected = []
    used = set()
    for target in targets:
        best_index = min(
            (idx for idx in range(len(frame_names)) if idx not in used),
            key=lambda idx: (abs(frame_ids[idx] - target), idx),
            default=None,
        )
        if best_index is None:
            break
        used.add(best_index)
        selected.append(frame_names[best_index])
    return set(selected), selected


def prune_rendered_images(per_frame, pred_dir, source_path):
    keep_names, keep_order = sampled_render_keep_names(
        [item["frame_name"] for item in per_frame],
        source_path,
    )
    deleted = []
    retained = []
    for item in per_frame:
        pred_path = Path(item["pred_path"])
        keep = item["frame_name"] in keep_names
        item["render_retained"] = keep
        if keep:
            retained.append(item["frame_name"])
            continue
        if pred_path.parent != pred_dir:
            raise RuntimeError(f"Refusing to delete render outside prediction directory: {pred_path}")
        if pred_path.exists():
            pred_path.unlink()
            deleted.append(item["frame_name"])

    return {
        "enabled": True,
        "interval": RETAIN_RENDER_INTERVAL,
        "selection": "nearest evaluated frame to each source-train timeline interval",
        "retained_frame_names": keep_order,
        "num_retained": len(retained),
        "num_deleted": len(deleted),
    }


def default_holdout_path(dycheck_scene_dir, source_path):
    candidates = [
        source_path / "rodygs_holdout_frames.json",
        dycheck_scene_dir / "preprocess_output" / "rodygs_split" / "test.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find RoDyGS holdout frames. Expected "
        f"{candidates[0]} or {candidates[1]}."
    )


def build_parser():
    parser = ArgumentParser(
        description="Render and evaluate Instant4D with RoDyGS iPhone holdout full-image metrics."
    )
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dycheck_scene_dir", required=True)
    parser.add_argument("--holdout_json", default="")
    parser.add_argument("--image_scale", default="2x")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--max_alignment_rmse", type=float, default=0.05)
    parser.add_argument("--max_alignment_rot_deg", type=float, default=10.0)
    parser.add_argument("--sanity_frames", type=int, default=5)
    parser.add_argument("--min_sanity_mean_rgb", type=float, default=1.0)
    parser.add_argument("--retain_sampled_renders", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--gaussian_dim", type=int, default=4)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument("--num_pts_ratio", type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--exhaust_test", action="store_true", default=False)

    return parser, lp, op, pp


def main():
    parser, lp, op, pp = build_parser()
    args = parser.parse_args()
    args = merge_config_into_args(args, args.config)

    checkpoint_path = Path(args.checkpoint)
    dycheck_scene_dir = Path(args.dycheck_scene_dir)
    source_path = Path(args.source_path)
    output_dir = Path(args.output_dir)
    pred_dir = output_dir / "rgb" / args.image_scale / "rodygs_holdout"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not dycheck_scene_dir.exists():
        raise FileNotFoundError(f"DyCheck scene directory not found: {dycheck_scene_dir}")
    if not source_path.exists():
        raise FileNotFoundError(f"Instant4D source path not found: {source_path}")

    holdout_path = Path(args.holdout_json) if args.holdout_json else default_holdout_path(dycheck_scene_dir, source_path)
    frame_names = load_frame_names(holdout_path)
    if args.max_frames > 0:
        frame_names = frame_names[: args.max_frames]

    alignment = build_alignment(
        dycheck_scene_dir,
        source_path,
        args.max_alignment_rmse,
        args.max_alignment_rot_deg,
    )

    torch.set_grad_enabled(False)
    dataset = lp.extract(args)
    pipe = pp.extract(args)

    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel

    time_duration = args.time_duration
    if dataset.frame_ratio > 1:
        time_duration = [time_duration[0] / dataset.frame_ratio, time_duration[1] / dataset.frame_ratio]

    gaussians = GaussianModel(
        dataset.sh_degree,
        gaussian_dim=args.gaussian_dim,
        time_duration=time_duration,
        rot_4d=args.rot_4d,
        force_sh_3d=args.force_sh_3d,
        sh_degree_t=2 if pipe.eval_shfs_4d else 0,
    )

    model_params, checkpoint_iter = torch.load(str(checkpoint_path), map_location="cuda", weights_only=False)
    gaussians.restore(model_params, None)
    if gaussians.env_map is not None and hasattr(gaussians.env_map, "shape") and gaussians.env_map.shape[0] > 0:
        pipe.env_map_res = gaussians.env_map.shape[0]
    metric_evaluator = FullImageMetricEvaluator("cuda")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    metadata = load_json(dycheck_scene_dir / "metadata.json")
    dataset_json = load_json(dycheck_scene_dir / "dataset.json")
    time_den = max(int(dataset_json["num_exemplars"]) - 1, 1)

    pred_dir.mkdir(parents=True, exist_ok=True)
    for old_prediction in pred_dir.glob("*.png"):
        old_prediction.unlink()

    per_frame = []
    render_stats = []
    gt_dir = dycheck_scene_dir / "rgb" / args.image_scale
    for index, frame_name in enumerate(frame_names):
        camera_json = load_json(dycheck_scene_dir / "camera" / f"{frame_name}.json")
        raw_c2w = dycheck_c2w_from_camera(camera_json)
        aligned_c2w = apply_sim3_to_c2w(
            raw_c2w,
            alignment["scale"],
            alignment["rotation"],
            alignment["translation"],
        )
        timestamp = float(metadata[frame_name]["warp_id"]) / time_den * 3.0
        gt_path = gt_dir / f"{frame_name}.png"
        camera = camera_from_c2w(frame_name, aligned_c2w, camera_json, gt_path, timestamp, dataset.data_device)
        render_pkg = render(camera.cuda(), gaussians, pipe, background)
        image_np = tensor_to_uint8_image(render_pkg["render"])
        pred_path = pred_dir / f"{frame_name}.png"
        Image.fromarray(image_np).save(pred_path)

        stats = {"frame_name": frame_name}
        stats.update(image_stats(image_np))
        render_stats.append(stats)
        if args.sanity_frames > 0 and len(render_stats) == args.sanity_frames:
            sanity_mean = float(np.mean([item["mean_rgb"] for item in render_stats]))
            sanity_max = max(item["max_rgb"] for item in render_stats)
            if sanity_max == 0 or sanity_mean < args.min_sanity_mean_rgb:
                raise RuntimeError(
                    "Rendered sanity frames are nearly black. "
                    f"mean_rgb={sanity_mean:.6f}, max_rgb={sanity_max}. "
                    "Check camera coordinate conversion before running metrics."
                )

        pred = image_np.astype(np.float32) / 255.0
        gt = load_rgb_float(gt_path)
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch for {frame_name}: pred {pred.shape}, gt {gt.shape}")
        image_metrics = metric_evaluator.evaluate(pred, gt)
        per_frame.append(
            {
                "frame_name": frame_name,
                "psnr": psnr_float(pred, gt),
                "ssim": image_metrics["ssim"],
                "lpips": image_metrics["lpips"],
                "pred_path": str(pred_path),
                "gt_path": str(gt_path),
            }
        )
        if not args.quiet and (index + 1) % 25 == 0:
            print(f"Rendered/evaluated {index + 1}/{len(frame_names)} frames")

    psnr_values = [item["psnr"] for item in per_frame]
    ssim_values = [item["ssim"] for item in per_frame]
    lpips_values = [item["lpips"] for item in per_frame]
    render_retention = {"enabled": False}
    if args.retain_sampled_renders:
        render_retention = prune_rendered_images(per_frame, pred_dir, source_path)

    summary = {
        "protocol": "rodygs_iphone_holdout_full_image",
        "checkpoint": str(checkpoint_path),
        "checkpoint_iteration": int(checkpoint_iter),
        "dycheck_scene_dir": str(dycheck_scene_dir),
        "source_path": str(source_path),
        "holdout_json": str(holdout_path),
        "image_scale": args.image_scale,
        "output_dir": str(output_dir),
        "prediction_dir": str(pred_dir),
        "num_frames": len(frame_names),
        "num_evaluated_frames": len(per_frame),
        "psnr": float(np.mean(psnr_values)) if psnr_values else None,
        "ssim": float(np.mean(ssim_values)) if ssim_values else None,
        "lpips": float(np.mean(lpips_values)) if lpips_values else None,
        "metric_details": {
            "psnr": "Full-image RGB PSNR on RoDyGS-style held-out frames.",
            "ssim": "Full-image SSIM from piqa.SSIM on RGB tensors in [0, 1].",
            "lpips": "Full-image LPIPS from piqa.LPIPS with AlexNet backbone on RGB tensors in [0, 1].",
        },
        "alignment": {
            "scale": alignment["scale"],
            "rotation": alignment["rotation"].tolist(),
            "translation": alignment["translation"].tolist(),
            "center_error_rmse": alignment["center_error_rmse"],
            "center_error_p50": alignment["center_error_p50"],
            "center_error_p90": alignment["center_error_p90"],
            "center_error_max": alignment["center_error_max"],
            "rotation_error_deg_p50": alignment["rotation_error_deg_p50"],
            "rotation_error_deg_p90": alignment["rotation_error_deg_p90"],
            "rotation_error_deg_max": alignment["rotation_error_deg_max"],
            "num_train_pairs": alignment["num_train_pairs"],
        },
        "render_stats": {
            "mean_rgb": float(np.mean([item["mean_rgb"] for item in render_stats])) if render_stats else None,
            "max_rgb": max([item["max_rgb"] for item in render_stats]) if render_stats else None,
            "mean_nonzero_ratio": float(np.mean([item["nonzero_ratio"] for item in render_stats])) if render_stats else None,
            "first_frames": render_stats[: min(10, len(render_stats))],
        },
        "render_retention": render_retention,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics_per_frame.json", "w", encoding="utf-8") as f:
        json.dump(per_frame, f, indent=2)
    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "render_manifest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
