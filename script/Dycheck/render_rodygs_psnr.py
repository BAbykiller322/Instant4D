import json
import math
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
        description="Render and evaluate Instant4D with the RoDyGS iPhone holdout PSNR protocol."
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
        per_frame.append(
            {
                "frame_name": frame_name,
                "psnr": psnr_float(pred, gt),
                "pred_path": str(pred_path),
                "gt_path": str(gt_path),
            }
        )
        if not args.quiet and (index + 1) % 25 == 0:
            print(f"Rendered/evaluated {index + 1}/{len(frame_names)} frames")

    psnr_values = [item["psnr"] for item in per_frame]
    summary = {
        "protocol": "rodygs_iphone_holdout_psnr",
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
