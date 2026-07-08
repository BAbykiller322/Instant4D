import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from PIL import Image

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import ssim


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


def tensor_to_uint8_image(image):
    image = image.detach().clamp(0.0, 1.0).cpu()
    image = image.permute(1, 2, 0).numpy()
    return (image * 255.0 + 0.5).astype(np.uint8)


def maybe_build_lpips(compute_lpips, net):
    if not compute_lpips:
        return None
    try:
        import lpips
    except ImportError as exc:
        raise ImportError(
            "LPIPS was requested but the lpips package is not installed in this environment. "
            "Run without --compute_lpips or use an environment with lpips installed."
        ) from exc
    model = lpips.LPIPS(net=net).cuda().eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def compute_lpips_value(model, pred, gt):
    if model is None:
        return None
    pred_lpips = pred.unsqueeze(0) * 2.0 - 1.0
    gt_lpips = gt.unsqueeze(0) * 2.0 - 1.0
    return float(model(pred_lpips, gt_lpips).mean().item())


def camera_iter(scene, split):
    if split == "train":
        return scene.getTrainCameras()
    if split == "test":
        return scene.getTestCameras()
    raise ValueError(f"Unsupported split: {split}")


def evaluate_split(args, scene, gaussians, pipe, background, split, lpips_model):
    dataset = camera_iter(scene, split)
    output_dir = Path(args.output_dir)
    pred_dir = output_dir / "rgb" / split
    if args.save_images:
        pred_dir.mkdir(parents=True, exist_ok=True)
        for old_prediction in pred_dir.glob("*.png"):
            old_prediction.unlink()

    per_frame = []
    frame_count = len(dataset)
    max_frames = frame_count if args.max_frames <= 0 else min(args.max_frames, frame_count)

    for index in range(max_frames):
        gt_image, camera = dataset[index]
        gt_image = gt_image.cuda().clamp(0.0, 1.0)
        camera = camera.cuda()
        render_pkg = render(camera, gaussians, pipe, background)
        pred = render_pkg["render"].clamp(0.0, 1.0)

        item = {
            "index": index,
            "image_name": camera.image_name,
            "timestamp": float(camera.timestamp),
            "PSNR": float(psnr(pred.unsqueeze(0), gt_image.unsqueeze(0)).mean().item()),
            "SSIM": float(ssim(pred.unsqueeze(0), gt_image.unsqueeze(0)).mean().item()),
        }
        lpips_value = compute_lpips_value(lpips_model, pred, gt_image)
        if lpips_value is not None:
            item["LPIPS"] = lpips_value
        per_frame.append(item)

        if args.save_images:
            Image.fromarray(tensor_to_uint8_image(pred)).save(pred_dir / f"{camera.image_name}.png")

        if not args.quiet and (index + 1) % 25 == 0:
            print(f"[{split}] rendered {index + 1}/{max_frames} frames")

    if not per_frame:
        raise RuntimeError(f"No frames evaluated for split {split}")

    summary = {
        "split": split,
        "num_frames_available": frame_count,
        "num_frames_evaluated": len(per_frame),
        "PSNR": float(np.mean([item["PSNR"] for item in per_frame])),
        "SSIM": float(np.mean([item["SSIM"] for item in per_frame])),
    }
    if "LPIPS" in per_frame[0]:
        summary["LPIPS"] = float(np.mean([item["LPIPS"] for item in per_frame]))

    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    with open(split_dir / "metrics_per_frame.json", "w", encoding="utf-8") as f:
        json.dump(per_frame, f, indent=2)
    with open(split_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def build_parser():
    parser = ArgumentParser(description="Render and evaluate Instant4D source-view metrics.")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", choices=["train", "test", "both"], default="test")
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--compute_lpips", action="store_true")
    parser.add_argument("--lpips_net", default="alex", choices=["alex", "vgg", "squeeze"])
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
    parser, lp, _, pp = build_parser()
    args = parser.parse_args()
    args = merge_config_into_args(args, args.config)

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not Path(args.source_path).exists():
        raise FileNotFoundError(f"Instant4D source path not found: {args.source_path}")

    torch.set_grad_enabled(False)
    dataset = lp.extract(args)
    pipe = pp.extract(args)

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
    scene = Scene(
        dataset,
        gaussians,
        shuffle=False,
        num_pts=args.num_pts,
        num_pts_ratio=args.num_pts_ratio,
        time_duration=time_duration,
        initialize_gaussians=False,
    )

    model_params, checkpoint_iter = torch.load(str(checkpoint_path), map_location="cuda", weights_only=False)
    gaussians.restore(model_params, None)
    if gaussians.env_map is not None and hasattr(gaussians.env_map, "shape") and gaussians.env_map.shape[0] > 0:
        pipe.env_map_res = gaussians.env_map.shape[0]

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    lpips_model = maybe_build_lpips(args.compute_lpips, args.lpips_net)

    splits = ["train", "test"] if args.split == "both" else [args.split]
    summaries = []
    for split in splits:
        summaries.append(evaluate_split(args, scene, gaussians, pipe, background, split, lpips_model))

    manifest = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_iteration": int(checkpoint_iter),
        "config": str(args.config),
        "source_path": str(args.source_path),
        "output_dir": str(output_dir),
        "split": args.split,
        "max_frames": args.max_frames,
        "save_images": bool(args.save_images),
        "compute_lpips": bool(args.compute_lpips),
        "summaries": summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "render_source_metrics_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
