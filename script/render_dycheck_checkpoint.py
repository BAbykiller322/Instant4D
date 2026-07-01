import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from omegaconf.dictconfig import DictConfig
from omegaconf import OmegaConf
from PIL import Image

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from arguments import ModelParams, OptimizationParams, PipelineParams


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
    image_np = image.detach().cpu().clamp(0.0, 1.0).numpy()
    image_np = np.transpose(image_np, (1, 2, 0))
    return (image_np * 255.0).round().astype(np.uint8)


def save_render_outputs(render_pkg, output_dir, stem, save_depth=False, save_alpha=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_uint8_image(render_pkg["render"])).save(output_dir / f"{stem}.png")

    if save_depth and "depth" in render_pkg:
        depth = render_pkg["depth"].detach().cpu().squeeze().numpy()
        if np.isfinite(depth).any():
            depth_min = np.nanmin(depth)
            depth_max = np.nanmax(depth)
            denom = max(depth_max - depth_min, 1e-8)
            depth_vis = ((depth - depth_min) / denom * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(depth_vis).save(output_dir / f"{stem}_depth.png")

    if save_alpha and "alpha" in render_pkg:
        alpha = render_pkg["alpha"].detach().cpu().squeeze().clamp(0.0, 1.0).numpy()
        alpha_vis = (alpha * 255.0).round().astype(np.uint8)
        Image.fromarray(alpha_vis).save(output_dir / f"{stem}_alpha.png")


def build_parser():
    parser = ArgumentParser(description="Render an Instant4D DyCheck checkpoint locally.")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--config", default=str(repo_root / "configs" / "dycheck" / "apple.yaml"))
    parser.add_argument("--checkpoint", default="", help="Checkpoint path. Defaults to chkpnt<iteration>.pth in model_path.")
    parser.add_argument("--iteration", type=int, default=5000, help="Checkpoint iteration used when --checkpoint is omitted.")
    parser.add_argument("--output_path", default="", help="Output directory. Defaults to <model_path>/renders/chkpnt<iteration>.")
    parser.add_argument("--split", choices=["train", "test", "both"], default="test")
    parser.add_argument("--max_frames", type=int, default=-1, help="Limit rendered frames for quick inspection.")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--save_depth", action="store_true")
    parser.add_argument("--save_alpha", action="store_true")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
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

    if args.frame_stride < 1:
        raise ValueError("--frame_stride must be >= 1")

    model_path = Path(args.model_path)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else model_path / f"chkpnt{args.iteration}.pth"
    output_path = Path(args.output_path) if args.output_path else model_path / "renders" / f"chkpnt{args.iteration}"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not Path(args.source_path).exists():
        raise FileNotFoundError(f"Source path not found: {args.source_path}")

    torch.set_grad_enabled(False)
    dataset = lp.extract(args)
    pipe = pp.extract(args)

    from gaussian_renderer import render
    from scene import Scene
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

    camera_sets = []
    if args.split in ("train", "both"):
        camera_sets.append(("train", scene.getTrainCameras()))
    if args.split in ("test", "both"):
        camera_sets.append(("test", scene.getTestCameras()))

    total_rendered = 0
    for split_name, cameras in camera_sets:
        rendered_in_split = 0
        split_output = output_path / split_name
        indices = list(range(0, len(cameras), args.frame_stride))
        if args.max_frames > 0:
            indices = indices[: args.max_frames]

        for out_idx, cam_idx in enumerate(indices):
            _, viewpoint_cam = cameras[cam_idx]
            viewpoint_cam = viewpoint_cam.cuda()
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            stem = f"{out_idx:05d}_cam{cam_idx:05d}_{viewpoint_cam.image_name}"
            save_render_outputs(render_pkg, split_output, stem, args.save_depth, args.save_alpha)
            rendered_in_split += 1

        total_rendered += rendered_in_split
        if not args.quiet:
            print(f"Rendered {rendered_in_split} {split_name} frames to {split_output}")

    if not args.quiet:
        print(f"Loaded checkpoint iteration {checkpoint_iter} from {checkpoint_path}")
        print(f"Rendered {total_rendered} frames total.")


if __name__ == "__main__":
    main()
