import json
import math
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
from scene.cameras import Camera

CAMERA_AXIS_FLIP = np.diag([1.0, -1.0, -1.0])


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


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dycheck_c2w_from_camera(camera_json):
    orientation = np.asarray(camera_json["orientation"], dtype=np.float64)
    position = np.asarray(camera_json["position"], dtype=np.float64)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = orientation.T @ CAMERA_AXIS_FLIP
    c2w[:3, 3] = position
    return c2w


def instant_c2w_from_transform(frame):
    c2w = np.asarray(frame["transform_matrix"], dtype=np.float64).copy()
    c2w[:3, 1:3] *= -1.0
    return c2w


def estimate_sim3(source_points, target_points):
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    if source_points.shape != target_points.shape or source_points.shape[0] < 3:
        raise ValueError("Need at least 3 paired camera centers for Sim(3) alignment")

    source_mean = source_points.mean(axis=0)
    target_mean = target_points.mean(axis=0)
    source_centered = source_points - source_mean
    target_centered = target_points - target_mean

    covariance = target_centered.T @ source_centered / source_points.shape[0]
    U, singular_values, Vt = np.linalg.svd(covariance)
    D = np.eye(3, dtype=np.float64)
    D[-1, -1] = np.sign(np.linalg.det(U @ Vt))
    rotation = U @ D @ Vt
    source_var = np.mean(np.sum(source_centered * source_centered, axis=1))
    scale = np.sum(singular_values * np.diag(D)) / source_var
    translation = target_mean - scale * rotation @ source_mean

    predicted = apply_sim3_to_points(source_points, scale, rotation, translation)
    errors = np.linalg.norm(predicted - target_points, axis=1)
    return scale, rotation, translation, errors


def apply_sim3_to_points(points, scale, rotation, translation):
    points = np.asarray(points, dtype=np.float64)
    return (scale * (rotation @ points.T)).T + translation


def apply_sim3_to_c2w(c2w, scale, rotation, translation):
    aligned = np.eye(4, dtype=np.float64)
    aligned[:3, :3] = rotation @ c2w[:3, :3]
    aligned[:3, 3] = scale * rotation @ c2w[:3, 3] + translation
    return aligned


def camera_from_c2w(frame_name, c2w, camera_json, gt_path, timestamp, data_device):
    with Image.open(gt_path) as image:
        width, height = image.size

    raw_width, raw_height = camera_json["image_size"]
    scale_x = width / raw_width
    scale_y = height / raw_height
    focal = float(camera_json["focal_length"])
    principal = camera_json["principal_point"]

    fl_x = focal * scale_x
    fl_y = focal * scale_y
    cx = float(principal[0]) * scale_x
    cy = float(principal[1]) * scale_y

    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3].T
    T = w2c[:3, 3]

    return Camera(
        colmap_id=0,
        R=R,
        T=T,
        FoVx=-1.0,
        FoVy=-1.0,
        image=np.empty(0),
        gt_alpha_mask=None,
        image_name=frame_name,
        uid=0,
        data_device=data_device,
        timestamp=timestamp,
        cx=cx,
        cy=cy,
        fl_x=fl_x,
        fl_y=fl_y,
        resolution=(width, height),
        image_path=str(gt_path),
        meta_only=True,
    )


def tensor_to_uint8_image(image):
    image_np = image.detach().cpu().clamp(0.0, 1.0).numpy()
    image_np = np.transpose(image_np, (1, 2, 0))
    return (image_np * 255.0).round().astype(np.uint8)


def load_frame_names(split_path):
    split = load_json(split_path)
    if "frame_names" in split:
        return list(split["frame_names"])
    if "ids" in split:
        return list(split["ids"])
    raise KeyError(f"No frame_names or ids in split file: {split_path}")


def build_alignment(dycheck_scene_dir, source_path, max_alignment_rmse, max_alignment_rot_deg):
    train_names = load_frame_names(dycheck_scene_dir / "splits" / "train.json")
    train_transforms = load_json(source_path / "transforms_train.json")["frames"]
    if len(train_names) != len(train_transforms):
        raise ValueError(
            f"Train split length {len(train_names)} does not match "
            f"transforms_train length {len(train_transforms)}"
        )

    raw_centers = []
    instant_centers = []
    rotation_errors = []
    for frame_name, instant_frame in zip(train_names, train_transforms):
        raw_camera = load_json(dycheck_scene_dir / "camera" / f"{frame_name}.json")
        raw_c2w = dycheck_c2w_from_camera(raw_camera)
        instant_c2w = instant_c2w_from_transform(instant_frame)
        raw_centers.append(raw_c2w[:3, 3])
        instant_centers.append(instant_c2w[:3, 3])

    scale, rotation, translation, center_errors = estimate_sim3(raw_centers, instant_centers)

    for frame_name, instant_frame in zip(train_names, train_transforms):
        raw_camera = load_json(dycheck_scene_dir / "camera" / f"{frame_name}.json")
        raw_c2w = dycheck_c2w_from_camera(raw_camera)
        instant_c2w = instant_c2w_from_transform(instant_frame)
        aligned_c2w = apply_sim3_to_c2w(raw_c2w, scale, rotation, translation)
        rel = aligned_c2w[:3, :3] @ instant_c2w[:3, :3].T
        cos_angle = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
        rotation_errors.append(math.degrees(math.acos(cos_angle)))

    rmse = float(np.sqrt(np.mean(center_errors * center_errors)))
    if rmse > max_alignment_rmse:
        raise RuntimeError(
            f"Train camera Sim(3) alignment RMSE {rmse:.6f} exceeds "
            f"--max_alignment_rmse {max_alignment_rmse:.6f}"
        )
    rot_p90 = float(np.percentile(rotation_errors, 90))
    if rot_p90 > max_alignment_rot_deg:
        raise RuntimeError(
            f"Train camera Sim(3) rotation p90 {rot_p90:.6f} deg exceeds "
            f"--max_alignment_rot_deg {max_alignment_rot_deg:.6f}"
        )

    return {
        "scale": float(scale),
        "rotation": rotation,
        "translation": translation,
        "center_error_rmse": rmse,
        "center_error_p50": float(np.percentile(center_errors, 50)),
        "center_error_p90": float(np.percentile(center_errors, 90)),
        "center_error_max": float(np.max(center_errors)),
        "rotation_error_deg_p50": float(np.percentile(rotation_errors, 50)),
        "rotation_error_deg_p90": float(np.percentile(rotation_errors, 90)),
        "rotation_error_deg_max": float(np.max(rotation_errors)),
        "num_train_pairs": len(train_names),
    }


def build_parser():
    parser = ArgumentParser(description="Render Instant4D on the official DyCheck validation split.")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dycheck_scene_dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--image_scale", default="2x")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--max_alignment_rmse", type=float, default=0.05)
    parser.add_argument("--max_alignment_rot_deg", type=float, default=10.0)
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
    pred_dir = output_dir / "rgb" / args.image_scale / args.split

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not dycheck_scene_dir.exists():
        raise FileNotFoundError(f"DyCheck scene directory not found: {dycheck_scene_dir}")
    if not source_path.exists():
        raise FileNotFoundError(f"Instant4D source path not found: {source_path}")

    val_names = load_frame_names(dycheck_scene_dir / "splits" / f"{args.split}.json")
    if args.max_frames > 0:
        val_names = val_names[: args.max_frames]

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

    rendered = []
    for index, frame_name in enumerate(val_names):
        camera_json = load_json(dycheck_scene_dir / "camera" / f"{frame_name}.json")
        raw_c2w = dycheck_c2w_from_camera(camera_json)
        aligned_c2w = apply_sim3_to_c2w(
            raw_c2w,
            alignment["scale"],
            alignment["rotation"],
            alignment["translation"],
        )
        timestamp = float(metadata[frame_name]["warp_id"]) / time_den * 3.0
        gt_path = dycheck_scene_dir / "rgb" / args.image_scale / f"{frame_name}.png"
        camera = camera_from_c2w(frame_name, aligned_c2w, camera_json, gt_path, timestamp, dataset.data_device)
        render_pkg = render(camera.cuda(), gaussians, pipe, background)
        Image.fromarray(tensor_to_uint8_image(render_pkg["render"])).save(pred_dir / f"{frame_name}.png")
        rendered.append(frame_name)
        if not args.quiet and (index + 1) % 25 == 0:
            print(f"Rendered {index + 1}/{len(val_names)} frames")

    manifest = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_iteration": int(checkpoint_iter),
        "dycheck_scene_dir": str(dycheck_scene_dir),
        "source_path": str(source_path),
        "split": args.split,
        "image_scale": args.image_scale,
        "output_dir": str(output_dir),
        "prediction_dir": str(pred_dir),
        "num_rendered": len(rendered),
        "rendered_frame_names": rendered,
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
    }
    with open(output_dir / "render_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if not args.quiet:
        print(f"Rendered {len(rendered)} official {args.split} frames to {pred_dir}")
        print(f"Alignment RMSE: {alignment['center_error_rmse']:.6f}")


if __name__ == "__main__":
    main()
