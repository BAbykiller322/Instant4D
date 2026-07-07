import json
import os
import sys
import types
import importlib.util
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_frame_names(split_path):
    split = load_json(split_path)
    if "frame_names" in split:
        return list(split["frame_names"])
    if "ids" in split:
        return list(split["ids"])
    raise KeyError(f"No frame_names or ids in split file: {split_path}")


def load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def load_mask(path):
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return (mask > 0.5).astype(np.float32)[..., None]


def _masked_mean(x, mask=None):
    import jax.numpy as jnp

    eps = 1e-6
    if mask is None:
        return x.mean()
    broadcast_to = jnp.broadcast_to if isinstance(x, jnp.ndarray) else np.broadcast_to
    mask = broadcast_to(mask, x.shape)
    return (x * mask).sum() / mask.sum().clip(eps)


def _install_metrics_stub():
    dycheck_mod = types.ModuleType("dycheck")
    nn_mod = types.ModuleType("dycheck.nn")
    functional_mod = types.ModuleType("dycheck.nn.functional")
    functional_mod.common = types.SimpleNamespace(masked_mean=_masked_mean)
    nn_mod.functional = functional_mod
    dycheck_mod.nn = nn_mod
    sys.modules["dycheck"] = dycheck_mod
    sys.modules["dycheck.nn"] = nn_mod
    sys.modules["dycheck.nn.functional"] = functional_mod


def _find_official_image_metrics(dycheck_code_root):
    candidates = []
    if dycheck_code_root:
        candidates.append(Path(dycheck_code_root))
    if os.environ.get("DYCHECK_CODE_ROOT"):
        candidates.append(Path(os.environ["DYCHECK_CODE_ROOT"]))
    for item in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if item:
            candidates.append(Path(item))
    candidates.append(Path.home() / "dycheck")

    for root in candidates:
        path = root / "dycheck" / "core" / "metrics" / "image.py"
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find KAIR-BAIR/dycheck image metrics. Set --dycheck_code_root "
        "or DYCHECK_CODE_ROOT to the official dycheck repo."
    )


def require_official_metrics(dycheck_code_root=None):
    try:
        import jax.numpy as jnp

        metrics_path = _find_official_image_metrics(dycheck_code_root)
        _install_metrics_stub()
        spec = importlib.util.spec_from_file_location("dycheck_official_image_metrics", metrics_path)
        metrics = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metrics)
    except Exception as exc:
        raise ImportError(
            "Official DyCheck image metrics are required. Set --dycheck_code_root "
            "to the KAIR-BAIR/dycheck repo and install jax/lpips dependencies."
        ) from exc
    return jnp, metrics


def scalar(value):
    return float(np.asarray(value))


def build_parser():
    parser = ArgumentParser(description="Evaluate Instant4D predictions with official DyCheck masked metrics.")
    parser.add_argument("--dycheck_scene_dir")
    parser.add_argument("--pred_dir")
    parser.add_argument("--split", default="val")
    parser.add_argument("--image_scale", default="2x")
    parser.add_argument("--output_json")
    parser.add_argument("--dycheck_code_root", default=os.environ.get("DYCHECK_CODE_ROOT", ""))
    parser.add_argument("--check_metrics_import", action="store_true")
    parser.add_argument(
        "--allow_missing_predictions",
        action="store_true",
        help="Only for smoke tests. Full official evaluation should leave this disabled.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    jnp, metrics = require_official_metrics(args.dycheck_code_root)
    if args.check_metrics_import:
        metrics.get_compute_lpips()
        print("Official DyCheck image metrics import OK")
        return

    if not args.dycheck_scene_dir or not args.pred_dir or not args.output_json:
        raise ValueError("--dycheck_scene_dir, --pred_dir, and --output_json are required for evaluation")

    dycheck_scene_dir = Path(args.dycheck_scene_dir)
    pred_dir = Path(args.pred_dir)
    output_json = Path(args.output_json)
    output_dir = output_json.parent
    per_frame_path = output_dir / "metrics_per_frame.json"

    compute_lpips = metrics.get_compute_lpips()

    frame_names = load_frame_names(dycheck_scene_dir / "splits" / f"{args.split}.json")
    gt_dir = dycheck_scene_dir / "rgb" / args.image_scale
    mask_dir = dycheck_scene_dir / "covisible" / args.image_scale / args.split

    if not mask_dir.exists():
        raise FileNotFoundError(
            f"Official covisibility mask directory not found: {mask_dir}. "
            "Generate it with KAIR-BAIR/dycheck tools/process_covisible.py first."
        )

    missing_predictions = []
    missing_masks = []
    missing_gt = []
    frame_metrics = []

    for frame_name in frame_names:
        pred_path = pred_dir / f"{frame_name}.png"
        gt_path = gt_dir / f"{frame_name}.png"
        mask_path = mask_dir / f"{frame_name}.png"

        if not pred_path.exists():
            missing_predictions.append(frame_name)
            continue
        if not gt_path.exists():
            missing_gt.append(frame_name)
            continue
        if not mask_path.exists():
            missing_masks.append(frame_name)
            continue

        pred = load_rgb(pred_path)
        gt = load_rgb(gt_path)
        mask = load_mask(mask_path)
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch for {frame_name}: pred {pred.shape}, gt {gt.shape}")
        if mask.shape[:2] != gt.shape[:2]:
            raise ValueError(f"Mask shape mismatch for {frame_name}: mask {mask.shape}, gt {gt.shape}")

        pred_jnp = jnp.asarray(pred)
        gt_jnp = jnp.asarray(gt)
        mask_jnp = jnp.asarray(mask)
        item = {
            "frame_name": frame_name,
            "mPSNR": scalar(metrics.compute_psnr(pred_jnp, gt_jnp, mask_jnp)),
            "mSSIM": scalar(metrics.compute_ssim(pred_jnp, gt_jnp, mask_jnp)),
            "mLPIPS": scalar(compute_lpips(pred_jnp, gt_jnp, mask_jnp)),
        }
        frame_metrics.append(item)

    if missing_gt:
        raise FileNotFoundError(f"Missing ground-truth images for {len(missing_gt)} frames: {missing_gt[:10]}")
    if missing_masks:
        raise FileNotFoundError(f"Missing covisibility masks for {len(missing_masks)} frames: {missing_masks[:10]}")
    if missing_predictions and not args.allow_missing_predictions:
        raise FileNotFoundError(
            f"Missing predictions for {len(missing_predictions)} frames: {missing_predictions[:10]}. "
            "Use --allow_missing_predictions only for smoke tests."
        )
    if not frame_metrics:
        raise RuntimeError("No frames were evaluated")

    summary = {
        "mPSNR": float(np.mean([x["mPSNR"] for x in frame_metrics])),
        "mSSIM": float(np.mean([x["mSSIM"] for x in frame_metrics])),
        "mLPIPS": float(np.mean([x["mLPIPS"] for x in frame_metrics])),
        "num_frames": len(frame_names),
        "num_evaluated_frames": len(frame_metrics),
        "num_missing_predictions": len(missing_predictions),
        "num_missing_masks": len(missing_masks),
        "num_missing_gt": len(missing_gt),
        "split": args.split,
        "image_scale": args.image_scale,
        "prediction_dir": str(pred_dir),
        "dycheck_scene_dir": str(dycheck_scene_dir),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(per_frame_path, "w", encoding="utf-8") as f:
        json.dump(frame_metrics, f, indent=2)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
