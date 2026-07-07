import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from arguments import ModelParams, OptimizationParams, PipelineParams
from scene.gaussian_model import GaussianModel
from utils.traj_loss import TrajLoss


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


def build_parser():
    parser = ArgumentParser(description="Evaluate CoTracker trajectory reprojection error for Instant4D.")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--assoc", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--eval_batch", type=int, default=8192)
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
    output_json = Path(args.output_json)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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
    model_params, checkpoint_iter = torch.load(str(checkpoint_path), map_location="cuda", weights_only=False)
    gaussians.restore(model_params, None)
    del model_params
    torch.cuda.empty_cache()

    traj_eval = TrajLoss(
        assoc_path=args.assoc,
        anchors_path=args.anchors,
        tracks_path=args.tracks,
        source_path=dataset.source_path,
        batch_size=args.eval_batch,
        device="cuda",
    )
    metrics = traj_eval.evaluate(gaussians, batch_size=args.eval_batch)
    metrics.update(
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_iteration": int(checkpoint_iter),
            "tracks": str(args.tracks),
            "anchors": str(args.anchors),
            "assoc": str(args.assoc),
            "source_path": dataset.source_path,
        }
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    if not args.quiet:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
