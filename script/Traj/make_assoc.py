import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from utils.general_utils import knn


def build_parser():
    parser = ArgumentParser(description="Build fixed GS associations for CoTracker trajectory anchors.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_geometry", type=float, default=0.5)
    parser.add_argument("--min_motion", type=float, default=0.5)
    parser.add_argument("--min_adjacent_pairs", type=int, default=2)
    parser.add_argument("--max_anchors", type=int, default=4000)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--query_chunk", type=int, default=1024)
    return parser


def select_anchors(anchors, min_geometry, min_motion, min_adjacent_pairs, max_anchors):
    geometry = anchors["geometry_match_score"].astype(np.float32)
    motion = anchors["point_prob_motion"].astype(np.float32)
    adjacent = anchors["segment_adjacent_pairs"].astype(np.int32)
    xyz = anchors["anchor_xyz"].astype(np.float32)
    score = geometry * motion

    keep = (
        np.isfinite(xyz).all(axis=1)
        & np.isfinite(score)
        & (geometry >= min_geometry)
        & (motion >= min_motion)
        & (adjacent >= min_adjacent_pairs)
    )
    selected = np.where(keep)[0]
    if selected.size == 0:
        raise RuntimeError("No anchors survived the association filters")

    order = np.argsort(score[selected])[::-1]
    selected = selected[order]
    if max_anchors > 0:
        selected = selected[:max_anchors]
    return selected.astype(np.int64), score[selected].astype(np.float32)


def main():
    args = build_parser().parse_args()
    checkpoint_path = Path(args.checkpoint)
    anchors_path = Path(args.anchors)
    output_path = Path(args.output)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not anchors_path.exists():
        raise FileNotFoundError(f"Anchor file not found: {anchors_path}")

    anchors = np.load(anchors_path)
    anchor_id, traj_score = select_anchors(
        anchors,
        args.min_geometry,
        args.min_motion,
        args.min_adjacent_pairs,
        args.max_anchors,
    )
    query_np = anchors["anchor_xyz"][anchor_id].astype(np.float32)

    model_params, checkpoint_iter = torch.load(str(checkpoint_path), map_location="cuda", weights_only=False)
    xyz = model_params[1].detach().float().contiguous().cuda()
    del model_params
    torch.cuda.empty_cache()
    src = xyz[None].contiguous()

    all_idx = []
    all_dist = []
    for start in range(0, len(query_np), args.query_chunk):
        query = torch.from_numpy(query_np[start : start + args.query_chunk]).float().cuda()[None].contiguous()
        idx, dist = knn(query, src, args.k)
        all_idx.append(idx[0].detach().cpu().numpy())
        all_dist.append(dist[0].detach().cpu().numpy())
        del query, idx, dist
        torch.cuda.empty_cache()

    gs_id = np.concatenate(all_idx, axis=0).astype(np.int64)
    knn_dist = np.concatenate(all_dist, axis=0).astype(np.float32)
    dist = np.sqrt(np.maximum(knn_dist, 0.0))
    idw_w = 1.0 / (dist + 1e-6)
    idw_w = (idw_w / idw_w.sum(axis=1, keepdims=True)).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_iteration": int(checkpoint_iter),
        "anchors": str(anchors_path),
        "num_input_anchors": int(len(anchors["anchor_xyz"])),
        "num_selected_anchors": int(len(anchor_id)),
        "k": int(args.k),
        "min_geometry": float(args.min_geometry),
        "min_motion": float(args.min_motion),
        "min_adjacent_pairs": int(args.min_adjacent_pairs),
        "max_anchors": int(args.max_anchors),
    }
    np.savez(
        output_path,
        anchor_id=anchor_id,
        gs_id=gs_id,
        idw_w=idw_w,
        knn_dist=knn_dist,
        traj_score=traj_score,
        anchor_xyz=query_np,
        metadata_json=np.array(json.dumps(meta, indent=2)),
    )
    print(json.dumps(meta, indent=2))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
