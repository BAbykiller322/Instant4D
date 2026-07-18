import argparse
import json
import os

import cv2
import numpy as np


def skew(v):
    return np.array(
        [[0.0, -v[2], v[1]],
         [v[2], 0.0, -v[0]],
         [-v[1], v[0], 0.0]],
        dtype=np.float64)


def resize_depth(depth, height, width):
    return cv2.resize(
        depth.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)


def scaled_intrinsic(intrinsic, source_height, source_width, target_height, target_width):
    k = intrinsic.astype(np.float64).copy()
    k[0, :] *= float(target_width) / float(source_width)
    k[1, :] *= float(target_height) / float(source_height)
    return k


def project_rigid_flow(depth, intrinsic, c2w_i, c2w_j):
    height, width = depth.shape
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32),
                         np.arange(height, dtype=np.float32))
    x = xs.reshape(-1) + 0.5
    y = ys.reshape(-1) + 0.5
    z = depth.reshape(-1).astype(np.float64)

    pix = np.stack([x, y, np.ones_like(x)], axis=0).astype(np.float64)
    cam_i = np.linalg.inv(intrinsic) @ pix
    cam_i *= z[None, :]

    world = cam_i.T @ c2w_i[:3, :3].T + c2w_i[:3, 3]
    w2c_j = np.linalg.inv(c2w_j)
    cam_j = world @ w2c_j[:3, :3].T + w2c_j[:3, 3]
    z_j = cam_j[:, 2]

    proj = cam_j / np.maximum(z_j[:, None], 1e-8)
    uv_j = proj @ intrinsic.T
    uv_j = uv_j[:, :2].reshape(height, width, 2).astype(np.float32)

    valid = (
        (z.reshape(height, width) > 1e-6)
        & (z_j.reshape(height, width) > 1e-6)
        & (uv_j[..., 0] >= 0.0)
        & (uv_j[..., 0] < width)
        & (uv_j[..., 1] >= 0.0)
        & (uv_j[..., 1] < height)
    )
    grid_center = np.stack([xs + 0.5, ys + 0.5], axis=-1)
    return uv_j - grid_center, valid


def sampson_residual(flow, intrinsic, c2w_i, c2w_j):
    height, width = flow.shape[:2]
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float64),
                         np.arange(height, dtype=np.float64))
    p0 = np.stack([xs.reshape(-1) + 0.5,
                   ys.reshape(-1) + 0.5,
                   np.ones(height * width)], axis=0)
    p1 = np.stack([(xs + flow[..., 0]).reshape(-1) + 0.5,
                   (ys + flow[..., 1]).reshape(-1) + 0.5,
                   np.ones(height * width)], axis=0)

    k_inv = np.linalg.inv(intrinsic)
    x0 = k_inv @ p0
    x1 = k_inv @ p1

    t_j_i = np.linalg.inv(c2w_j) @ c2w_i
    r = t_j_i[:3, :3]
    t = t_j_i[:3, 3]
    e = skew(t) @ r

    ex0 = e @ x0
    etx1 = e.T @ x1
    x1ex0 = np.sum(x1 * ex0, axis=0)
    denom = ex0[0] ** 2 + ex0[1] ** 2 + etx1[0] ** 2 + etx1[1] ** 2 + 1e-12
    residual = np.abs(x1ex0) / np.sqrt(denom)
    focal = 0.5 * (intrinsic[0, 0] + intrinsic[1, 1])
    return (residual * focal).reshape(height, width).astype(np.float32)


def clean_mask(mask, min_area, max_area_ratio, close_radius, open_radius):
    mask_u8 = mask.astype(np.uint8)
    if open_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * open_radius + 1, 2 * open_radius + 1))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    if close_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * close_radius + 1, 2 * close_radius + 1))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    out = np.zeros_like(mask_u8)
    max_area = max_area_ratio * mask_u8.size
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area and area <= max_area:
            out[labels == label] = 1
    return out.astype(bool)


def save_debug_overlay(debug_dir, frame_idx, rgb, mask):
    os.makedirs(debug_dir, exist_ok=True)
    image = rgb.copy()
    overlay = image.copy()
    overlay[mask] = np.array([255, 0, 0], dtype=np.uint8)
    blended = (0.65 * image + 0.35 * overlay).astype(np.uint8)
    cv2.imwrite(
        os.path.join(debug_dir, f"{frame_idx:05d}_temporal_motion.png"),
        blended[:, :, ::-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvd_path", required=True,
                        help="path to <scene>_sgd_cvd_hr.npz")
    parser.add_argument("--flow_dir", required=True,
                        help="directory containing flows.npy, flows_masks.npy, ii-jj.npy")
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--score_out_path", default=None)
    parser.add_argument("--debug_dir", default=None)
    parser.add_argument("--steps", default="1",
                        help="comma-separated flow time offsets to use")
    parser.add_argument("--reproj_thresh", type=float, default=3.0)
    parser.add_argument("--sampson_thresh", type=float, default=1.5)
    parser.add_argument("--min_component_area", type=int, default=30)
    parser.add_argument("--max_component_area_ratio", type=float, default=0.08)
    parser.add_argument("--open_radius", type=int, default=1)
    parser.add_argument("--close_radius", type=int, default=2)
    parser.add_argument("--debug_frames", default="0,120,240,360,438")
    args = parser.parse_args()

    cvd = np.load(args.cvd_path, allow_pickle=True)
    images = cvd["images"]
    depths = cvd["depths"]
    intrinsic = cvd["intrinsic"]
    cam_c2w = cvd["cam_c2w"]
    num_frames, src_h, src_w = depths.shape

    flows = np.load(os.path.join(args.flow_dir, "flows.npy"), mmap_mode="r")
    flow_masks = np.load(os.path.join(args.flow_dir, "flows_masks.npy"),
                         mmap_mode="r")
    ii_jj = np.load(os.path.join(args.flow_dir, "ii-jj.npy"))
    use_steps = {int(step) for step in args.steps.split(",") if step.strip()}

    flow_h, flow_w = flows.shape[-2:]
    k_flow = scaled_intrinsic(intrinsic, src_h, src_w, flow_h, flow_w)
    mask_acc = np.zeros((num_frames, flow_h, flow_w), dtype=bool)
    score_acc = np.zeros((num_frames, flow_h, flow_w), dtype=np.float32)
    counts = np.zeros(num_frames, dtype=np.int32)

    for pair_idx in range(flows.shape[0]):
        i = int(ii_jj[0, pair_idx])
        j = int(ii_jj[1, pair_idx])
        step = j - i
        if step not in use_steps or i < 0 or j >= num_frames:
            continue

        flow = flows[pair_idx].astype(np.float32).transpose(1, 2, 0)
        fb_valid = flow_masks[pair_idx, 0].astype(bool)
        depth_i = resize_depth(depths[i], flow_h, flow_w)
        rigid_flow, rigid_valid = project_rigid_flow(
            depth_i, k_flow, cam_c2w[i].astype(np.float64),
            cam_c2w[j].astype(np.float64))
        reproj = np.linalg.norm(flow - rigid_flow, axis=-1)
        sampson = sampson_residual(
            flow, k_flow, cam_c2w[i].astype(np.float64),
            cam_c2w[j].astype(np.float64))

        valid = fb_valid & rigid_valid
        raw = valid & (reproj > args.reproj_thresh) & (sampson > args.sampson_thresh)
        cleaned = clean_mask(
            raw,
            args.min_component_area,
            args.max_component_area_ratio,
            args.close_radius,
            args.open_radius)
        score = np.maximum(
            reproj / max(args.reproj_thresh, 1e-6),
            sampson / max(args.sampson_thresh, 1e-6))
        score[~valid] = 0.0

        mask_acc[i] |= cleaned
        score_acc[i] = np.maximum(score_acc[i], score.astype(np.float32))
        counts[i] += 1

    full_mask = np.empty((num_frames, src_h, src_w), dtype=np.uint8)
    for i in range(num_frames):
        full_mask[i] = cv2.resize(
            mask_acc[i].astype(np.uint8), (src_w, src_h),
            interpolation=cv2.INTER_NEAREST)

    np.save(args.out_path, full_mask)
    if args.score_out_path:
        full_score = np.empty((num_frames, src_h, src_w), dtype=np.float16)
        for i in range(num_frames):
            full_score[i] = cv2.resize(
                score_acc[i].astype(np.float32), (src_w, src_h),
                interpolation=cv2.INTER_LINEAR).astype(np.float16)
        np.save(args.score_out_path, full_score)

    debug_frames = [int(x) for x in args.debug_frames.split(",") if x.strip()]
    if args.debug_dir:
        for frame_idx in debug_frames:
            if 0 <= frame_idx < num_frames:
                save_debug_overlay(
                    args.debug_dir, frame_idx, images[frame_idx],
                    full_mask[frame_idx].astype(bool))

    summary = {
        "cvd_path": args.cvd_path,
        "flow_dir": args.flow_dir,
        "out_path": args.out_path,
        "shape": list(full_mask.shape),
        "used_steps": sorted(use_steps),
        "reproj_thresh": args.reproj_thresh,
        "sampson_thresh": args.sampson_thresh,
        "min_component_area": args.min_component_area,
        "max_component_area_ratio": args.max_component_area_ratio,
        "mask_ratio": float(full_mask.astype(bool).mean()),
        "frames_with_pairs": int((counts > 0).sum()),
        "per_frame_mask_ratio_p50": float(np.percentile(full_mask.reshape(num_frames, -1).mean(axis=1), 50)),
        "per_frame_mask_ratio_p90": float(np.percentile(full_mask.reshape(num_frames, -1).mean(axis=1), 90)),
    }
    with open(os.path.splitext(args.out_path)[0] + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
