import numpy as np
import cv2
import json
from typing import NamedTuple
import os
import argparse
class pcd (NamedTuple):
    xyz: np.ndarray
    rgb: np.ndarray
    prob_motion: np.ndarray
    time_stamp: np.ndarray

def downsample_point_cloud_on_voxel_grid(voxel_size, xyz, *features):
    """Numpy replacement for point_cloud_utils voxel-grid downsampling."""
    if xyz.shape[0] == 0:
        return (xyz, *features)
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")

    keys = np.floor(xyz / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True)

    def voxel_mean(values):
        values = np.asarray(values)
        out = np.zeros((counts.shape[0],) + values.shape[1:],
                       dtype=np.float64)
        np.add.at(out, inverse, values)
        out /= counts.reshape((-1,) + (1,) * (values.ndim - 1))
        return out.astype(values.dtype, copy=False)

    return (voxel_mean(xyz), *(voxel_mean(feature) for feature in features))

def back_project(depth, intrinsic, cam_c2w):
    """
    Vectorized back-projection of depth maps to 3D points in world coordinates.
    
    Args:
        depth: B, H, W numpy array
        intrinsic: 3, 3 numpy array
        cam_c2w: B, 4, 4 numpy array
    
    Returns:
        xyz: B, H*W, 3 numpy array of 3D points in world coordinates
    """
    B, H, W = depth.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x = x.reshape(-1) + 0.5  # Add 0.5 for pixel center
    y = y.reshape(-1) + 0.5
    
    # Create homogeneous coordinates
    homogeneous_coords = np.vstack((x, y, np.ones_like(x)))
    
    # Apply inverse intrinsics
    cam_points = np.linalg.inv(intrinsic) @ homogeneous_coords  # 3 x (H*W)
    
    # Reshape depth and multiply
    depth_flat = depth.reshape(B, -1)  # B x (H*W)
    
    # Scale points by depth for each batch
    # Expand cam_points to B x 3 x (H*W)
    cam_points_expanded = np.tile(cam_points[None, :, :], (B, 1, 1))
    
    # Multiply by depth along the correct dimension
    cam_points_scaled = cam_points_expanded * depth_flat[:, None, :]  # B x 3 x (H*W)
    
    # Transform to world coordinates
    world_points = np.zeros((B, H*W, 3))
    for b in range(B):
        world_points[b] = (cam_points_scaled[b].T @ cam_c2w[b, :3, :3].T) + cam_c2w[b, :3, 3]
    
    return world_points

def read_droid_data(droid_path, motion_path, save_dir):
    droid_data = np.load(droid_path)
    print(droid_data.keys())

    print(droid_data['images'].shape)
    print(droid_data['depths'].shape)
    print(droid_data['intrinsic'].shape)
    print(droid_data['cam_c2w'].shape)

    color = droid_data['images'] # B, H, W, 3
    depth = droid_data['depths'] # B, H, W
    intrinsic = droid_data['intrinsic'] # 3, 3
    cam_c2w = droid_data['cam_c2w'] # B, 4, 4
    motion_prob = np.load(motion_path)
    # resize motion_prob to the same shape as color
    B = color.shape[0]
    H_new = color.shape[1]
    W_new = color.shape[2]
    if motion_prob.shape[0] != B:
        raise ValueError(
            f"motion_prob has {motion_prob.shape[0]} frames but CVD has {B}")
    
    
    resized_motion = np.empty((B, H_new, W_new), dtype=np.float32)
    
    for i in range(motion_prob.shape[0]):
        resized_motion[i] = cv2.resize(motion_prob[i], (W_new, H_new), interpolation=cv2.INTER_LINEAR)
        resized_motion[i] = resized_motion[i]



    print(f"motion_prob shape: {resized_motion.shape}")
    print(f"color shape: {color.shape}")
    print(f"depth shape: {depth.shape}")
    print(f"cam_c2w shape: {cam_c2w.shape}")
    print(f"intrinsic shape: {intrinsic.shape}")

    # color = np.concatenate([color[12:13], color[1:11], color[13:14]], axis=0)
    # depth = np.concatenate([depth[12:13], depth[1:11], depth[13:14]], axis=0)
    # cam_c2w = np.concatenate([cam_c2w[12:13], cam_c2w[1:11], cam_c2w[13:14]], axis=0)
    # motion_prob = np.concatenate([motion_prob[12:13], motion_prob[1:11], motion_prob[13:14]], axis=0)
    
    # # # save each of color into a png file
    # # for i in range(color.shape[0]):
    # #     color_resized = cv2.resize(color[i], (480, 270))
    # #     color_resized = color_resized[:, :, ::-1]
    # #     cv2.imwrite(f"{save_dir}/img/color_{i}.png", color_resized)
    
    

    return depth, color, resized_motion, intrinsic, cam_c2w

def load_split_frame_names(split_path):
    if split_path is None:
        return None
    with open(split_path, "r") as f:
        split = json.load(f)
    if "frame_names" in split:
        return list(split["frame_names"])
    if "ids" in split:
        return list(split["ids"])
    raise KeyError(f"No frame_names or ids in split file: {split_path}")

def frame_time_values(frame_names, fallback_count, time_scale=3.0,
                      denominator_frame_names=None):
    if frame_names is None:
        time_den = max(fallback_count - 1, 1)
        return np.arange(fallback_count, dtype=np.float32) / time_den * time_scale

    time_ids = np.array([int(name.split("_")[-1]) for name in frame_names],
                        dtype=np.float32)
    denom_names = (denominator_frame_names
                   if denominator_frame_names is not None
                   else frame_names)
    denom_ids = np.array([int(name.split("_")[-1]) for name in denom_names],
                         dtype=np.float32)
    time_den = max(float(denom_ids.max()), 1.0)
    return time_ids / time_den * time_scale

def process_data(depth, color, motion_prob, intrinsic, cam_c2w,
                 frame_times=None):
    B, H, W = depth.shape

    xyz = back_project(depth, intrinsic, cam_c2w).reshape(-1, 3)
    rgb = color.reshape(-1, 3).astype(np.float32)/255.0

    if frame_times is None:
        frame_times = np.arange(B).astype(np.float32) / B * 3
    frame_times = np.asarray(frame_times, dtype=np.float32)
    if frame_times.shape[0] != B:
        raise ValueError(f"frame_times length {frame_times.shape[0]} != B {B}")
    time_stamp = np.repeat(frame_times, xyz.shape[0]//B)
    time_stamp = time_stamp.reshape(-1, 1)
    
    prob_motion = motion_prob

    print(f"prob_motion range from {np.min(prob_motion)} to {np.max(prob_motion)}")
    print(f"prob_motion shape: {prob_motion.shape}")
    
    pc = pcd(xyz=xyz, rgb=rgb, prob_motion=prob_motion, time_stamp=time_stamp)
    
    return pc

def dynamic_static_split(pc, threshold=0.7):
    # this is a simpler version just use the threshold 0.5
    dynamic_region = (pc.prob_motion > 0.5).reshape(-1)
    static_region = ~dynamic_region
    
    print(f"shape of dynamic region: {dynamic_region.shape}")
    print(f"shape of static region: {static_region.shape}")
    print(f"shape of pc xyz: {pc.xyz.shape}")

    xyz_dynamic = pc.xyz[dynamic_region]
    rgb_dynamic = pc.rgb[dynamic_region]
    prob_motion_dynamic = pc.prob_motion[dynamic_region]
    time_stamp_dynamic = pc.time_stamp[dynamic_region]

    xyz_static = pc.xyz[static_region]
    rgb_static = pc.rgb[static_region]
    prob_motion_static = pc.prob_motion[static_region]
    time_stamp_static  = pc.time_stamp[static_region]
    
    dynamic_pcd = pcd(xyz=xyz_dynamic, rgb=rgb_dynamic, prob_motion=prob_motion_dynamic, time_stamp=time_stamp_dynamic)
    static_pcd =  pcd(xyz=xyz_static,  rgb=rgb_static,  prob_motion=prob_motion_static,  time_stamp=time_stamp_static)
    
    return dynamic_pcd, static_pcd

def make_transforms(intrinsic, cam_c2w, save_dir, scene, H, W,
                    train_frame_names=None, test_frame_names=None,
                    train_frame_times=None):
    scale_factor = 480/W
    B = cam_c2w.shape[0]
    print(f"cam_c2w: {cam_c2w.shape}")
    if train_frame_names is not None and len(train_frame_names) != B:
        raise ValueError(
            f"train_frame_names length {len(train_frame_names)} does not match "
            f"cam_c2w length {B}")

    dict_to_save = {}
    dict_to_save["w"]    = int(W * scale_factor)
    dict_to_save["h"]    = int(H * scale_factor)

    dict_to_save["fl_x"] = (intrinsic[0, 0] * scale_factor).item()
    dict_to_save["fl_y"] = (intrinsic[1, 1] * scale_factor).item()
    dict_to_save["cx"]   = (intrinsic[0, 2] * scale_factor).item()
    dict_to_save["cy"]   = (intrinsic[1, 2] * scale_factor).item()
    frame = []

    selected = range(B)

    print(f"selected_len: {len(selected)}")
    print(f"heldout_len: {0 if test_frame_names is None else len(test_frame_names)}")

    train_frame = []
    for i in selected:
        time_value = (train_frame_times[i].item()
                      if train_frame_times is not None
                      else i/max(B - 1, 1)*3)
        frame_dict = {
            "file_path": f"{scene}/{i:05d}",
            "transform_matrix": cam_c2w[i].tolist(),
            "time": time_value
        }
        if train_frame_names is not None:
            frame_dict["dycheck_frame_name"] = train_frame_names[i]
        train_frame.append(frame_dict)

    dict_to_save["frames"] = train_frame

    with open(f"{save_dir}/transforms_train.json", "w") as f:
        json.dump(dict_to_save, f, indent=4)
        
    # Held-out RoDyGS iPhone frames are intentionally not exported here:
    # Mega-SAM/CVD has only seen the train-only split, so there are no CVD poses
    # or point-cloud samples for test frames in this source directory.
    dict_to_save["frames"] = []

    with open(f"{save_dir}/transforms_test.json", "w") as f:
        json.dump(dict_to_save, f, indent=4)
    if test_frame_names is not None:
        with open(f"{save_dir}/rodygs_holdout_frames.json", "w") as f:
            json.dump({"frame_names": test_frame_names}, f, indent=4)
def export_source_images(color, save_dir, scene, H, W, image_output_dir=None):
    scale_factor = 480 / W
    out_w = int(W * scale_factor)
    out_h = int(H * scale_factor)
    image_dir = image_output_dir or os.path.join(save_dir, scene)
    os.makedirs(image_dir, exist_ok=True)

    for i in range(color.shape[0]):
        color_resized = cv2.resize(color[i], (out_w, out_h),
                                   interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(image_dir, f"{i:05d}.png"),
                    color_resized[:, :, ::-1])

    scene_link = os.path.join(save_dir, scene)
    if image_output_dir is None:
        return
    if os.path.islink(scene_link):
        os.unlink(scene_link)
    if os.path.exists(scene_link):
        if os.path.abspath(scene_link) != os.path.abspath(image_dir):
            raise FileExistsError(
                f"{scene_link} exists and is not the requested image link")
        return
    os.symlink(os.path.relpath(image_dir, save_dir), scene_link,
               target_is_directory=True)

def voxel_filter(droid_path, motion_path, save_dir, scene, use_mask=False,
                 prune_stride=3, image_output_dir=None,
                 train_split_path=None, test_split_path=None):
    depth, color, motion_prob, intrinsic, cam_c2w = read_droid_data(droid_path, motion_path, save_dir)

    B, H, W = depth.shape
    train_frame_names = load_split_frame_names(train_split_path)
    test_frame_names = load_split_frame_names(test_split_path)
    if train_frame_names is not None and len(train_frame_names) != B:
        raise ValueError(
            f"train split has {len(train_frame_names)} frames but CVD has {B}")
    time_denominator_names = train_frame_names
    if train_frame_names is not None and test_frame_names is not None:
        time_denominator_names = train_frame_names + test_frame_names
    train_frame_times = frame_time_values(
        train_frame_names, B, denominator_frame_names=time_denominator_names)

    print(f"depth shape: {depth.shape}")
    make_transforms(intrinsic, cam_c2w, save_dir, scene, H, W,
                    train_frame_names=train_frame_names,
                    test_frame_names=test_frame_names,
                    train_frame_times=train_frame_times)
    export_source_images(color, save_dir, scene, H, W, image_output_dir)
    
    # select every 10th frame 
    # n H W 
    color = color[::prune_stride]
    depth = depth[::prune_stride]
    cam_c2w = cam_c2w[::prune_stride]
    motion_prob = motion_prob[::prune_stride]
    train_frame_times = train_frame_times[::prune_stride]
    
    motion_prob = motion_prob.reshape(-1, 1).astype(np.float32)
    
    print(f"motion_prob shape: {motion_prob.shape}")
    print(f"color shape: {color.shape}")
    print(f"depth shape: {depth.shape}")
    print(f"cam_c2w shape: {cam_c2w.shape}")
    

    
    pc = process_data(depth, color, motion_prob, intrinsic, cam_c2w,
                      frame_times=train_frame_times)
    pcd_dynamic, pcd_static = dynamic_static_split(pc)
    
    mean_depth = np.mean(depth[0])
    focal = intrinsic[0, 0]

    # voxel size for dynamic region
    voxel_size_dynamic = mean_depth / focal * 0.5
    voxel_size_static  = mean_depth / focal * 2
    
    xyz_static, rgb_static, prob_motion_static = downsample_point_cloud_on_voxel_grid(
        voxel_size_static, pcd_static.xyz, pcd_static.rgb, pcd_static.prob_motion)
    
    xyz_dynamic, rgb_dynamic, prob_motion_dynamic, time_stamp_dynamic = downsample_point_cloud_on_voxel_grid(
        voxel_size_dynamic, pcd_dynamic.xyz, pcd_dynamic.rgb,
        pcd_dynamic.prob_motion, pcd_dynamic.time_stamp)
    
    

    
    time_stamp_static = np.repeat(1, xyz_static.shape[0])
    scale_time_static = np.repeat(3, xyz_static.shape[0])
    if train_frame_times.shape[0] > 1:
        dynamic_time_step = float(np.median(np.diff(np.sort(train_frame_times))))
    else:
        dynamic_time_step = 3 / max(B - 1, 1)
    scale_time_dynamic = np.repeat(dynamic_time_step / 10, xyz_dynamic.shape[0])

    xyz_sampled = np.concatenate([xyz_static, xyz_dynamic], axis=0)
    rgb_sampled = np.concatenate([rgb_static, rgb_dynamic], axis=0)
    prob_motion_sampled = np.concatenate([prob_motion_static.squeeze(), prob_motion_dynamic.squeeze()], axis=0)
    time_stamp_sampled =  np.concatenate([time_stamp_static.squeeze(),  time_stamp_dynamic.squeeze()], axis=0)
    scale_time_sampled = np.concatenate([scale_time_static, scale_time_dynamic], axis=0)
    # xyz_sampled = xyz_static
    # rgb_sampled = rgb_static
    # prob_motion_sampled = prob_motion_static
    # time_stamp_sampled = time_stamp_static
    # scale_time_sampled = scale_time_static
    
    print(f"--------------------------------")
    print(f"Scene: {scene}")
    print(f"xyz_static: {xyz_static.shape}")
    print(f"xyz_dynamic: {pcd_dynamic.xyz.shape}")
    print(f"xyz_sampled: {xyz_sampled.shape}")
    print(f"time_stamp: {time_stamp_sampled.shape}")
    print(f"prob_motion: {prob_motion_sampled.shape}")
    print(f"scale_time: {scale_time_sampled.shape}")
    np.savez(f"{save_dir}/filtered_cvd.npz", 
            xyz=xyz_sampled,
            rgb=rgb_sampled,
            prob_motion=prob_motion_sampled,
            time_stamp=time_stamp_sampled,
            scale_time=scale_time_sampled,
            intrinsic=intrinsic,
            cam_c2w=cam_c2w)
    manifest = {
        "scene": scene,
        "train_split_path": train_split_path,
        "test_split_path": test_split_path,
        "num_train_frames": B,
        "num_pointcloud_frames": int(depth.shape[0]),
        "prune_stride": prune_stride,
        "train_frame_names": train_frame_names,
        "heldout_frame_names": test_frame_names,
    }
    with open(f"{save_dir}/source_manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--droid_path", required=True,
                        help="path to <scene>_sgd_cvd_hr.npz")
    parser.add_argument("--motion_path", required=True,
                        help="path to motion_prob.npy")
    parser.add_argument("--save_dir", required=True,
                        help="output directory for filtered_cvd.npz and transforms")
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--prune_stride", "--stride", type=int, default=3,
                        help="frame stride for point-cloud initialization")
    parser.add_argument("--image_output_dir", default=None,
                        help="optional real image export dir; save_dir/scene_name becomes a symlink")
    parser.add_argument("--train_split_path", default=None,
                        help="DyCheck/RoDyGS train-only split used to create the CVD input")
    parser.add_argument("--test_split_path", default=None,
                        help="DyCheck/RoDyGS held-out split recorded for evaluation")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    voxel_filter(args.droid_path, args.motion_path, args.save_dir,
                 args.scene_name, use_mask=False,
                 prune_stride=args.prune_stride,
                 image_output_dir=args.image_output_dir,
                 train_split_path=args.train_split_path,
                 test_split_path=args.test_split_path)
