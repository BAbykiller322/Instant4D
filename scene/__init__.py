#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, 
# research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.data_utils import CameraDataset
import numpy as np
from PIL import Image
from utils.loss_utils import l1_loss, ssim, msssim
from utils.image_utils import psnr, easy_cmap

class Scene:

    gaussians : GaussianModel
    evaluation_metrics = {}

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], num_pts=100_000, num_pts_ratio=1.0, time_duration=None, initialize_gaussians=True):
        """b
        :param path: Path to colmap scene main folder.
        """
        
        self.evaluation_metrics["psnr"] = []
        self.evaluation_metrics["ssim"] = []
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.white_background = args.white_background

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        print(f"------args.source_path: {args.source_path}")
        print((os.path.join(args.source_path, "transforms_train.json")))

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, num_pts_ratio=num_pts_ratio)
        elif os.path.exists(os.path.join(args.source_path, "filtered_cvd.npz")):
            print("Found filter.npz file, assuming Mega Sam data set!")
            scene_info = sceneLoadTypeCallbacks["MegaSam"](args.source_path, args.white_background, args.eval, num_pts=num_pts, time_duration=time_duration, extension=args.extension, num_extra_pts=args.num_extra_pts, frame_ratio=args.frame_ratio, dataloader=args.dataloader)
        elif os.path.exists(os.path.join(args.source_path, "points3d.ply")):
            print("Found points_3d.ply file, assuming Cuter data set!")
            scene_info = sceneLoadTypeCallbacks["Cuter"](args.source_path, args.white_background, args.eval, num_pts=num_pts, time_duration=time_duration, extension=args.extension, num_extra_pts=args.num_extra_pts, frame_ratio=args.frame_ratio, dataloader=args.dataloader)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, num_pts=num_pts, time_duration=time_duration, extension=args.extension, num_extra_pts=args.num_extra_pts, frame_ratio=args.frame_ratio, dataloader=args.dataloader)
        else:
            print(f"------args.source_path: {args.source_path}")
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            # with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
            #     dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            
        if initialize_gaussians:
            if args.loaded_pth:
                self.gaussians.create_from_pth(args.loaded_pth, self.cameras_extent)
            else:
                if self.loaded_iter:
                    self.gaussians.load_ply(os.path.join(self.model_path,
                                                                "point_cloud",
                                                                "iteration_" + str(self.loaded_iter),
                                                                "point_cloud.ply"))
                else:
                    self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        torch.save((self.gaussians.capture(), iteration), self.model_path + "/chkpnt" + str(iteration) + ".pth")

    def getTrainCameras(self, scale=1.0):
        return CameraDataset(self.train_cameras[scale].copy(), self.white_background)
        
    def getTestCameras(self, scale=1.0):
        return CameraDataset(self.test_cameras[scale].copy(), self.white_background)
    
 
    def render_evaluate_sora(self, path, gaussians, pipe, background):
        os.makedirs(os.path.join(path, "test"), exist_ok=True)  
        
        from gaussian_renderer import render
        from copy import deepcopy
        import math
        import cv2
        
        total_frames = len(self.test_cameras[1.0])
        half_frames = total_frames // 2
        movement_amplitude = 0.05  # Small movement amplitude
        fps = 30  # Video frame rate
        
        # Define movement variants - testing multiple wobble factors
        wobble_factors = [0.05, 0.1, 0.25, 0.5, 1, 2]
        movement_variants = []
        for wobble_factor in wobble_factors:
            movement_variants.append({"name": f"wobble_factor_{wobble_factor}", "movements": ["wobble"], "wobble_factor": wobble_factor})
        
        # movement_variants = [
        #     {"name": "forward_circular", "movements": ["forward_back", "circular"]},
        #     {"name": "lateral_vertical", "movements": ["lateral", "vertical"]},
        #     {"name": "wobble", "movements": ["wobble"]}
        # ]
        
        # Sort test cameras by timestamp to ensure proper temporal ordering
        sorted_test_cameras = sorted(self.test_cameras[1.0], key=lambda cam: cam.timestamp)
        
        # Render first half (original views) once
        first_half_frames = []
        print("Rendering first half (original views)...")
        for idx in range(half_frames):
            cam = sorted_test_cameras[idx]
            original_cam = cam.cuda()
            
            render_pkg = render(original_cam, gaussians, pipe, background)
            image = render_pkg["render"]
            
            image = image.cpu().numpy()
            image = image.transpose(1, 2, 0)
            image = np.clip(image, 0, 1)
            image = (image * 255).astype(np.uint8)
            first_half_frames.append(image)
        
        # Render second half (original views) once
        second_half_frames = []
        print("Rendering second half (original views)...")
        for idx in range(half_frames, total_frames):
            cam = sorted_test_cameras[idx]
            original_cam = cam.cuda()
            
            render_pkg = render(original_cam, gaussians, pipe, background)
            image = render_pkg["render"]
            
            image = image.cpu().numpy()
            image = image.transpose(1, 2, 0)
            image = np.clip(image, 0, 1)
            image = (image * 255).astype(np.uint8)
            second_half_frames.append(image)
        
        # Get middle timestamp camera as base for movements
        middle_idx = total_frames // 2
        base_cam = sorted_test_cameras[middle_idx]
            
        def generate_wobble_transformation(radius, t_vals):
            """Generate wobble transformation matrices"""
            tf_matrices = []
            for t in t_vals:
                # Create 4x4 identity matrix
                tf = torch.eye(4, dtype=torch.float32)
                # Convert t to tensor for torch operations
                t_tensor = torch.tensor(t, dtype=torch.float32)
                # Add wobble translation in image plane (x, y axes)
                tf[0, 3] = torch.sin(2 * math.pi * t_tensor) * radius  # x translation
                tf[1, 3] = -torch.cos(2 * math.pi * t_tensor) * radius  # y translation
                tf_matrices.append(tf)
            return torch.stack(tf_matrices)
        
        # Create combined movement videos for each wobble factor
        for variant in movement_variants:
            print(f"Creating video for {variant['name']} movement...")
            current_wobble_factor = variant['wobble_factor']
            
            # Calculate wobble radius for this variant
            if len(sorted_test_cameras) >= 2:
                # Get camera positions from first two cameras
                cam_0_pos = torch.tensor(sorted_test_cameras[0].camera_center[:3])
                cam_1_pos = torch.tensor(sorted_test_cameras[1].camera_center[:3])
                delta = 0.1
                calculated_radius = delta * current_wobble_factor
                
                # Ensure minimum radius for visible movement
                min_radius = 0.01  # Reduced minimum for smaller wobble factors
                wobble_radius = max(calculated_radius, min_radius)
                
                print(f"Wobble factor: {current_wobble_factor}")
                print(f"Camera distance delta: {delta:.6f}")
                print(f"Calculated wobble radius: {calculated_radius:.6f}")  
                print(f"Final wobble radius (with min {min_radius}): {wobble_radius:.6f}")
            else:
                wobble_radius = 0.1 * current_wobble_factor  # Scale fallback by wobble factor
                print(f"Using fallback wobble radius: {wobble_radius:.4f}")
            
            # Collect all frames for this video
            video_frames = []
            
            # Add first half frames
            video_frames.extend(first_half_frames)
            
            # Generate combined movement frames
            combined_movement_frames = []
            num_movement_frames = 60  # Number of frames for each movement sequence
            
            base_cam_cuda = base_cam.cuda()
            
            # Process each movement in the variant sequentially
            for movement_name in variant['movements']:
                for frame_idx in range(num_movement_frames):
                    moved_cam = deepcopy(base_cam_cuda)
                    
                    # Create a smooth movement pattern: 0 -> max -> 0
                    t = frame_idx / (num_movement_frames - 1)  # 0 to 1
                    movement_factor = math.sin(t * math.pi)  # 0 -> 1 -> 0 smooth curve
                    
                    if movement_name == "circular":  # Circular orbit
                        angle = t * 2 * math.pi  # Complete circle
                        offset_x = movement_amplitude * movement_factor * math.cos(angle)
                        offset_z = movement_amplitude * movement_factor * math.sin(angle)
                        moved_cam.T[0] += offset_x
                        moved_cam.T[2] += offset_z
                        
                    elif movement_name == "forward_back":  # Forward-backward translation
                        offset_z = movement_amplitude * movement_factor
                        moved_cam.T[2] += offset_z
                        
                    elif movement_name == "lateral":  # Lateral translation
                        # Oscillate left-right
                        offset_x = movement_amplitude * movement_factor * math.sin(t * 2 * math.pi)
                        moved_cam.T[0] += offset_x
                        
                    elif movement_name == "vertical":  # Vertical translation
                        # Move up and down
                        offset_y = movement_amplitude * movement_factor * math.sin(t * 2 * math.pi) 
                        moved_cam.T[1] += offset_y
                        
                    elif movement_name == "wobble":  # Wobble movement
                        # Apply wobble directly to camera translation in image plane
                        # Wobble creates circular motion in the camera's local x-y plane
                        t_tensor = torch.tensor(t, dtype=torch.float32)
                        
                        # Get camera's right and up vectors from rotation matrix
                        R = torch.tensor(moved_cam.R, dtype=torch.float32)
                        right_vec = R[:, 0]  # Camera's right direction (x-axis)
                        up_vec = R[:, 1]     # Camera's up direction (y-axis)
                        
                        # Create wobble offsets in camera's local coordinate system
                        wobble_x = torch.sin(2 * math.pi * t_tensor) * wobble_radius
                        wobble_y = -torch.cos(2 * math.pi * t_tensor) * wobble_radius
                        
                        # Apply wobble offset to camera position
                        wobble_offset = wobble_x * right_vec + wobble_y * up_vec
                        moved_cam.T += wobble_offset.numpy()
                        
                        print(f"Frame {frame_idx}: t={t:.3f}, wobble_x={wobble_x:.4f}, wobble_y={wobble_y:.4f}")
                    
                    # Recalculate camera matrices after position change
                    from utils.graphics_utils import getWorld2View2
                    moved_cam.world_view_transform = torch.tensor(getWorld2View2(moved_cam.R, moved_cam.T, moved_cam.trans, moved_cam.scale)).transpose(0, 1).to(moved_cam.data_device)
                    moved_cam.full_proj_transform = (moved_cam.world_view_transform.unsqueeze(0).bmm(moved_cam.projection_matrix.unsqueeze(0))).squeeze(0)
                    moved_cam.camera_center = moved_cam.world_view_transform.inverse()[3, :3]
                    
                    # Render with moved camera
                    render_pkg = render(moved_cam, gaussians, pipe, background)
                    image = render_pkg["render"]
                    
                    image = image.cpu().numpy()
                    image = image.transpose(1, 2, 0)
                    image = np.clip(image, 0, 1)
                    image = (image * 255).astype(np.uint8)
                    combined_movement_frames.append(image)
            
            # Add combined movement frames to video
            video_frames.extend(combined_movement_frames)
            
            # Add actual second half frames
            video_frames.extend(second_half_frames)
            
            # Save video
            if video_frames:
                height, width = video_frames[0].shape[:2]
                video_path = os.path.join(path, "test", f"novel_view_{variant['name']}.mp4")
                
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                
                for frame in video_frames:
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                
                out.release()
                print(f"Saved video: {video_path}")
                
                # # Also save individual frames for this movement variant
                # frames_dir = os.path.join(path, "test", f"frames_{variant['name']}")
                # os.makedirs(frames_dir, exist_ok=True)
                # for i, frame in enumerate(video_frames):
                #     frame_path = os.path.join(frames_dir, f"{i:05d}.png")
                #     Image.fromarray(frame).save(frame_path)
        

    
