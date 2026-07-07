import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from utils.general_utils import build_scaling_rotation_4d


def _frame_name(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return Path(str(value)).stem


def _track_resolution(tracks_npz):
    if "resolution" not in tracks_npz.files:
        return None
    resolution = tracks_npz["resolution"].astype(np.float32)
    return float(resolution[0]), float(resolution[1])


def _camera_table_from_cameras(cameras, frame_names, track_res, device):
    by_name = {_frame_name(camera.image_name): camera for camera in cameras}
    count = len(frame_names)
    world_view = torch.zeros((count, 4, 4), dtype=torch.float32, device=device)
    intr = torch.zeros((count, 4), dtype=torch.float32, device=device)
    timestamp = torch.zeros((count,), dtype=torch.float32, device=device)
    xy_scale = np.ones((count, 2), dtype=np.float32)
    valid = np.zeros((count,), dtype=bool)

    for frame_id, frame_name in enumerate(frame_names):
        camera = by_name.get(_frame_name(frame_name))
        if camera is None:
            continue
        world_view[frame_id] = camera.world_view_transform.to(device=device, dtype=torch.float32)
        intr[frame_id] = torch.tensor(
            [camera.fl_x, camera.fl_y, camera.cx, camera.cy],
            dtype=torch.float32,
            device=device,
        )
        timestamp[frame_id] = float(camera.timestamp)
        if track_res is not None:
            xy_scale[frame_id, 0] = float(camera.image_width) / track_res[0]
            xy_scale[frame_id, 1] = float(camera.image_height) / track_res[1]
        valid[frame_id] = True

    return world_view, intr, timestamp, xy_scale, valid


def _camera_table_from_transforms(source_path, frame_names, track_res, device):
    with open(Path(source_path) / "transforms_train.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    by_name = {_frame_name(frame["file_path"]): frame for frame in meta["frames"]}
    count = len(frame_names)
    world_view = torch.zeros((count, 4, 4), dtype=torch.float32, device=device)
    intr = torch.zeros((count, 4), dtype=torch.float32, device=device)
    timestamp = torch.zeros((count,), dtype=torch.float32, device=device)
    xy_scale = np.ones((count, 2), dtype=np.float32)
    valid = np.zeros((count,), dtype=bool)
    width = float(meta["w"])
    height = float(meta["h"])

    for frame_id, frame_name in enumerate(frame_names):
        frame = by_name.get(_frame_name(frame_name))
        if frame is None:
            continue
        c2w = np.asarray(frame["transform_matrix"], dtype=np.float32)
        w2c = np.linalg.inv(c2w).astype(np.float32)
        world_view[frame_id] = torch.from_numpy(w2c.T).to(device=device)
        intr[frame_id] = torch.tensor(
            [meta["fl_x"], meta["fl_y"], meta["cx"], meta["cy"]],
            dtype=torch.float32,
            device=device,
        )
        timestamp[frame_id] = float(frame.get("time", frame_id))
        if track_res is not None:
            xy_scale[frame_id, 0] = width / track_res[0]
            xy_scale[frame_id, 1] = height / track_res[1]
        valid[frame_id] = True

    return world_view, intr, timestamp, xy_scale, valid


def current_gaussian_xyz(gaussians, ids, timestamp):
    xyz = gaussians.get_xyz[ids]
    if gaussians.gaussian_dim != 4 or not gaussians.rot_4d:
        return xyz

    scaling_xyz = torch.exp(gaussians._scaling[ids, 0:1]).repeat(1, 3)
    scaling_t = torch.exp(gaussians._scaling_t[ids])
    scaling = torch.cat([scaling_xyz, scaling_t], dim=1)
    rotation_l = gaussians._rotation[ids]
    rotation_r = gaussians._rotation_r[ids]

    l_mat = build_scaling_rotation_4d(scaling, rotation_l, rotation_r)
    cov = l_mat @ l_mat.transpose(1, 2)
    cov_12 = cov[:, 0:3, 3]
    cov_t = cov[:, 3:4, 3].clamp_min(1e-8)
    dt = timestamp.reshape(-1, 1) - gaussians.get_t[ids]
    return xyz + cov_12 / cov_t * dt


class TrajLoss(nn.Module):
    def __init__(
        self,
        assoc_path,
        anchors_path,
        tracks_path,
        cameras=None,
        source_path=None,
        batch_size=1024,
        device="cuda",
    ):
        super().__init__()
        self.batch_size = int(batch_size)
        self.device = torch.device(device)

        anchors = np.load(anchors_path)
        tracks_npz = np.load(tracks_path)
        assoc = np.load(assoc_path)

        frame_names = [_frame_name(x) for x in anchors["frame_names"]]
        track_res = _track_resolution(tracks_npz)
        if cameras is not None:
            world_view, intr, timestamp, xy_scale, frame_valid = _camera_table_from_cameras(
                cameras,
                frame_names,
                track_res,
                self.device,
            )
        else:
            world_view, intr, timestamp, xy_scale, frame_valid = _camera_table_from_transforms(
                source_path,
                frame_names,
                track_res,
                self.device,
            )

        anchor_id = assoc["anchor_id"].astype(np.int64)
        traj_score = assoc["traj_score"].astype(np.float32)
        tracks = tracks_npz["tracks"].astype(np.float32)
        visibility = tracks_npz["visibility"].astype(bool)
        confidence = tracks_npz["confidence"].astype(np.float32)

        obs_assoc = []
        obs_frame = []
        obs_xy = []
        obs_weight = []
        for assoc_idx, anchor_idx in enumerate(anchor_id):
            track_id = int(anchors["anchor_track_id"][anchor_idx])
            start = int(anchors["segment_start"][anchor_idx])
            end = int(anchors["segment_end"][anchor_idx])
            frames = np.arange(start, end + 1, dtype=np.int64)
            keep = visibility[frames, track_id] & frame_valid[frames]
            if not keep.any():
                continue
            frames = frames[keep]
            xy = tracks[frames, track_id] * xy_scale[frames]
            weight = traj_score[assoc_idx] * confidence[frames, track_id]
            keep_weight = weight > 0.0
            if not keep_weight.any():
                continue
            frames = frames[keep_weight]
            xy = xy[keep_weight]
            weight = weight[keep_weight]
            obs_assoc.append(np.full(len(frames), assoc_idx, dtype=np.int64))
            obs_frame.append(frames)
            obs_xy.append(xy.astype(np.float32))
            obs_weight.append(weight.astype(np.float32))

        if not obs_assoc:
            raise RuntimeError("No valid trajectory observations were built")

        self.register_buffer("world_view", world_view)
        self.register_buffer("intr", intr)
        self.register_buffer("timestamp", timestamp)
        self.register_buffer("gs_id", torch.from_numpy(assoc["gs_id"].astype(np.int64)).to(self.device))
        self.register_buffer("idw_w", torch.from_numpy(assoc["idw_w"].astype(np.float32)).to(self.device))
        self.register_buffer("obs_assoc", torch.from_numpy(np.concatenate(obs_assoc)).to(self.device))
        self.register_buffer("obs_frame", torch.from_numpy(np.concatenate(obs_frame)).to(self.device))
        self.register_buffer("obs_xy", torch.from_numpy(np.concatenate(obs_xy)).to(self.device))
        self.register_buffer("obs_weight", torch.from_numpy(np.concatenate(obs_weight)).to(self.device))

    @property
    def num_obs(self):
        return int(self.obs_assoc.shape[0])

    @property
    def num_anchors(self):
        return int(self.gs_id.shape[0])

    def _project_obs(self, gaussians, obs_ids):
        assoc_idx = self.obs_assoc[obs_ids]
        frame_id = self.obs_frame[obs_ids]
        gs_id = self.gs_id[assoc_idx]
        idw_w = self.idw_w[assoc_idx]
        batch_size, k = gs_id.shape

        obs_t = self.timestamp[frame_id]
        flat_ids = gs_id.reshape(-1)
        flat_t = obs_t[:, None].expand(batch_size, k).reshape(-1)
        gs_xyz = current_gaussian_xyz(gaussians, flat_ids, flat_t).reshape(batch_size, k, 3)
        pred_xyz = (gs_xyz * idw_w[..., None]).sum(dim=1)

        ones = torch.ones((batch_size, 1), dtype=pred_xyz.dtype, device=pred_xyz.device)
        pred_h = torch.cat([pred_xyz, ones], dim=1)
        cam = torch.bmm(pred_h[:, None, :], self.world_view[frame_id]).squeeze(1)
        z = cam[:, 2].clamp_min(1e-6)
        intr = self.intr[frame_id]
        uv = torch.stack(
            [
                intr[:, 0] * cam[:, 0] / z + intr[:, 2],
                intr[:, 1] * cam[:, 1] / z + intr[:, 3],
            ],
            dim=1,
        )
        valid = cam[:, 2] > 1e-6
        return uv, valid

    def obs_error(self, gaussians, obs_ids):
        uv, valid = self._project_obs(gaussians, obs_ids)
        target = self.obs_xy[obs_ids]
        weight = self.obs_weight[obs_ids] * valid.float()
        err = torch.sqrt(((uv - target) ** 2).sum(dim=1) + 1e-3)
        return err, weight, valid

    def forward(self, gaussians):
        obs_ids = torch.randint(0, self.num_obs, (self.batch_size,), device=self.device)
        err, weight, _ = self.obs_error(gaussians, obs_ids)
        return (err * weight).sum() / weight.sum().clamp_min(1e-6)

    @torch.no_grad()
    def evaluate(self, gaussians, batch_size=8192):
        errors = []
        weights = []
        valid_count = 0
        for start in range(0, self.num_obs, batch_size):
            obs_ids = torch.arange(
                start,
                min(start + batch_size, self.num_obs),
                dtype=torch.long,
                device=self.device,
            )
            err, weight, valid = self.obs_error(gaussians, obs_ids)
            errors.append(err.detach().cpu().numpy())
            weights.append(weight.detach().cpu().numpy())
            valid_count += int(valid.sum().item())

        err_np = np.concatenate(errors)
        weight_np = np.concatenate(weights)
        keep = np.isfinite(err_np) & (weight_np > 0.0)
        if not keep.any():
            raise RuntimeError("No valid weighted trajectory reprojection errors")

        err_np = err_np[keep]
        weight_np = weight_np[keep]
        return {
            "px_mean": float(err_np.mean()),
            "px_p50": float(np.percentile(err_np, 50)),
            "px_p90": float(np.percentile(err_np, 90)),
            "px_wmean": float((err_np * weight_np).sum() / weight_np.sum()),
            "num_anchors": self.num_anchors,
            "num_obs": self.num_obs,
            "num_valid_obs": int(valid_count),
            "num_invalid_obs": int(self.num_obs - valid_count),
        }
