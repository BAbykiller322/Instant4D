import argparse
import json
import os

import numpy as np


def read_dataset(path, image_scale="2x"):
    data_path = os.path.join(path, "dataset.json")
    with open(data_path, "r") as f:
        dataset = json.load(f)

    count = dataset["count"]
    train_ids = dataset["train_ids"]
    val_ids = dataset["val_ids"]

    train_dicts = []
    val_dicts = []

    for id in train_ids:
        train_dict = {}
        train_dict["img"] = os.path.join(path, "rgb", image_scale, f"{id}")
        train_dict["cam"] = os.path.join(path, "camera", f"{id}.json")
        # id is pos_time, e.g. 0_00000
        train_dict["pos_id"] = int(id.split("_")[0])
        train_dict["time_id"] = int(id.split("_")[1])
        train_dicts.append(train_dict)

    for id in val_ids:
        val_dict = {}
        val_dict["img"] = os.path.join(path, "rgb", image_scale, f"{id}")
        val_dict["cam"] = os.path.join(path, "camera", f"{id}.json")
        val_dict["pos_id"] = int(id.split("_")[0])
        val_dict["time_id"] = int(id.split("_")[1])
        val_dicts.append(val_dict)

    assert len(train_dicts) + len(val_dicts) == count

    return train_dicts, val_dicts


def read_camera(path):
    with open(path, "r") as f:
        camera = json.load(f)

    focal_length = camera["focal_length"]
    image_size = camera["image_size"]
    orientation = camera["orientation"]
    position = camera["position"]
    principal_point = camera["principal_point"]

    return focal_length, image_size, orientation, position, principal_point


def make_frame_dict(frame, output_dir, max_time_id, time_scale):
    frame_dict = {}

    image_path = frame["img"]
    try:
        image_path_no_ext = os.path.splitext(
            os.path.relpath(image_path + ".png", output_dir)
        )[0]
    except ValueError:
        image_path_no_ext = image_path
    frame_dict["file_path"] = image_path_no_ext.replace("\\", "/")

    _, _, orientation, position, _ = read_camera(frame["cam"])
    c2w = np.eye(4)

    R = np.asarray(orientation)  # list [3, 3]
    T = np.asarray(position)  # list [3]

    c2w[:3, :3] = R
    c2w[:3, 3] = T

    frame_dict["transform_matrix"] = c2w.tolist()
    frame_dict["time"] = frame["time_id"] / max_time_id * time_scale
    frame_dict["pos_id"] = frame["pos_id"]

    return frame_dict


def read_iphone_scene(
    dir,
    scene_name,
    output_dir=None,
    image_scale="2x",
    time_scale=3.0,
):
    scene_dir = os.path.join(dir, scene_name)
    train_dicts, val_dicts = read_dataset(scene_dir, image_scale=image_scale)

    if output_dir is None:
        output_dir = os.path.join(scene_dir, "preprocess_output", "instant4d_source")
    os.makedirs(output_dir, exist_ok=True)

    dict_to_save = {}

    cam0 = train_dicts[0]["cam"]
    focal_length, image_size, _, _, principal_point = read_camera(cam0)

    image_downscale = float(image_scale.rstrip("x"))
    dict_to_save["w"] = image_size[0] / image_downscale
    dict_to_save["h"] = image_size[1] / image_downscale
    dict_to_save["fl_x"] = focal_length / image_downscale
    dict_to_save["fl_y"] = focal_length / image_downscale
    dict_to_save["cx"] = principal_point[0] / image_downscale
    dict_to_save["cy"] = principal_point[1] / image_downscale

    all_time_ids = [frame["time_id"] for frame in train_dicts + val_dicts]
    max_time_id = max(all_time_ids)

    frame_train = []
    frame_val = []

    for train_dict in train_dicts:
        frame_train.append(
            make_frame_dict(train_dict, output_dir, max_time_id, time_scale)
        )

    for val_dict in val_dicts:
        frame_val.append(make_frame_dict(val_dict, output_dir, max_time_id, time_scale))

    dict_to_save["frames"] = frame_train
    with open(os.path.join(output_dir, "transforms_train.json"), "w") as f:
        json.dump(dict_to_save, f, indent=4)

    dict_to_save["frames"] = frame_val
    with open(os.path.join(output_dir, "transforms_test.json"), "w") as f:
        json.dump(dict_to_save, f, indent=4)

    print(f"Saved transforms to {output_dir}")
    print(f"train frames: {len(frame_train)}, val frames: {len(frame_val)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--image_scale", default="2x")
    parser.add_argument("--time_scale", type=float, default=3.0)
    args = parser.parse_args()

    read_iphone_scene(
        args.data_dir,
        args.scene_name,
        output_dir=args.output_dir,
        image_scale=args.image_scale,
        time_scale=args.time_scale,
    )
