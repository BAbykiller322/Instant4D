import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MEGASAM_ROOT = REPO_ROOT / "SLAM" / "mega-sam"
sys.path.insert(0, str(MEGASAM_ROOT))

from dycheck_utils import collect_image_files, load_split_frame_names


def test_collect_image_files_uses_time_id_order(tmp_path):
  image_dir = tmp_path / "rgb" / "2x"
  image_dir.mkdir(parents=True)
  (image_dir / "0_00002.png").write_bytes(b"")
  (image_dir / "0_00000.jpg").write_bytes(b"")
  (image_dir / "0_00001.png").write_bytes(b"")

  split_path = tmp_path / "train.json"
  split_path.write_text(
      json.dumps({
          "frame_names": ["0_00002", "0_00000", "0_00001"],
          "time_ids": [2, 0, 1],
      }),
      encoding="utf-8")

  assert load_split_frame_names(split_path) == [
      "0_00000", "0_00001", "0_00002"]
  assert [Path(p).stem for p in collect_image_files(image_dir, split_path)] == [
      "0_00000", "0_00001", "0_00002"]


def test_collect_image_files_reports_missing_split_frame(tmp_path):
  image_dir = tmp_path / "rgb" / "2x"
  image_dir.mkdir(parents=True)
  (image_dir / "0_00000.png").write_bytes(b"")
  split_path = tmp_path / "train.json"
  split_path.write_text(
      json.dumps({"frame_names": ["0_00000", "0_00001"]}),
      encoding="utf-8")

  try:
    collect_image_files(image_dir, split_path)
  except FileNotFoundError as exc:
    assert "0_00001" in str(exc)
  else:
    raise AssertionError("missing split frame should raise FileNotFoundError")
