#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash script/reconstruct_dycheck.sh --scene SCENE --data-root DATA_ROOT [options]

Options:
  --scene NAME             DyCheck scene name, e.g. apple. Required.
  --data-root PATH         DyCheck root containing scene folders. Required.
  --image-scale SCALE      Image scale folder under rgb/; default: 2x.
  --cache-root PATH        Intermediate cache root; default: $SCRATCH/instant4d_preprocess_cache.
  --gpu ID                 CUDA_VISIBLE_DEVICES value; default: 0.
  --stride N               Frame stride for mono depth, DroidSLAM, and flow; default: 1.
  --temporal-init-mode M   Source temporal init: motion_split, all_static, or temporal_motion_mask; default: motion_split.
  --temporal-motion-mask-path PATH
                           Optional temporal motion mask output/input path.
  --temporal-motion-threshold X
                           Threshold for temporal_motion_mask source export; default: 0.5.
  --temporal-dynamic-scale-floor X
                           Minimum scale_time for temporal-dynamic points; default: 0.0.
  --keep-cache             Keep intermediate cache after final Instant4D source is written.
  --clean-cache            Remove this scene's cache before starting.
EOF
}

SCENE=""
DATA_ROOT=""
IMAGE_SCALE="2x"
GPU="0"
STRIDE="1"
KEEP_CACHE="0"
CLEAN_CACHE="0"
TEMPORAL_INIT_MODE="motion_split"
TEMPORAL_MOTION_MASK_PATH=""
TEMPORAL_MOTION_THRESHOLD="0.5"
TEMPORAL_DYNAMIC_SCALE_FLOOR="0.0"
CACHE_ROOT="${SCRATCH:-/tmp}/instant4d_preprocess_cache"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene)
      SCENE="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --image-scale)
      IMAGE_SCALE="$2"
      shift 2
      ;;
    --cache-root)
      CACHE_ROOT="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --stride)
      STRIDE="$2"
      shift 2
      ;;
    --temporal-init-mode)
      TEMPORAL_INIT_MODE="$2"
      shift 2
      ;;
    --temporal-motion-mask-path)
      TEMPORAL_MOTION_MASK_PATH="$2"
      shift 2
      ;;
    --temporal-motion-threshold)
      TEMPORAL_MOTION_THRESHOLD="$2"
      shift 2
      ;;
    --temporal-dynamic-scale-floor)
      TEMPORAL_DYNAMIC_SCALE_FLOOR="$2"
      shift 2
      ;;
    --keep-cache)
      KEEP_CACHE="1"
      shift
      ;;
    --clean-cache)
      CLEAN_CACHE="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$SCENE" || -z "$DATA_ROOT" ]]; then
  usage >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEGASAM_ROOT="$REPO_ROOT/SLAM/mega-sam"
SCENE_DIR="$DATA_ROOT/$SCENE"
IMAGE_DIR="$SCENE_DIR/rgb/$IMAGE_SCALE"
CAMERA_DIR="$SCENE_DIR/camera"
DYCHECK_TRAIN_SPLIT="$SCENE_DIR/splits/train.json"
PREPROCESS_ROOT="$SCENE_DIR/preprocess_output"
CACHE_SCENE_ROOT="$CACHE_ROOT/$SCENE"
MEGASAM_OUT="$PREPROCESS_ROOT/mega_sam"
INSTANT4D_SOURCE="$PREPROCESS_ROOT/instant4d_source"

if [[ ! -d "$SCENE_DIR" ]]; then
  echo "Scene directory not found: $SCENE_DIR" >&2
  exit 1
fi
if [[ ! -d "$IMAGE_DIR" ]]; then
  echo "Image directory not found: $IMAGE_DIR" >&2
  exit 1
fi
if [[ ! -d "$CAMERA_DIR" ]]; then
  echo "Camera directory not found: $CAMERA_DIR" >&2
  exit 1
fi
if [[ ! -f "$DYCHECK_TRAIN_SPLIT" ]]; then
  echo "Train split not found: $DYCHECK_TRAIN_SPLIT" >&2
  exit 1
fi

if [[ "$CLEAN_CACHE" == "1" ]]; then
  rm -rf "$CACHE_SCENE_ROOT"
fi

mkdir -p "$CACHE_SCENE_ROOT" "$MEGASAM_OUT" "$INSTANT4D_SOURCE"

echo "Using official DyCheck train split: $DYCHECK_TRAIN_SPLIT"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$MEGASAM_ROOT:$MEGASAM_ROOT/UniDepth:${PYTHONPATH:-}"

DEPTH_ANYTHING_OUT="$CACHE_SCENE_ROOT/depth_anything/$SCENE"
UNIDEPTH_OUT="$CACHE_SCENE_ROOT/unidepth"
RECON_DIR="$CACHE_SCENE_ROOT/reconstructions"
FLOW_CACHE_DIR="$CACHE_SCENE_ROOT/cache_flow"
DROID_OUT_DIR="$MEGASAM_OUT/droid"
CVD_OUT_DIR="$MEGASAM_OUT/outputs_cvd"
if [[ -z "$TEMPORAL_MOTION_MASK_PATH" ]]; then
  TEMPORAL_MOTION_MASK_PATH="$MEGASAM_OUT/temporal_motion_mask.npy"
fi

cd "$MEGASAM_ROOT"

echo "=== [1/6] UniDepth: $SCENE ==="
python UniDepth/scripts/demo_mega-sam.py \
  --scene-name "$SCENE" \
  --img-path "$IMAGE_DIR" \
  --split-path "$DYCHECK_TRAIN_SPLIT" \
  --stride "$STRIDE" \
  --outdir "$UNIDEPTH_OUT"

echo "=== [2/6] DepthAnything: $SCENE ==="
python Depth-Anything/run_videos.py \
  --encoder vitl \
  --load-from Depth-Anything/checkpoints/depth_anything_vitl14.pth \
  --img-path "$IMAGE_DIR" \
  --split-path "$DYCHECK_TRAIN_SPLIT" \
  --stride "$STRIDE" \
  --outdir "$DEPTH_ANYTHING_OUT"

echo "=== [3/6] DroidSLAM tracking: $SCENE ==="
python camera_tracking_scripts/test_dycheck.py \
  --image_path "$IMAGE_DIR" \
  --camera_path "$CAMERA_DIR" \
  --split_path "$DYCHECK_TRAIN_SPLIT" \
  --image_downscale "${IMAGE_SCALE%x}" \
  --weights checkpoints/megasam_final.pth \
  --scene_name "$SCENE" \
  --mono_depth_path "$DEPTH_ANYTHING_OUT" \
  --metric_depth_path "$UNIDEPTH_OUT/$SCENE" \
  --reconstruction_dir "$RECON_DIR" \
  --droid_output_dir "$DROID_OUT_DIR" \
  --stride "$STRIDE" \
  --disable_vis

echo "=== [4/6] RAFT flow cache: $SCENE ==="
python cvd_opt/preprocess_flow.py \
  --datapath "$IMAGE_DIR" \
  --split_path "$DYCHECK_TRAIN_SPLIT" \
  --stride "$STRIDE" \
  --model cvd_opt/raft-things.pth \
  --scene_name "$SCENE" \
  --cache_dir "$FLOW_CACHE_DIR" \
  --mixed_precision

echo "=== [5/6] CVD optimization: $SCENE ==="
python cvd_opt/cvd_opt.py \
  --scene_name "$SCENE" \
  --reconstruction_dir "$RECON_DIR" \
  --cache_dir "$FLOW_CACHE_DIR" \
  --output_dir "$CVD_OUT_DIR" \
  --w_grad 2.0 \
  --w_normal 5.0

if [[ "$TEMPORAL_INIT_MODE" == "temporal_motion_mask" ]]; then
  echo "=== [5.5/6] Temporal motion mask: $SCENE ==="
  cd "$REPO_ROOT"
  python script/make_temporal_motion_mask.py \
    --cvd_path "$CVD_OUT_DIR/${SCENE}_sgd_cvd_hr.npz" \
    --flow_dir "$FLOW_CACHE_DIR/$SCENE" \
    --out_path "$TEMPORAL_MOTION_MASK_PATH" \
    --debug_dir "$MEGASAM_OUT/temporal_motion_mask_debug"
fi

echo "=== [6/6] Instant4D source export: $SCENE ==="
cd "$REPO_ROOT"
rm -f "$INSTANT4D_SOURCE/rodygs_holdout_frames.json"
PRUNE_ARGS=(
  --droid_path "$CVD_OUT_DIR/${SCENE}_sgd_cvd_hr.npz" \
  --motion_path "$RECON_DIR/$SCENE/motion_prob.npy" \
  --save_dir "$INSTANT4D_SOURCE" \
  --scene_name "$SCENE" \
  --image_output_dir "$MEGASAM_OUT/cvd_images" \
  --prune_stride 3 \
  --temporal_init_mode "$TEMPORAL_INIT_MODE" \
  --temporal_motion_threshold "$TEMPORAL_MOTION_THRESHOLD" \
  --temporal_dynamic_scale_floor "$TEMPORAL_DYNAMIC_SCALE_FLOOR" \
  --train_split_path "$DYCHECK_TRAIN_SPLIT"
)
if [[ "$TEMPORAL_INIT_MODE" == "temporal_motion_mask" ]]; then
  PRUNE_ARGS+=(--temporal_motion_mask_path "$TEMPORAL_MOTION_MASK_PATH")
fi
python script/prune.py "${PRUNE_ARGS[@]}"

cat > "$MEGASAM_OUT/manifest.json" <<EOF
{
  "scene": "$SCENE",
  "scene_dir": "$SCENE_DIR",
  "image_scale": "$IMAGE_SCALE",
  "cache_root": "$CACHE_SCENE_ROOT",
  "protocol": "dycheck_official_train",
  "dycheck_train_split": "$DYCHECK_TRAIN_SPLIT",
  "source_train_split": "$DYCHECK_TRAIN_SPLIT",
  "instant4d_source": "$INSTANT4D_SOURCE",
  "cvd_npz": "$CVD_OUT_DIR/${SCENE}_sgd_cvd_hr.npz",
  "droid_npz": "$DROID_OUT_DIR/${SCENE}_droid.npz",
  "temporal_init_mode": "$TEMPORAL_INIT_MODE",
  "temporal_motion_mask_path": "$TEMPORAL_MOTION_MASK_PATH",
  "temporal_motion_threshold": $TEMPORAL_MOTION_THRESHOLD,
  "temporal_dynamic_scale_floor": $TEMPORAL_DYNAMIC_SCALE_FLOOR,
  "stride": $STRIDE
}
EOF

if [[ "$KEEP_CACHE" != "1" ]]; then
  echo "=== cleaning cache: $CACHE_SCENE_ROOT ==="
  rm -rf "$CACHE_SCENE_ROOT"
fi

echo "=== done: $INSTANT4D_SOURCE ==="
