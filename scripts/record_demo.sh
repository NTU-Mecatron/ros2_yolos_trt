#!/usr/bin/env bash
# Copyright 2026 YOLOs-TRT Team
# SPDX-License-Identifier: AGPL-3.0
#
# Record an RViz demo GIF for one task, from real video input.
#
# Requires: a CUDA GPU, a sourced ROS 2 install with this package built,
# rviz2, Xvfb, ffmpeg, xdotool.
#
# Usage:
#   ./scripts/record_demo.sh <task> <model> <video> [seconds]
#     task: detection | segmentation | pose | depth
#
# Example:
#   ./scripts/record_demo.sh depth yolo26n-depth.onnx street.mp4 12
set -euo pipefail

TASK="${1:?usage: record_demo.sh <task> <model> <video> [seconds]}"
MODEL="${2:?usage: record_demo.sh <task> <model> <video> [seconds]}"
VIDEO="${3:?usage: record_demo.sh <task> <model> <video> [seconds]}"
RECORD_SECONDS="${4:-12}"
LABELS="${LABELS:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
OUT_DIR="${PKG_DIR}/assets"
WORK_DIR="$(mktemp -d)"
cleanup() { kill $(jobs -p) 2>/dev/null || true; rm -rf "$WORK_DIR"; }
trap cleanup EXIT

case "$TASK" in
  detection)    NODE=yolos_trt_detector;   LAUNCH=detector.launch.py;  TITLE="Object Detection";      DEBUG_TOPIC=/yolos_trt_detector/debug_image ;;
  segmentation) NODE=yolos_trt_segmentor;  LAUNCH=segmentor.launch.py; TITLE="Instance Segmentation"; DEBUG_TOPIC=/yolos_trt_segmentor/debug_image ;;
  pose)         NODE=yolos_trt_pose;       LAUNCH=pose.launch.py;      TITLE="Pose Estimation";       DEBUG_TOPIC=/yolos_trt_pose/debug_image ;;
  depth)        NODE=yolos_trt_depth;      LAUNCH=depth.launch.py;     TITLE="Metric Depth (metres)"; DEBUG_TOPIC=/yolos_trt_depth/depth/debug_image ;;
  *) echo "unknown task: $TASK" >&2; exit 1 ;;
esac

# RViz docks Image displays into a fixed-width left column, so a full-window
# grab is mostly empty 3D view. Frame the capture on that column instead.
CROP="${CROP:-350:396:0:0}"
SCALE_WIDTH="${SCALE_WIDTH:-560}"
GIF_FPS="${GIF_FPS:-8}"
GIF_COLORS="${GIF_COLORS:-64}"

mkdir -p "$OUT_DIR"
sed -e "s|@TITLE@|${TITLE}|g" -e "s|@DEBUG_TOPIC@|${DEBUG_TOPIC}|g" \
    "${PKG_DIR}/rviz/demo.rviz.in" > "${WORK_DIR}/demo.rviz"

echo "== virtual display =="
# On a GPU container, glvnd offers NVIDIA's EGL vendor to Xvfb, which then
# segfaults during GLX init. Restricting the vendor list to Mesa keeps the
# virtual display on software rendering, which is what a headless capture wants.
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export __GLX_VENDOR_LIBRARY_NAME=mesa
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export DISPLAY=:99

rm -f /tmp/.X99-lock
Xvfb :99 -screen 0 1280x720x24 &
sleep 4
xdpyinfo >/dev/null 2>&1 || { echo "Xvfb failed to start on :99" >&2; exit 1; }

echo "== publishing $VIDEO =="
ros2 run image_publisher image_publisher_node "$VIDEO" \
  --ros-args -r __ns:=/camera -p publish_rate:=15.0 &
sleep 4

echo "== $TASK node =="
LAUNCH_ARGS=(model_path:="$MODEL" image_topic:=/camera/image_raw publish_debug_image:=true)
[ -n "$LABELS" ] && LAUNCH_ARGS+=(labels_path:="$LABELS")
ros2 launch ros2_yolos_trt "$LAUNCH" "${LAUNCH_ARGS[@]}" &
sleep 6

echo "== lifecycle (first configure builds the engine) =="
ros2 lifecycle set "/$NODE" configure
ros2 lifecycle set "/$NODE" activate

echo "== waiting for annotated output =="
timeout 300 ros2 topic echo --once "$DEBUG_TOPIC" > /dev/null \
  || { echo "no output on $DEBUG_TOPIC" >&2; exit 1; }

echo "== rviz2 =="
rviz2 -d "${WORK_DIR}/demo.rviz" &
sleep 12
if command -v xdotool >/dev/null 2>&1; then
  WID="$(xdotool search --name RViz | head -1 || true)"
  [ -n "${WID:-}" ] && xdotool windowsize "$WID" 1280 720 && sleep 2
fi

echo "== recording ${RECORD_SECONDS}s =="
ffmpeg -y -loglevel error -video_size 1280x720 -framerate 15 -f x11grab -i :99 \
  -t "$RECORD_SECONDS" -vf "crop=${CROP}" "${WORK_DIR}/capture.mp4"

echo "== gif =="
ffmpeg -y -loglevel error -i "${WORK_DIR}/capture.mp4" \
  -vf "fps=${GIF_FPS},scale=${SCALE_WIDTH}:-1:flags=lanczos,palettegen=max_colors=${GIF_COLORS}" \
  "${WORK_DIR}/palette.png"
ffmpeg -y -loglevel error -i "${WORK_DIR}/capture.mp4" -i "${WORK_DIR}/palette.png" \
  -lavfi "fps=${GIF_FPS},scale=${SCALE_WIDTH}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4" \
  "${OUT_DIR}/${TASK}.gif"

echo "wrote ${OUT_DIR}/${TASK}.gif"
ls -lh "${OUT_DIR}/${TASK}.gif"
