#!/usr/bin/env bash
# Copyright 2026 YOLOs-TRT Team
# SPDX-License-Identifier: AGPL-3.0
#
# Regenerate assets/depth_rviz.gif — an RViz recording of the depth node.
#
# Requires: a CUDA GPU, a sourced ROS 2 install with this package built,
# rviz2, Xvfb, ffmpeg, and a video or image to feed in.
#
# Usage:
#   ./scripts/record_depth_demo.sh <model.onnx|model.trt> <input> [seconds]
set -euo pipefail

MODEL="${1:?usage: record_depth_demo.sh <model> <input> [seconds]}"
INPUT="${2:?usage: record_depth_demo.sh <model> <input> [seconds]}"
RECORD_SECONDS="${3:-12}"

# RViz docks its Image displays into a fixed-width left column, so a full-window
# capture is mostly empty 3D view. Frame the capture on that column instead.
# Override if your RViz lays out differently: CROP=w:h:x:y
CROP="${CROP:-350:396:0:0}"
SCALE_WIDTH="${SCALE_WIDTH:-560}"
GIF_FPS="${GIF_FPS:-8}"
GIF_COLORS="${GIF_COLORS:-64}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
OUT_DIR="${PKG_DIR}/assets"
WORK_DIR="$(mktemp -d)"
cleanup() { kill $(jobs -p) 2>/dev/null || true; rm -rf "$WORK_DIR"; }
trap cleanup EXIT

mkdir -p "$OUT_DIR"

echo "== virtual display =="
# On a GPU container, glvnd offers NVIDIA's EGL vendor to Xvfb, which then
# segfaults during GLX init (libEGL_nvidia -> libnvidia-egl-gbm). Restricting
# the vendor list to Mesa keeps the virtual display on software rendering,
# which is what a headless capture wants anyway.
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export __GLX_VENDOR_LIBRARY_NAME=mesa
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export DISPLAY=:99

rm -f /tmp/.X99-lock
Xvfb :99 -screen 0 1280x720x24 &
sleep 4
if ! xdpyinfo >/dev/null 2>&1; then
  echo "Xvfb failed to start on :99" >&2
  exit 1
fi

echo "== publishing frames from $INPUT =="
ros2 run image_publisher image_publisher_node "$INPUT" \
  --ros-args -r __ns:=/camera -p publish_rate:=15.0 &
sleep 4

echo "== depth node =="
ros2 launch ros2_yolos_trt depth.launch.py \
  model_path:="$MODEL" \
  image_topic:=/camera/image_raw \
  publish_debug_image:=true &
sleep 6

echo "== lifecycle: configure (builds the engine on first run) =="
ros2 lifecycle set /yolos_trt_depth configure
ros2 lifecycle set /yolos_trt_depth activate

echo "== waiting for depth output =="
timeout 300 ros2 topic echo --once /yolos_trt_depth/depth/image > /dev/null \
  || { echo "no depth published"; exit 1; }

echo "== rviz2 =="
rviz2 -d "${PKG_DIR}/rviz/depth_demo.rviz" &
sleep 12
# Deterministic window size so the crop above lands in the same place.
if command -v xdotool >/dev/null 2>&1; then
  WID="$(xdotool search --name RViz | head -1 || true)"
  [ -n "${WID:-}" ] && xdotool windowsize "$WID" 1280 720 && sleep 2
fi

echo "== recording ${RECORD_SECONDS}s =="
ffmpeg -y -loglevel error -video_size 1280x720 -framerate 15 -f x11grab -i :99 \
  -t "$RECORD_SECONDS" -vf "crop=${CROP}" "${WORK_DIR}/capture.mp4"

echo "== gif (two-pass palette keeps it small and clean) =="
ffmpeg -y -loglevel error -i "${WORK_DIR}/capture.mp4" \
  -vf "fps=${GIF_FPS},scale=${SCALE_WIDTH}:-1:flags=lanczos,palettegen=max_colors=${GIF_COLORS}" \
  "${WORK_DIR}/palette.png"
ffmpeg -y -loglevel error -i "${WORK_DIR}/capture.mp4" -i "${WORK_DIR}/palette.png" \
  -lavfi "fps=${GIF_FPS},scale=${SCALE_WIDTH}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4" \
  "${OUT_DIR}/depth_rviz.gif"

echo "wrote ${OUT_DIR}/depth_rviz.gif"
ls -lh "${OUT_DIR}/depth_rviz.gif"
