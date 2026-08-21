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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
OUT_DIR="${PKG_DIR}/assets"
WORK_DIR="$(mktemp -d)"
cleanup() { kill $(jobs -p) 2>/dev/null || true; rm -rf "$WORK_DIR"; }
trap cleanup EXIT

mkdir -p "$OUT_DIR"

echo "== virtual display =="
export DISPLAY=:99
Xvfb :99 -screen 0 1280x720x24 &
sleep 3

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
sleep 10

echo "== recording ${RECORD_SECONDS}s =="
ffmpeg -y -loglevel error -video_size 1280x720 -framerate 15 -f x11grab -i :99 \
  -t "$RECORD_SECONDS" "${WORK_DIR}/capture.mp4"

echo "== gif (two-pass palette keeps it small and clean) =="
ffmpeg -y -loglevel error -i "${WORK_DIR}/capture.mp4" \
  -vf "fps=10,scale=960:-1:flags=lanczos,palettegen" "${WORK_DIR}/palette.png"
ffmpeg -y -loglevel error -i "${WORK_DIR}/capture.mp4" -i "${WORK_DIR}/palette.png" \
  -lavfi "fps=10,scale=960:-1:flags=lanczos[x];[x][1:v]paletteuse" \
  "${OUT_DIR}/depth_rviz.gif"

echo "wrote ${OUT_DIR}/depth_rviz.gif"
ls -lh "${OUT_DIR}/depth_rviz.gif"
