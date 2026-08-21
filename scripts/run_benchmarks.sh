#!/usr/bin/env bash
# Copyright 2026 YOLOs-TRT Team
# SPDX-License-Identifier: AGPL-3.0
#
# Benchmark every task through its ROS 2 node and write a markdown table.
#
# Latency is reported by the nodes themselves on ~/timing, so the numbers
# measure inference plus conversion — not DDS transport or subscriber
# scheduling. Publish rate is measured at a subscriber and is therefore
# bounded by the input frame rate; raise --publish-rate to find the ceiling.
#
# Usage:
#   ./scripts/run_benchmarks.sh <models_dir> <video> [samples]
set -euo pipefail

MODELS_DIR="${1:?usage: run_benchmarks.sh <models_dir> <video> [samples]}"
VIDEO="${2:?usage: run_benchmarks.sh <models_dir> <video> [samples]}"
SAMPLES="${3:-300}"
PUBLISH_RATE="${PUBLISH_RATE:-60.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
OUT_DIR="${PKG_DIR}/benchmarks"
mkdir -p "$OUT_DIR"
RESULTS="${OUT_DIR}/results.json"
TABLE="${OUT_DIR}/results.md"

# task | model glob | launch file | node | output topic | labels
TASKS=(
  "detection|yolo*n.onnx|detector.launch.py|yolos_trt_detector|/yolos_trt_detector/detections|coco.names"
  "segmentation|yolo*-seg.onnx|segmentor.launch.py|yolos_trt_segmentor|/yolos_trt_segmentor/detections|coco.names"
  "pose|yolo*-pose.onnx|pose.launch.py|yolos_trt_pose|/yolos_trt_pose/detections|coco.names"
  "depth|yolo*-depth.onnx|depth.launch.py|yolos_trt_depth|/yolos_trt_depth/depth/image|"
)

echo "[" > "$RESULTS"
FIRST=1

for spec in "${TASKS[@]}"; do
  IFS='|' read -r task glob launch node topic labels <<< "$spec"
  model="$(ls ${MODELS_DIR}/${glob} 2>/dev/null | head -1 || true)"
  if [ -z "$model" ]; then
    echo "== skipping $task: no model matching ${glob} in ${MODELS_DIR}"
    continue
  fi

  echo "== benchmarking $task with $(basename "$model")"
  ARGS=(model_path:="$model" image_topic:=/camera/image_raw publish_timing:=true)
  [ -n "$labels" ] && [ -f "${MODELS_DIR}/${labels}" ] && ARGS+=(labels_path:="${MODELS_DIR}/${labels}")

  ros2 run image_publisher image_publisher_node "$VIDEO" \
    --ros-args -r __ns:=/camera -p publish_rate:="$PUBLISH_RATE" &
  PUB_PID=$!
  sleep 3

  ros2 launch ros2_yolos_trt "$launch" "${ARGS[@]}" &
  LAUNCH_PID=$!
  sleep 6

  # configure can take minutes on the first run: the engine is built here
  ros2 lifecycle set "/$node" configure
  ros2 lifecycle set "/$node" activate
  sleep 3

  "${SCRIPT_DIR}/benchmark_task.py" --node "/$node" --output-topic "$topic" \
    --samples "$SAMPLES" --json "${OUT_DIR}/${task}.json" || true

  # Killing the launch shell is not enough: ros2 launch children survive it and
  # keep publishing, which contaminates the next task's measurements. Match on
  # the command line, since pkill -x cannot match names longer than 15
  # characters (image_publisher_node is 20). The bracketed first character stops
  # the pattern matching this script itself.
  kill $LAUNCH_PID $PUB_PID 2>/dev/null || true
  pkill -f "[i]mage_publisher_node" 2>/dev/null || true
  pkill -f "[y]olos_trt_.*_node" 2>/dev/null || true
  wait $LAUNCH_PID 2>/dev/null || true
  wait $PUB_PID 2>/dev/null || true
  sleep 4

  if [ -f "${OUT_DIR}/${task}.json" ]; then
    [ $FIRST -eq 0 ] && echo "," >> "$RESULTS"
    python3 -c "
import json,sys
d=json.load(open('${OUT_DIR}/${task}.json'))
d['task']='${task}'
d['model']='$(basename "$model")'
sys.stdout.write(json.dumps(d, indent=2))
" >> "$RESULTS"
    FIRST=0
  fi
done

echo "" >> "$RESULTS"
echo "]" >> "$RESULTS"

python3 - "$RESULTS" "$TABLE" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
lines = [
    "| Task | Model | Inference mean | p50 | p95 | p99 | Inference FPS | Publish rate |",
    "|------|-------|---------------:|----:|----:|----:|--------------:|-------------:|",
]
for r in rows:
    lines.append(
        f"| {r['task'].capitalize()} | `{r['model']}` | "
        f"{r['inference_ms_mean']} ms | {r['inference_ms_p50']} ms | "
        f"{r['inference_ms_p95']} ms | {r['inference_ms_p99']} ms | "
        f"**{r['inference_fps']}** | {r['publish_rate_hz']} Hz |"
    )
open(sys.argv[2], "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
PY

echo "wrote $TABLE"
