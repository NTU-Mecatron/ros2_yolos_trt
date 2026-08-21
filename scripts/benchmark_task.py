#!/usr/bin/env python3
# Copyright 2026 YOLOs-TRT Team
# SPDX-License-Identifier: AGPL-3.0
"""Measure a running ros2_yolos_trt node's throughput and latency.

Subscribes to the node's ~/timing topic (published when publish_timing is
true) and to its primary output topic, then reports inference latency
percentiles and the achieved publish rate.

Latency comes from the node itself rather than from wall-clock differences at
the subscriber, so the numbers are not polluted by DDS transport or by this
script's own scheduling.

Usage:
  ./benchmark_task.py --node /yolos_trt_depth \\
                      --output-topic /yolos_trt_depth/depth/image \\
                      --samples 300 --json out.json
"""

import argparse
import json
import statistics
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from std_msgs.msg import Float64MultiArray


class Benchmark(Node):
    def __init__(self, node_ns: str, output_topic: str, samples: int):
        super().__init__("yolos_trt_benchmark")
        self.target_samples = samples
        self.inference_ms: list[float] = []
        self.total_ms: list[float] = []
        self.output_count = 0
        self.first_output_time: float | None = None
        self.last_output_time: float | None = None

        self.create_subscription(
            Float64MultiArray, f"{node_ns}/timing", self._on_timing, 50
        )

        # Subscribe generically so one script covers every task's output type.
        self._output_topic = output_topic
        self.create_timer(0.5, self._try_bind_output)
        self._bound = False

    def _try_bind_output(self):
        if self._bound:
            return
        for name, types in self.get_topic_names_and_types():
            if name == self._output_topic and types:
                msg_type = self._load(types[0])
                if msg_type is None:
                    return
                self.create_subscription(
                    msg_type, name, self._on_output,
                    QoSPresetProfiles.SENSOR_DATA.value,
                )
                self._bound = True
                self.get_logger().info(f"bound to {name} [{types[0]}]")
                return

    @staticmethod
    def _load(type_string: str):
        try:
            pkg, _, msg = type_string.split("/")
            module = __import__(f"{pkg}.msg", fromlist=[msg])
            return getattr(module, msg)
        except Exception:
            return None

    def _on_timing(self, msg: Float64MultiArray):
        if len(msg.data) >= 2:
            self.inference_ms.append(float(msg.data[0]))
            self.total_ms.append(float(msg.data[1]))

    def _on_output(self, _msg):
        now = time.monotonic()
        if self.first_output_time is None:
            self.first_output_time = now
        self.last_output_time = now
        self.output_count += 1

    def done(self) -> bool:
        return len(self.inference_ms) >= self.target_samples

    def report(self) -> dict:
        def pct(values, q):
            if not values:
                return None
            ordered = sorted(values)
            idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
            return round(ordered[idx], 3)

        # Discard the first few samples: the CUDA graph and allocator settle in.
        warm = self.inference_ms[5:] if len(self.inference_ms) > 15 else self.inference_ms

        rate = None
        if (self.first_output_time is not None
                and self.last_output_time is not None
                and self.output_count > 1):
            span = self.last_output_time - self.first_output_time
            if span > 0:
                rate = round((self.output_count - 1) / span, 1)

        return {
            "samples": len(warm),
            "inference_ms_mean": round(statistics.fmean(warm), 3) if warm else None,
            "inference_ms_p50": pct(warm, 0.50),
            "inference_ms_p95": pct(warm, 0.95),
            "inference_ms_p99": pct(warm, 0.99),
            "inference_fps": round(1000.0 / statistics.fmean(warm), 1) if warm else None,
            "callback_ms_mean": round(statistics.fmean(self.total_ms[5:] or self.total_ms), 3)
            if self.total_ms else None,
            "published_msgs": self.output_count,
            "publish_rate_hz": rate,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", required=True, help="node namespace, e.g. /yolos_trt_depth")
    parser.add_argument("--output-topic", required=True)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    rclpy.init()
    bench = Benchmark(args.node, args.output_topic, args.samples)

    deadline = time.monotonic() + args.timeout
    while rclpy.ok() and not bench.done() and time.monotonic() < deadline:
        rclpy.spin_once(bench, timeout_sec=0.1)

    report = bench.report()
    if not report["samples"]:
        print("ERROR: no timing samples received. Is publish_timing set to true "
              "and the node activated?", file=sys.stderr)
        rclpy.shutdown()
        return 1

    print(json.dumps(report, indent=2))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2)

    bench.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
