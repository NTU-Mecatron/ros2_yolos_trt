#!/usr/bin/env python3
# Copyright 2026 YOLOs-TRT Team
# SPDX-License-Identifier: AGPL-3.0
"""Splice benchmarks/results.json into the README's benchmark section.

Keeps the published numbers honest: the table in the README is generated from
a real run rather than typed by hand.

Usage:
  ./scripts/update_readme_benchmarks.py [--gpu "NVIDIA RTX 4090"] [--notes "..."]
"""

import argparse
import json
import pathlib
import sys

MARKER = "<!-- BENCHMARK_TABLE -->"
END_MARKER = "<!-- /BENCHMARK_TABLE -->"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="")
    parser.add_argument("--trt", default="")
    parser.add_argument("--ros", default="")
    parser.add_argument("--results", default="benchmarks/results.json")
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()

    results_path = pathlib.Path(args.results)
    if not results_path.exists():
        print(f"no results at {results_path}", file=sys.stderr)
        return 1

    rows = json.loads(results_path.read_text())
    if not rows:
        print("results file is empty", file=sys.stderr)
        return 1

    header = []
    if args.gpu:
        bits = [f"**{args.gpu}**"]
        if args.trt:
            bits.append(f"TensorRT {args.trt}")
        if args.ros:
            bits.append(args.ros)
        header.append("Measured on " + " · ".join(bits) + ", fp16 engines.")
        header.append("")

    header += [
        "Latency is reported by the nodes themselves on `~/timing`, so it covers",
        "inference plus message conversion — not DDS transport or subscriber",
        "scheduling. The first samples are discarded while the CUDA graph and",
        "allocator settle.",
        "",
        "| Task | Model | Mean | p50 | p95 | p99 | Inference FPS |",
        "|------|-------|-----:|----:|----:|----:|--------------:|",
    ]

    for r in rows:
        header.append(
            f"| {r['task'].capitalize()} | `{r['model']}` | "
            f"{r['inference_ms_mean']} ms | {r['inference_ms_p50']} ms | "
            f"{r['inference_ms_p95']} ms | {r['inference_ms_p99']} ms | "
            f"**{r['inference_fps']}** |"
        )

    header += [
        "",
        "> `Inference FPS` is `1000 / mean`, i.e. the rate one node sustains on one",
        "> GPU with nothing else running. Achieved publish rate is bounded by the",
        "> input frame rate, so it is not a model-capacity figure.",
        "",
        "Reproduce with:",
        "",
        "```bash",
        "./scripts/run_benchmarks.sh /path/to/models /path/to/video.mp4 250",
        "```",
    ]

    table = "\n".join(header)
    readme = pathlib.Path(args.readme)
    text = readme.read_text()

    if MARKER not in text:
        print(f"{MARKER} not found in {readme}", file=sys.stderr)
        return 1

    start = text.index(MARKER)
    if END_MARKER in text:
        end = text.index(END_MARKER) + len(END_MARKER)
    else:
        end = start + len(MARKER)

    text = text[:start] + MARKER + "\n" + table + "\n" + END_MARKER + text[end:]
    readme.write_text(text)
    print(f"updated {readme} with {len(rows)} benchmark rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
