# ros2_yolos_trt — Design

**Date:** 2026-08-21
**Status:** Approved (approach B)

## Purpose

A ROS 2 wrapper around [YOLOs-CPP-TensorRT](https://github.com/Geekgineer/YOLOs-CPP-TensorRT)
**v3.1.0**, exposing its six inference tasks as lifecycle nodes. It is the
TensorRT sibling of `ros2_yolos_cpp`, which wraps the ONNX Runtime library.

This is an independent repository. It is not part of the YOLOs-CPP-TensorRT
project and must never be committed into it.

### Success criteria

1. `colcon build` succeeds on a machine with CUDA, TensorRT 10.x and ROS 2,
   with no manual prerequisite beyond those.
2. Each of the six tasks runs as a composable lifecycle node publishing
   standard `vision_msgs` / `sensor_msgs` types.
3. Depth output is directly consumable by existing ROS depth tooling.
4. A user can point a node at a `.onnx` file and get inference without
   knowing anything about TensorRT engines.
5. Inference results match the upstream library's own test expectations.

## Approach

Mirror `ros2_yolos_cpp`'s layering and file naming so the two packages are
navigable by the same mental model, but write the implementation for
TensorRT semantics rather than porting ONNX-era concepts.

Rejected alternatives: a straight fork (carries meaningless parameters such
as `use_gpu`, and six copies of lifecycle boilerplate), and a single
templated node (least code, but the per-task publisher and conversion
differences make the traits layer worse than the duplication it removes).

## Architecture

Four layers, each independently testable, with dependencies pointing one way:

```
sensor_msgs/Image
      |
   [ nodes ]        lifecycle, parameters, topics, timing
      |             depends on -> adapter interfaces only
      v
 [ adapters ]       wrap yolos:: classes behind pimpl
      |             the ONLY layer that includes TensorRT headers
      v
[ converters ]      DTO -> ROS message
      |
      v
  ROS messages
```

Plus one piece of shared infrastructure the ONNX package did not need:

```
[ engine_cache ]    .onnx -> .trt build + cache, used by every adapter
```

### Layer responsibilities

**`include/ros2_yolos_trt/adapters/yolos_adapter_base.hpp`** — library-agnostic
DTOs (`DetectionResult`, `PoseResult`, `DepthResult`, …) and abstract adapter
interfaces. No `yolos::` type appears here, so an upstream version bump touches
only adapter `.cpp` files. Carried over from `ros2_yolos_cpp` deliberately,
with `YolosConfig` changed as described under Configuration.

**`src/adapters/*.cpp`** — one adapter per task, pimpl so TensorRT headers stay
out of the public interface. Each calls `engine_cache` to resolve its
`model_path` to a loadable engine, then constructs the relevant `yolos::` class.

**`src/conversion/*.cpp`** — pure functions from DTO to ROS message. No ROS node
state, no library types; the easiest layer to unit-test and the only one tested
without a GPU.

**`src/nodes/*.cpp`** — six lifecycle nodes. Shared parameter declaration,
lifecycle transitions and timing publication live in
`YolosLifecycleNodeBase`, so each node contributes only its publisher types
and its one image callback.

### Engine cache

The genuinely new component, and the one that carries real risk.

`resolveEngine(model_path, precision, dla_core) -> std::filesystem::path`

- `.trt` / `.engine` input: returned unchanged.
- `.onnx` input: look for a cached engine; build it if absent, then return it.
- Cache key must include everything that invalidates an engine:
  compute capability, TensorRT version, precision, and the ONNX file's
  mtime+size. TensorRT engines are not portable across GPU or TRT version, and
  a stale engine fails in confusing ways, so the key is deliberately
  conservative — a false miss costs a rebuild, a false hit costs a debugging
  session.
- Cache location: `engine_cache_dir` if set, else alongside the ONNX file, else
  `$ROS_HOME/yolos_trt_engines`. Filename embeds the key.
- Building happens in `on_configure`, which can legitimately take minutes.
  The node logs progress and returns `FAILURE` on error, so lifecycle state
  reports the truth rather than a node that is up but broken.

Engine building reuses upstream's `NvOnnxParser` approach, the same logic as
the `onnx2trt` tool shipped in v3.1.0.

## Nodes and topics

| Node | Subscribes | Publishes |
|------|-----------|-----------|
| `yolos_trt_detector_node` | `image` | `detections` `vision_msgs/Detection2DArray` |
| `yolos_trt_segmentor_node` | `image` | `detections` + `masks` `sensor_msgs/Image` |
| `yolos_trt_pose_node` | `image` | `detections`, `keypoints` `KeyPoint2DArray` * |
| `yolos_trt_obb_node` | `image` | `detections` `OBBDetection2DArray` |
| `yolos_trt_classifier_node` | `image` | `classification` `vision_msgs/Classification` |
| `yolos_trt_depth_node` | `image` | `depth/image` `sensor_msgs/Image` **32FC1, metres** |

Every node additionally offers `debug_image` (`bgr8`, annotated) and `timing`
(`Float64MultiArray`), both off by default.

\* Marked because it is an addition, not a mirror — see below.

### Intentional deviations from ros2_yolos_cpp

**Pose publishes keypoints.** The reference package ships `KeyPoint2D.msg` and
`KeyPoint2DArray.msg`, and forward-declares `KeyPoint2DArray` in
`pose_converter.hpp`, but never implements or publishes the conversion. Its pose
node publishes only `Detection2DArray`, which has no keypoint field — so the
library computes keypoints and they are then discarded. Since keypoints are the
entire output of pose estimation, this package implements
`toKeyPoint2DArray()` and publishes them on `~/keypoints`. Boxes remain on
`~/detections` so a consumer written against the reference package still works.

**No `use_gpu` / `yolo_version` parameters.** Both are meaningless against this
backend, and silently-ignored parameters are worse than absent ones.

Depth publishes 32FC1 in metres because that is the ROS convention, which makes
it work unmodified with `depth_image_proc`, `rtabmap` and RViz. The upstream
model already emits metric depth, so no scaling is applied. The colorized view
goes to `debug_image`, never to `depth/image`.

## Configuration

```yaml
/**:
  ros__parameters:
    model_path: ""            # .onnx (auto-built) or .trt/.engine (loaded)
    labels_path: ""           # not required by the depth task
    precision: "fp16"         # fp32 | fp16 — used only when building
    engine_cache_dir: ""      # default: alongside the model
    dla_core: -1              # -1 = GPU, 0/1 = DLA on Jetson
    conf_threshold: 0.4
    nms_threshold: 0.45
    max_detections: 300       # OBB
    depth_min: 0.0            # debug colorization clamp, metres; 0 = auto
    depth_max: 0.0
    publish_debug_image: false
    publish_timing: false
```

Differences from `ros2_yolos_cpp`, each because the backend changed:
`use_gpu` is gone (TensorRT is always GPU), `yolo_version` is gone (the library
detects the version from output tensor shapes), and `precision`,
`engine_cache_dir`, `dla_core` are new.

## Build

`cmake_minimum_required(VERSION 3.18)` — required for first-class CUDA support.

Two ordering constraints matter, both learned from real failures upstream:

1. **`CMAKE_CUDA_ARCHITECTURES` must be set before `project()`.** `project(...
   LANGUAGES CUDA)` seeds it from the nvcc default (sm_52); a guard placed
   after `project()` is dead code. An sm_52 build of the letterbox kernel still
   runs on a modern GPU but returns wrong pixels, corrupting every task's
   output with no error. The build hard-fails if the value is `52`.
2. **TensorRT must be pinned to 10.x.** `libnvinfer-dev` now resolves to
   TensorRT 11.x built against CUDA 13, which does not compile against the TRT
   10 tensor API this library uses. The package detects the major version and
   fails configuration with an explanatory message rather than a wall of
   compiler errors.

Upstream is fetched at configure time from the **v3.1.0 tag** (not a branch, so
builds are reproducible). The package compiles
`cuda_preprocessing.cu` from that source into its own component library, since
the upstream library is header-only apart from that one translation unit.

## Testing

- **Unit, no GPU** — converter round-trips (DTO to message: geometry, class ids,
  encodings, empty inputs), and the engine cache key function (same inputs give
  the same key; changing GPU / TRT version / precision / model mtime changes
  it). These run in CI.
- **Integration, GPU required** — one per task, gated on env vars as in
  `ros2_yolos_cpp`. The depth test additionally asserts the published image is
  `32FC1`, matches the input dimensions, and carries plausible metric values,
  since a silently-wrong depth scale is the most likely regression.
- **CI** — build + unit tests on a standard runner; integration tests only where
  a GPU runner exists.

## README

Mirrors `ros2_yolos_cpp`'s structure, and leads with an RViz recording of the
depth node running the new YOLO26 depth model, since that is the capability
this package adds over its sibling. Recording procedure is scripted and
committed so the asset can be regenerated rather than being a one-off artifact.

## Out of scope

PointCloud2 projection from depth (needs CameraInfo intrinsics), INT8
calibration, batched multi-image topics, and tracking. Each is a separate
increment.
