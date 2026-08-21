# ros2_yolos_trt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A ROS 2 package exposing the six inference tasks of YOLOs-CPP-TensorRT v3.1.0 (detection, segmentation, pose, OBB, classification, metric depth) as composable lifecycle nodes.

**Architecture:** Four layers with one-way dependencies — lifecycle nodes depend only on abstract adapter interfaces; pimpl adapters are the sole layer that includes TensorRT headers; converters are pure DTO-to-message functions; a shared engine cache turns `.onnx` into a loadable `.trt`. Mirrors `ros2_yolos_cpp`'s layering and file naming so both packages read the same way.

**Tech Stack:** ROS 2 (rclcpp, rclcpp_lifecycle, rclcpp_components, cv_bridge, vision_msgs), TensorRT 10.x, CUDA 12.x, OpenCV 4.x, ament_cmake, GoogleTest.

**Spec:** `docs/superpowers/specs/2026-08-21-ros2-yolos-trt-design.md`

## Global Constraints

Every task's requirements implicitly include this section.

- **Package name / namespace:** `ros2_yolos_trt`. Include guards are `ROS2_YOLOS_TRT__<PATH>_HPP_`. Visibility macro is `ROS2_YOLOS_TRT_PUBLIC`.
- **Upstream dependency:** YOLOs-CPP-TensorRT at tag **`v3.1.0`** exactly. Never a branch — builds must be reproducible.
- **`CMAKE_CUDA_ARCHITECTURES` MUST be set before `project()`.** `project(... LANGUAGES CUDA)` seeds it from the nvcc default (sm_52), so a guard after `project()` is dead code. An sm_52 build of the letterbox kernel runs but returns wrong pixels, silently corrupting every task. Default `75;80;86;89;90`; Jetson `72;87`. Configuration hard-fails if the value is `52`.
- **TensorRT must be 10.x.** `libnvinfer-dev` now resolves to 11.x built against CUDA 13, which does not compile against the TRT 10 tensor API. Detect the major version from `NvInferVersion.h` and fail configuration with an explanatory message.
- **C++ standard:** C++17. **CMake:** >= 3.18 (first-class CUDA support).
- **License header on every source file:** `// Copyright 2026 YOLOs-TRT Team` then `// SPDX-License-Identifier: AGPL-3.0`.
- **No `use_gpu` or `yolo_version` parameters.** TensorRT is always GPU; the library detects version from output tensor shapes. Silently-ignored parameters are worse than absent ones.
- **Depth is published as `32FC1` in metres**, never colorized. Colorized output goes to `debug_image` only.
- **Commit after every task.** Conventional-commit subjects, no AI/tool attribution in messages.

## File Structure

```
package.xml                                     ament manifest, deps
CMakeLists.txt                                  build: arch-before-project, TRT pin, v3.1.0 fetch
config/default_params.yaml                      shared parameter defaults
msg/OrientedBoundingBox2D.msg                   OBB geometry
msg/OBBDetection2D.msg                          one OBB detection
msg/OBBDetection2DArray.msg                     OBB array
msg/KeyPoint2D.msg                              one pose keypoint
msg/KeyPoint2DArray.msg                         keypoints for one instance

include/ros2_yolos_trt/
  visibility_control.hpp                        export macros
  adapters/yolos_adapter_base.hpp               DTOs + abstract interfaces (no yolos:: types)
  adapters/{detector,segmentor,pose,obb,classifier,depth}_adapter.hpp
  conversion/{detection,segmentation,pose,obb,classification,depth}_converter.hpp
  engine/engine_cache.hpp                       .onnx -> .trt resolve + cache
  nodes/yolos_lifecycle_node_base.hpp           shared params/lifecycle/timing
  nodes/{detector,segmentor,pose,obb,classifier,depth}_node.hpp

src/
  engine/engine_cache.cpp                       the only novel infrastructure
  adapters/*.cpp                                pimpl; ONLY place TensorRT headers appear
  conversion/*.cpp                              pure functions
  nodes/*.cpp                                   lifecycle nodes

launch/{detector,segmentor,pose,obb,classifier,depth}.launch.py
test/unit/                                      no GPU: converters + cache key
test/integration/                               GPU-gated: one per task
scripts/record_depth_demo.sh                    regenerates the README RViz GIF
assets/depth_rviz.gif                           README asset
.github/workflows/ci.yml                        build + unit tests
README.md
```

Rationale for splits: converters are separated from adapters because they are the only layer testable without a GPU, and keeping them pure keeps CI meaningful. `engine_cache` is its own unit because it is the one genuinely new and risky component and deserves isolated tests. `yolos_lifecycle_node_base` exists because six nodes would otherwise carry six copies of identical parameter and lifecycle boilerplate.

---

### Task 1: Package skeleton with a CUDA-safe build

**Files:**
- Create: `package.xml`, `CMakeLists.txt`, `config/default_params.yaml`
- Create: `include/ros2_yolos_trt/visibility_control.hpp`
- Create: `.gitignore`
- Test: `test/unit/test_build_smoke.cpp`

**Interfaces:**
- Consumes: nothing.
- Produces: a buildable `ros2_yolos_trt_components` shared library target; CMake variables `YOLOS_TRT_DIR` (upstream source root) and `TENSORRT_MAJOR`.

- [ ] **Step 1: Write the failing test**

`test/unit/test_build_smoke.cpp` — asserts the upstream headers and the CUDA kernel declaration are reachable, which is what the whole build task is really delivering.

```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include <gtest/gtest.h>

#include <NvInferVersion.h>

#include "yolos/core/cuda_preprocessing.hpp"
#include "yolos/core/version.hpp"

// TensorRT 11.x removed API this library depends on; the build must pin 10.x.
TEST(BuildSmoke, TensorRtMajorVersionIsTen) {
  EXPECT_EQ(NV_TENSORRT_MAJOR, 10);
}

TEST(BuildSmoke, UpstreamVersionEnumIsReachable) {
  EXPECT_EQ(yolos::version::toString(yolos::YOLOVersion::V26), "V26");
}

TEST(BuildSmoke, CudaPreprocessDeclarationLinks) {
  // Only checks the symbol resolves at link time; no CUDA call is made.
  auto* fn = &yolos::cuda::letterboxPreprocess;
  EXPECT_NE(fn, nullptr);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `colcon build --packages-select ros2_yolos_trt`
Expected: FAIL — no `CMakeLists.txt` / `package.xml` yet, so colcon does not recognise the package.

- [ ] **Step 3: Write `package.xml`**

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>ros2_yolos_trt</name>
  <version>0.1.0</version>
  <description>
    ROS 2 wrapper for the YOLOs-CPP-TensorRT inference library. Exposes
    detection, segmentation, pose, OBB, classification and monocular metric
    depth as composable lifecycle nodes.
  </description>
  <maintainer email="abdalrahman.ibrahim@agilox.net">YOLOs-TRT Team</maintainer>
  <license>AGPL-3.0</license>

  <url type="repository">https://github.com/Geekgineer/ros2_yolos_trt</url>
  <url type="website">https://github.com/Geekgineer/YOLOs-CPP-TensorRT</url>

  <buildtool_depend>ament_cmake</buildtool_depend>
  <buildtool_depend>rosidl_default_generators</buildtool_depend>

  <depend>rclcpp</depend>
  <depend>rclcpp_lifecycle</depend>
  <depend>rclcpp_components</depend>
  <depend>image_transport</depend>
  <depend>cv_bridge</depend>
  <depend>sensor_msgs</depend>
  <depend>vision_msgs</depend>
  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>lifecycle_msgs</depend>

  <member_of_group>rosidl_interface_packages</member_of_group>
  <exec_depend>rosidl_default_runtime</exec_depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>
  <test_depend>ament_cmake_gtest</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

- [ ] **Step 4: Write `CMakeLists.txt`**

The architecture block MUST precede `project()`. Read the Global Constraints note before editing it.

```cmake
# =============================================================================
# ros2_yolos_trt - ROS 2 wrapper for YOLOs-CPP-TensorRT
# =============================================================================
cmake_minimum_required(VERSION 3.18)

# -----------------------------------------------------------------------------
# CUDA architectures — MUST be chosen BEFORE project()
# -----------------------------------------------------------------------------
# project(... LANGUAGES CUDA) seeds CMAKE_CUDA_ARCHITECTURES from the nvcc
# default (sm_52) when it is not already set, which makes any guard placed
# after project() dead code. An sm_52 build of the upstream letterbox kernel
# still runs on a modern GPU but returns WRONG pixel data, silently corrupting
# every task's output. Decide here, where the assignment takes effect.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64" AND
     (EXISTS "/etc/nv_tegra_release" OR EXISTS "/usr/lib/aarch64-linux-gnu/tegra"))
    set(CMAKE_CUDA_ARCHITECTURES "72;87")          # Jetson Xavier / Orin
  else()
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90") # Turing -> Hopper
  endif()
endif()

project(ros2_yolos_trt VERSION 0.1.0 LANGUAGES C CXX CUDA)

if(CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
  message(FATAL_ERROR
    "CMAKE_CUDA_ARCHITECTURES is 52 (the nvcc default). The upstream CUDA "
    "preprocessing kernel returns incorrect data when built for sm_52 and run "
    "on a modern GPU. Pass -DCMAKE_CUDA_ARCHITECTURES=<arch> explicitly.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# -----------------------------------------------------------------------------
# ROS 2 / system dependencies
# -----------------------------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(lifecycle_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# -----------------------------------------------------------------------------
# TensorRT — locate, then verify it is 10.x
# -----------------------------------------------------------------------------
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_DIR}/include $ENV{TENSORRT_DIR}/include
        /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu /usr/include)
find_library(NVINFER_LIB nvinfer
  HINTS ${TENSORRT_DIR}/lib $ENV{TENSORRT_DIR}/lib
        /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu)
find_library(NVONNXPARSER_LIB nvonnxparser
  HINTS ${TENSORRT_DIR}/lib $ENV{TENSORRT_DIR}/lib
        /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu)

if(NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER_LIB)
  message(FATAL_ERROR "TensorRT not found. Set -DTENSORRT_DIR=/path/to/tensorrt")
endif()
if(NOT NVONNXPARSER_LIB)
  message(FATAL_ERROR
    "libnvonnxparser not found. It is required to build engines from .onnx "
    "models. Install libnvonnxparser-dev (10.x).")
endif()

file(READ "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" _trt_version_header)
string(REGEX MATCH "define NV_TENSORRT_MAJOR ([0-9]+)" _ "${_trt_version_header}")
set(TENSORRT_MAJOR "${CMAKE_MATCH_1}")
if(NOT TENSORRT_MAJOR STREQUAL "10")
  message(FATAL_ERROR
    "Found TensorRT major version ${TENSORRT_MAJOR}, but 10.x is required. "
    "This package targets the TensorRT 10 tensor API (enqueueV3 / "
    "setTensorAddress); 11.x removed it. Pin apt with:\n"
    "  Package: libnvinfer* libnvonnxparsers* tensorrt*\n"
    "  Pin: version 10.*\n  Pin-Priority: 1001")
endif()
message(STATUS "TensorRT ${TENSORRT_MAJOR}.x: ${NVINFER_LIB}")

# -----------------------------------------------------------------------------
# Upstream YOLOs-CPP-TensorRT — fetched at the pinned release tag
# -----------------------------------------------------------------------------
set(YOLOS_TRT_VERSION "v3.1.0" CACHE STRING "YOLOs-CPP-TensorRT release tag")
set(YOLOS_TRT_DIR "${CMAKE_BINARY_DIR}/YOLOs-CPP-TensorRT")
set(YOLOS_TRT_URL
  "https://github.com/Geekgineer/YOLOs-CPP-TensorRT/archive/refs/tags/${YOLOS_TRT_VERSION}.tar.gz")

if(NOT EXISTS "${YOLOS_TRT_DIR}/include/yolos/yolos.hpp")
  message(STATUS "Downloading YOLOs-CPP-TensorRT ${YOLOS_TRT_VERSION}")
  file(DOWNLOAD "${YOLOS_TRT_URL}" "${CMAKE_BINARY_DIR}/yolos_trt.tar.gz"
       SHOW_PROGRESS STATUS _dl_status)
  list(GET _dl_status 0 _dl_code)
  if(NOT _dl_code EQUAL 0)
    message(FATAL_ERROR "Failed to download ${YOLOS_TRT_URL}: ${_dl_status}")
  endif()
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_BINARY_DIR}/yolos_trt.tar.gz"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}" RESULT_VARIABLE _x)
  if(NOT _x EQUAL 0)
    message(FATAL_ERROR "Failed to extract YOLOs-CPP-TensorRT")
  endif()
  file(GLOB _extracted "${CMAKE_BINARY_DIR}/YOLOs-CPP-TensorRT-*")
  list(GET _extracted 0 _extracted_dir)
  file(RENAME "${_extracted_dir}" "${YOLOS_TRT_DIR}")
  file(REMOVE "${CMAKE_BINARY_DIR}/yolos_trt.tar.gz")
endif()
message(STATUS "YOLOs-CPP-TensorRT source: ${YOLOS_TRT_DIR}")

# -----------------------------------------------------------------------------
# Messages
# -----------------------------------------------------------------------------
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/OrientedBoundingBox2D.msg"
  "msg/OBBDetection2D.msg"
  "msg/OBBDetection2DArray.msg"
  "msg/KeyPoint2D.msg"
  "msg/KeyPoint2DArray.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# -----------------------------------------------------------------------------
# Component library
# -----------------------------------------------------------------------------
# The upstream library is header-only APART FROM cuda_preprocessing.cu, which
# must be compiled here. This is why the package enables the CUDA language.
add_library(${PROJECT_NAME}_components SHARED
  ${YOLOS_TRT_DIR}/include/yolos/core/cuda_preprocessing.cu
)

target_include_directories(${PROJECT_NAME}_components PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_include_directories(${PROJECT_NAME}_components SYSTEM PUBLIC
  ${YOLOS_TRT_DIR}/include
  ${TENSORRT_INCLUDE_DIR}
)
target_link_libraries(${PROJECT_NAME}_components
  ${NVINFER_LIB} ${NVONNXPARSER_LIB} CUDA::cudart ${OpenCV_LIBS} Threads::Threads
)
set_target_properties(${PROJECT_NAME}_components PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
target_compile_definitions(${PROJECT_NAME}_components
  PRIVATE "ROS2_YOLOS_TRT_BUILDING_DLL")

ament_target_dependencies(${PROJECT_NAME}_components
  rclcpp rclcpp_lifecycle rclcpp_components image_transport cv_bridge
  sensor_msgs vision_msgs std_msgs geometry_msgs lifecycle_msgs)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(${PROJECT_NAME}_components "${cpp_typesupport_target}")

# -----------------------------------------------------------------------------
# Install
# -----------------------------------------------------------------------------
install(TARGETS ${PROJECT_NAME}_components
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch OPTIONAL)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config OPTIONAL)

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_build_smoke test/unit/test_build_smoke.cpp)
  target_link_libraries(test_build_smoke ${PROJECT_NAME}_components)
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME}_components)
ament_export_dependencies(rclcpp rclcpp_lifecycle rclcpp_components
  image_transport cv_bridge sensor_msgs vision_msgs std_msgs geometry_msgs)
ament_package()

message(STATUS "")
message(STATUS "=== ros2_yolos_trt ===")
message(STATUS "  Upstream:           ${YOLOS_TRT_VERSION}")
message(STATUS "  TensorRT:           ${TENSORRT_MAJOR}.x  ${NVINFER_LIB}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  OpenCV:             ${OpenCV_VERSION}")
message(STATUS "")
```

- [ ] **Step 5: Write `include/ros2_yolos_trt/visibility_control.hpp`**

```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ROS2_YOLOS_TRT__VISIBILITY_CONTROL_HPP_
#define ROS2_YOLOS_TRT__VISIBILITY_CONTROL_HPP_

#if defined _WIN32 || defined __CYGWIN__
  #ifdef __GNUC__
    #define ROS2_YOLOS_TRT_EXPORT __attribute__((dllexport))
    #define ROS2_YOLOS_TRT_IMPORT __attribute__((dllimport))
  #else
    #define ROS2_YOLOS_TRT_EXPORT __declspec(dllexport)
    #define ROS2_YOLOS_TRT_IMPORT __declspec(dllimport)
  #endif
  #ifdef ROS2_YOLOS_TRT_BUILDING_DLL
    #define ROS2_YOLOS_TRT_PUBLIC ROS2_YOLOS_TRT_EXPORT
  #else
    #define ROS2_YOLOS_TRT_PUBLIC ROS2_YOLOS_TRT_IMPORT
  #endif
  #define ROS2_YOLOS_TRT_LOCAL
#else
  #define ROS2_YOLOS_TRT_EXPORT __attribute__((visibility("default")))
  #define ROS2_YOLOS_TRT_IMPORT
  #define ROS2_YOLOS_TRT_PUBLIC __attribute__((visibility("default")))
  #define ROS2_YOLOS_TRT_LOCAL  __attribute__((visibility("hidden")))
#endif

#endif  // ROS2_YOLOS_TRT__VISIBILITY_CONTROL_HPP_
```

- [ ] **Step 6: Write `config/default_params.yaml`**

```yaml
# Default parameters for ros2_yolos_trt nodes.
# Override via launch arguments or a parameter YAML.
/**:
  ros__parameters:
    # Model. Accepts either a .onnx (an engine is built and cached on first
    # configure) or a prebuilt .trt/.engine (loaded directly).
    model_path: ""
    labels_path: ""          # not required by the depth task
    precision: "fp16"        # fp32 | fp16 — used only when building an engine
    engine_cache_dir: ""     # default: alongside the model file
    dla_core: -1             # -1 = GPU, 0/1 = DLA on Jetson

    # Thresholds
    conf_threshold: 0.4
    nms_threshold: 0.45
    max_detections: 300      # OBB only

    # Depth debug colorization clamp, metres. 0 = auto-range per frame.
    depth_min: 0.0
    depth_max: 0.0

    # Diagnostics
    publish_debug_image: false
    publish_timing: false
```

- [ ] **Step 7: Write `.gitignore`**

```gitignore
build/
install/
log/
__pycache__/
*.trt
*.engine
*.onnx
```

- [ ] **Step 8: Create empty msg files so message generation configures**

The five message bodies are written in Task 2; create them now with a single comment line each so `rosidl_generate_interfaces` does not fail this task's build.

```bash
mkdir -p msg
for m in OrientedBoundingBox2D OBBDetection2D OBBDetection2DArray KeyPoint2D KeyPoint2DArray; do
  echo "# placeholder — defined in Task 2" > "msg/${m}.msg"
done
```

Note: an empty `.msg` is valid and generates an empty struct, so the build succeeds. Task 2 replaces all five.

- [ ] **Step 9: Run the test to verify it passes**

Run:
```bash
colcon build --packages-select ros2_yolos_trt
colcon test --packages-select ros2_yolos_trt --ctest-args -R test_build_smoke
colcon test-result --verbose
```
Expected: build succeeds; `test_build_smoke` passes all three cases. The configure log must show `CUDA architectures: 75;80;86;89;90` (or the Jetson pair) — **if it shows `52`, the build is wrong**; check that the architecture block precedes `project()`.

- [ ] **Step 10: Commit**

```bash
git add package.xml CMakeLists.txt config .gitignore msg include/ros2_yolos_trt/visibility_control.hpp test/unit/test_build_smoke.cpp
git commit -m "feat: package skeleton with CUDA-safe TensorRT build

Fetches YOLOs-CPP-TensorRT at the pinned v3.1.0 tag and compiles its
cuda_preprocessing.cu translation unit into the component library.

Two guards encode failures seen upstream: CUDA architectures are chosen
before project(), because project(LANGUAGES CUDA) otherwise seeds them
from the nvcc default of sm_52 and an sm_52 letterbox kernel silently
returns wrong pixels; and configuration fails unless TensorRT is 10.x,
since 11.x removed the tensor API this library uses."
```

---
### Task 2: Custom messages

**Files:**
- Modify: `msg/OrientedBoundingBox2D.msg`, `msg/OBBDetection2D.msg`, `msg/OBBDetection2DArray.msg`, `msg/KeyPoint2D.msg`, `msg/KeyPoint2DArray.msg` (all created as placeholders in Task 1)
- Test: `test/unit/test_messages.cpp`
- Modify: `CMakeLists.txt` (register the new gtest)

**Interfaces:**
- Consumes: the component library target from Task 1.
- Produces: C++ types `ros2_yolos_trt::msg::OrientedBoundingBox2D`, `OBBDetection2D`, `OBBDetection2DArray`, `KeyPoint2D`, `KeyPoint2DArray`.

- [ ] **Step 1: Write the failing test**

`test/unit/test_messages.cpp`

```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include <gtest/gtest.h>

#include "ros2_yolos_trt/msg/key_point2_d.hpp"
#include "ros2_yolos_trt/msg/key_point2_d_array.hpp"
#include "ros2_yolos_trt/msg/obb_detection2_d.hpp"
#include "ros2_yolos_trt/msg/obb_detection2_d_array.hpp"
#include "ros2_yolos_trt/msg/oriented_bounding_box2_d.hpp"

TEST(Messages, OrientedBoundingBoxHoldsGeometry) {
  ros2_yolos_trt::msg::OrientedBoundingBox2D obb;
  obb.center.x = 10.0;
  obb.center.y = 20.0;
  obb.size_x = 30.0;
  obb.size_y = 40.0;
  obb.theta = 0.5;
  EXPECT_DOUBLE_EQ(obb.center.x, 10.0);
  EXPECT_DOUBLE_EQ(obb.theta, 0.5);
}

TEST(Messages, OBBDetectionArrayNestsDetections) {
  ros2_yolos_trt::msg::OBBDetection2DArray arr;
  ros2_yolos_trt::msg::OBBDetection2D det;
  det.class_id = 3;
  det.class_name = "ship";
  det.score = 0.9;
  arr.detections.push_back(det);
  ASSERT_EQ(arr.detections.size(), 1u);
  EXPECT_EQ(arr.detections[0].class_name, "ship");
}

TEST(Messages, KeyPointArrayNestsKeypoints) {
  ros2_yolos_trt::msg::KeyPoint2DArray arr;
  ros2_yolos_trt::msg::KeyPoint2D kp;
  kp.x = 1.0;
  kp.y = 2.0;
  kp.confidence = 0.8;
  kp.id = 5;
  arr.keypoints.push_back(kp);
  arr.class_id = 0;
  ASSERT_EQ(arr.keypoints.size(), 1u);
  EXPECT_EQ(arr.keypoints[0].id, 5);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `colcon build --packages-select ros2_yolos_trt`
Expected: FAIL — the placeholder messages have no fields, so `obb.center`, `arr.detections`, `arr.keypoints` do not compile.

- [ ] **Step 3: Write the message definitions**

`msg/OrientedBoundingBox2D.msg`
```
# Oriented (rotated) bounding box in 2D image space.

# Centre point of the box (z unused).
geometry_msgs/Point center

# Extent before rotation, pixels.
float64 size_x
float64 size_y

# Rotation, radians, counter-clockwise from the x-axis.
float64 theta
```

`msg/OBBDetection2D.msg`
```
# A single oriented bounding box detection.

std_msgs/Header header

ros2_yolos_trt/OrientedBoundingBox2D bbox

int32 class_id
string class_name

# Confidence, 0.0 - 1.0.
float64 score
```

`msg/OBBDetection2DArray.msg`
```
# All oriented bounding box detections for one image.

std_msgs/Header header

ros2_yolos_trt/OBBDetection2D[] detections
```

`msg/KeyPoint2D.msg`
```
# A single 2D pose keypoint.

# Position in pixels.
float64 x
float64 y

# Confidence, 0.0 - 1.0.
float64 confidence

# Keypoint index (COCO ordering for the released pose models).
int32 id
```

`msg/KeyPoint2DArray.msg`
```
# Keypoints belonging to one detected instance.

std_msgs/Header header

ros2_yolos_trt/KeyPoint2D[] keypoints

# Bounding box of the instance these keypoints belong to.
geometry_msgs/Point bounding_box_center
float64 bounding_box_size_x
float64 bounding_box_size_y

# Detection confidence for the instance.
float64 confidence

# Class id (0 = person for the released pose models).
int32 class_id
```

- [ ] **Step 4: Register the test in `CMakeLists.txt`**

Inside the existing `if(BUILD_TESTING)` block, after the `test_build_smoke` lines:

```cmake
  ament_add_gtest(test_messages test/unit/test_messages.cpp)
  target_link_libraries(test_messages ${PROJECT_NAME}_components "${cpp_typesupport_target}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
colcon build --packages-select ros2_yolos_trt
colcon test --packages-select ros2_yolos_trt --ctest-args -R test_messages
colcon test-result --verbose
```
Expected: PASS, 3 cases.

- [ ] **Step 6: Commit**

```bash
git add msg test/unit/test_messages.cpp CMakeLists.txt
git commit -m "feat: add OBB and keypoint messages

vision_msgs has no oriented-bounding-box or keypoint type, so OBB and pose
results need package-local messages to be publishable without discarding
the rotation angle or the keypoints themselves."
```

---

### Task 3: Adapter interfaces and DTOs

**Files:**
- Create: `include/ros2_yolos_trt/adapters/yolos_adapter_base.hpp`
- Test: `test/unit/test_dtos.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `visibility_control.hpp` from Task 1.
- Produces: `YolosConfig`; DTOs `BoundingBox2D`, `OrientedBoundingBox2D`, `KeyPoint2D`, `DetectionResult`, `SegmentationResult`, `PoseResult`, `OBBResult`, `ClassificationResult`, `DepthResult`; interfaces `IYolosAdapterBase`, `IDetectorAdapter`, `ISegmentorAdapter`, `IPoseAdapter`, `IOBBAdapter`, `IClassifierAdapter`, `IDepthAdapter`; factories `createDetectorAdapter()`, `createSegmentorAdapter()`, `createPoseAdapter()`, `createOBBAdapter()`, `createClassifierAdapter()`, `createDepthAdapter()`.

Every later adapter and converter task depends on the exact names in this file.

- [ ] **Step 1: Write the failing test**

`test/unit/test_dtos.cpp`

```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include <gtest/gtest.h>

#include "ros2_yolos_trt/adapters/yolos_adapter_base.hpp"

using ros2_yolos_trt::DepthResult;
using ros2_yolos_trt::DetectionResult;
using ros2_yolos_trt::YolosConfig;

TEST(Dtos, ConfigDefaultsAreTensorRtAppropriate) {
  YolosConfig cfg;
  // No use_gpu field exists: TensorRT is always GPU. dla_core -1 means GPU.
  EXPECT_EQ(cfg.dla_core, -1);
  EXPECT_EQ(cfg.precision, "fp16");
  EXPECT_FLOAT_EQ(cfg.conf_threshold, 0.4f);
  EXPECT_FLOAT_EQ(cfg.nms_threshold, 0.45f);
}

TEST(Dtos, DetectionResultHoldsBoxAndClass) {
  DetectionResult d;
  d.bbox.x = 5;
  d.bbox.width = 20;
  d.confidence = 0.75f;
  d.class_id = 2;
  d.class_name = "car";
  EXPECT_EQ(d.bbox.x, 5);
  EXPECT_EQ(d.class_name, "car");
}

TEST(Dtos, DepthResultCarriesMetricMap) {
  DepthResult r;
  r.depth_metres = cv::Mat::zeros(4, 6, CV_32FC1);
  r.min_metres = 1.5f;
  r.max_metres = 9.0f;
  EXPECT_EQ(r.depth_metres.type(), CV_32FC1);
  EXPECT_EQ(r.depth_metres.cols, 6);
  EXPECT_FLOAT_EQ(r.min_metres, 1.5f);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `colcon build --packages-select ros2_yolos_trt`
Expected: FAIL — `yolos_adapter_base.hpp` does not exist.

- [ ] **Step 3: Write `include/ros2_yolos_trt/adapters/yolos_adapter_base.hpp`**

```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ROS2_YOLOS_TRT__ADAPTERS__YOLOS_ADAPTER_BASE_HPP_
#define ROS2_YOLOS_TRT__ADAPTERS__YOLOS_ADAPTER_BASE_HPP_

// =============================================================================
// Abstract adapter interfaces and library-agnostic DTOs
// =============================================================================
// No yolos:: type appears in this header. Nodes depend on it, adapters
// implement it, and an upstream version bump therefore touches only adapter
// .cpp files.
// =============================================================================

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "ros2_yolos_trt/visibility_control.hpp"

namespace ros2_yolos_trt {

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

/// @brief Configuration common to every adapter.
///
/// There is deliberately no use_gpu field: TensorRT execution is always on the
/// GPU (or a DLA core). There is deliberately no yolo_version field either:
/// the library detects the version from output tensor shapes.
struct YolosConfig {
  /// .onnx (an engine is built and cached) or .trt/.engine (loaded directly).
  std::string model_path;
  std::string labels_path;
  /// "fp32" or "fp16". Used only when building an engine from .onnx.
  std::string precision{"fp16"};
  /// Empty means "alongside the model file".
  std::string engine_cache_dir;
  /// -1 = GPU, 0/1 = DLA core on Jetson.
  int dla_core{-1};
  float conf_threshold{0.4f};
  float nms_threshold{0.45f};
  int max_detections{300};
};

// -----------------------------------------------------------------------------
// Result DTOs
// -----------------------------------------------------------------------------

struct BoundingBox2D {
  int x{0};
  int y{0};
  int width{0};
  int height{0};
};

struct OrientedBoundingBox2D {
  float center_x{0.0f};
  float center_y{0.0f};
  float width{0.0f};
  float height{0.0f};
  float angle{0.0f};  ///< radians
};

struct KeyPoint2D {
  float x{0.0f};
  float y{0.0f};
  float confidence{0.0f};
  int id{0};
};

struct DetectionResult {
  BoundingBox2D bbox;
  float confidence{0.0f};
  int class_id{0};
  std::string class_name;
};

struct SegmentationResult {
  BoundingBox2D bbox;
  float confidence{0.0f};
  int class_id{0};
  std::string class_name;
  cv::Mat mask;  ///< CV_8UC1 binary mask, original image size
};

struct PoseResult {
  BoundingBox2D bbox;
  float confidence{0.0f};
  int class_id{0};
  std::string class_name;
  std::vector<KeyPoint2D> keypoints;
};

struct OBBResult {
  OrientedBoundingBox2D bbox;
  float confidence{0.0f};
  int class_id{0};
  std::string class_name;
};

struct ClassificationResult {
  int class_id{0};
  float confidence{0.0f};
  std::string class_name;
};

/// @brief Monocular metric depth for one image.
struct DepthResult {
  /// CV_32FC1 at the original image size. Values are METRES, already
  /// calibrated by the exported model; no scaling is applied anywhere.
  cv::Mat depth_metres;
  float min_metres{0.0f};
  float max_metres{0.0f};
};

// -----------------------------------------------------------------------------
// Interfaces
// -----------------------------------------------------------------------------

class ROS2_YOLOS_TRT_PUBLIC IYolosAdapterBase {
public:
  virtual ~IYolosAdapterBase() = default;

  /// @brief Load the model, building an engine from .onnx if required.
  /// @return false on any failure; the caller reports lifecycle FAILURE.
  virtual bool initialize(const YolosConfig& config) = 0;

  [[nodiscard]] virtual bool isInitialized() const noexcept = 0;
  virtual void shutdown() = 0;
  [[nodiscard]] virtual const std::vector<std::string>& getClassNames() const = 0;
};

class ROS2_YOLOS_TRT_PUBLIC IDetectorAdapter : public IYolosAdapterBase {
public:
  virtual std::vector<DetectionResult> detect(
    const cv::Mat& image, float conf_threshold, float nms_threshold) = 0;
  virtual void drawDetections(
    cv::Mat& image, const std::vector<DetectionResult>& detections) = 0;
};

class ROS2_YOLOS_TRT_PUBLIC ISegmentorAdapter : public IYolosAdapterBase {
public:
  virtual std::vector<SegmentationResult> segment(
    const cv::Mat& image, float conf_threshold, float nms_threshold) = 0;
  virtual void drawSegmentations(
    cv::Mat& image, const std::vector<SegmentationResult>& segmentations,
    float mask_alpha = 0.5f) = 0;
};

class ROS2_YOLOS_TRT_PUBLIC IPoseAdapter : public IYolosAdapterBase {
public:
  virtual std::vector<PoseResult> detect(
    const cv::Mat& image, float conf_threshold, float nms_threshold) = 0;
  virtual void drawPoses(
    cv::Mat& image, const std::vector<PoseResult>& poses,
    int kpt_radius = 4, float kpt_threshold = 0.5f) = 0;
};

class ROS2_YOLOS_TRT_PUBLIC IOBBAdapter : public IYolosAdapterBase {
public:
  virtual std::vector<OBBResult> detect(
    const cv::Mat& image, float conf_threshold, float nms_threshold,
    int max_det = 300) = 0;
  virtual void drawDetections(
    cv::Mat& image, const std::vector<OBBResult>& detections) = 0;
};

class ROS2_YOLOS_TRT_PUBLIC IClassifierAdapter : public IYolosAdapterBase {
public:
  virtual ClassificationResult classify(const cv::Mat& image) = 0;
  virtual void drawResult(cv::Mat& image, const ClassificationResult& result) = 0;
};

class ROS2_YOLOS_TRT_PUBLIC IDepthAdapter : public IYolosAdapterBase {
public:
  /// @brief Estimate metric depth. Returns an empty depth_metres on failure.
  virtual DepthResult estimate(const cv::Mat& image) = 0;

  /// @brief Colorized visualization for the debug topic only.
  /// @param min_metres,max_metres clamp range; <= 0 means auto-range.
  virtual cv::Mat colorize(
    const DepthResult& result, float min_metres, float max_metres) = 0;
};

// -----------------------------------------------------------------------------
// Factories
// -----------------------------------------------------------------------------

ROS2_YOLOS_TRT_PUBLIC std::unique_ptr<IDetectorAdapter>   createDetectorAdapter();
ROS2_YOLOS_TRT_PUBLIC std::unique_ptr<ISegmentorAdapter>  createSegmentorAdapter();
ROS2_YOLOS_TRT_PUBLIC std::unique_ptr<IPoseAdapter>       createPoseAdapter();
ROS2_YOLOS_TRT_PUBLIC std::unique_ptr<IOBBAdapter>        createOBBAdapter();
ROS2_YOLOS_TRT_PUBLIC std::unique_ptr<IClassifierAdapter> createClassifierAdapter();
ROS2_YOLOS_TRT_PUBLIC std::unique_ptr<IDepthAdapter>      createDepthAdapter();

}  // namespace ros2_yolos_trt

#endif  // ROS2_YOLOS_TRT__ADAPTERS__YOLOS_ADAPTER_BASE_HPP_
```

- [ ] **Step 4: Register the test in `CMakeLists.txt`**

```cmake
  ament_add_gtest(test_dtos test/unit/test_dtos.cpp)
  target_link_libraries(test_dtos ${PROJECT_NAME}_components)
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
colcon build --packages-select ros2_yolos_trt
colcon test --packages-select ros2_yolos_trt --ctest-args -R test_dtos
colcon test-result --verbose
```
Expected: PASS, 3 cases.

- [ ] **Step 6: Commit**

```bash
git add include/ros2_yolos_trt/adapters/yolos_adapter_base.hpp test/unit/test_dtos.cpp CMakeLists.txt
git commit -m "feat: add library-agnostic adapter interfaces and DTOs

Nodes depend on these interfaces rather than on yolos:: types, so an
upstream version bump touches only adapter implementations.

Drops use_gpu and yolo_version relative to the ONNX package: TensorRT is
always GPU, and the library detects the model version from output tensor
shapes, so both would be parameters that silently do nothing. Adds
precision, engine_cache_dir and dla_core, which TensorRT does need."
```

---

### Task 4: Engine cache

The one genuinely new component, and the highest-risk one: a stale or mismatched engine fails in confusing ways at load time.

**Files:**
- Create: `include/ros2_yolos_trt/engine/engine_cache.hpp`, `src/engine/engine_cache.cpp`
- Test: `test/unit/test_engine_cache.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `std::string ros2_yolos_trt::engine::cacheKey(const std::string& onnx_path, const std::string& precision, int dla_core)`
  - `std::filesystem::path ros2_yolos_trt::engine::resolveEngine(const std::string& model_path, const std::string& precision, int dla_core, const std::string& cache_dir, std::string& error_out)`
  - `bool ros2_yolos_trt::engine::isEnginePath(const std::string& path)`

- [ ] **Step 1: Write the failing test**

Only the pure logic is unit-tested; actually building an engine needs a GPU and is covered by the integration tests in Task 12.

`test/unit/test_engine_cache.cpp`

```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "ros2_yolos_trt/engine/engine_cache.hpp"

namespace fs = std::filesystem;
using ros2_yolos_trt::engine::cacheKey;
using ros2_yolos_trt::engine::isEnginePath;
using ros2_yolos_trt::engine::resolveEngine;

namespace {
fs::path makeTempOnnx(const std::string& name, const std::string& contents) {
  const fs::path p = fs::temp_directory_path() / name;
  std::ofstream f(p, std::ios::binary);
  f << contents;
  f.close();
  return p;
}
}  // namespace

TEST(EngineCache, RecognisesEngineExtensions) {
  EXPECT_TRUE(isEnginePath("/models/a.trt"));
  EXPECT_TRUE(isEnginePath("/models/a.engine"));
  EXPECT_FALSE(isEnginePath("/models/a.onnx"));
  EXPECT_FALSE(isEnginePath("/models/a"));
}

TEST(EngineCache, KeyIsStableForIdenticalInputs) {
  const auto onnx = makeTempOnnx("ec_stable.onnx", "abc");
  EXPECT_EQ(cacheKey(onnx.string(), "fp16", -1),
            cacheKey(onnx.string(), "fp16", -1));
}

TEST(EngineCache, KeyChangesWithPrecision) {
  const auto onnx = makeTempOnnx("ec_prec.onnx", "abc");
  EXPECT_NE(cacheKey(onnx.string(), "fp16", -1),
            cacheKey(onnx.string(), "fp32", -1));
}

TEST(EngineCache, KeyChangesWithDlaCore) {
  const auto onnx = makeTempOnnx("ec_dla.onnx", "abc");
  EXPECT_NE(cacheKey(onnx.string(), "fp16", -1),
            cacheKey(onnx.string(), "fp16", 0));
}

// An engine built from different model bytes must not be reused, or the node
// silently runs the wrong network.
TEST(EngineCache, KeyChangesWithModelContents) {
  const auto a = makeTempOnnx("ec_a.onnx", "aaaa");
  const auto b = makeTempOnnx("ec_b.onnx", "bbbbbbbb");
  EXPECT_NE(cacheKey(a.string(), "fp16", -1), cacheKey(b.string(), "fp16", -1));
}

TEST(EngineCache, PrebuiltEnginePathIsReturnedUnchanged) {
  std::string err;
  const auto out = resolveEngine("/models/x.trt", "fp16", -1, "", err);
  EXPECT_EQ(out.string(), "/models/x.trt");
  EXPECT_TRUE(err.empty());
}

TEST(EngineCache, MissingModelReportsAnError) {
  std::string err;
  const auto out = resolveEngine("/nope/missing.onnx", "fp16", -1, "", err);
  EXPECT_TRUE(out.empty());
  EXPECT_NE(err.find("not found"), std::string::npos);
}

TEST(EngineCache, UnsupportedExtensionReportsAnError) {
  const auto bad = makeTempOnnx("ec_bad.pt", "x");
  std::string err;
  const auto out = resolveEngine(bad.string(), "fp16", -1, "", err);
  EXPECT_TRUE(out.empty());
  EXPECT_NE(err.find(".onnx"), std::string::npos);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `colcon build --packages-select ros2_yolos_trt`
Expected: FAIL — `engine_cache.hpp` does not exist.

- [ ] **Step 3: Write `include/ros2_yolos_trt/engine/engine_cache.hpp`**

```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ROS2_YOLOS_TRT__ENGINE__ENGINE_CACHE_HPP_
#define ROS2_YOLOS_TRT__ENGINE__ENGINE_CACHE_HPP_

#include <filesystem>
#include <string>

#include "ros2_yolos_trt/visibility_control.hpp"

namespace ros2_yolos_trt {
namespace engine {

/// @brief True if the path names an already-serialized engine.
ROS2_YOLOS_TRT_PUBLIC bool isEnginePath(const std::string& path);

/// @brief Identity of an engine built from this model in this configuration.
///
/// TensorRT engines are portable across neither GPU architecture nor TensorRT
/// version, so both are part of the key, along with the build precision, the
/// DLA core and the model's own size and modification time. The key is
/// deliberately conservative: a false miss costs one rebuild, a false hit
/// costs a debugging session.
ROS2_YOLOS_TRT_PUBLIC std::string cacheKey(
  const std::string& onnx_path, const std::string& precision, int dla_core);

/// @brief Resolve a model path to a loadable engine, building one if needed.
///
/// - `.trt` / `.engine` input is returned unchanged.
/// - `.onnx` input returns a cached engine, building it first if absent.
///
/// @param cache_dir Directory for built engines; empty means alongside the model.
/// @param error_out Set to a human-readable reason when the return value is empty.
/// @return Path to a loadable engine, or an empty path on failure.
ROS2_YOLOS_TRT_PUBLIC std::filesystem::path resolveEngine(
  const std::string& model_path, const std::string& precision, int dla_core,
  const std::string& cache_dir, std::string& error_out);

}  // namespace engine
}  // namespace ros2_yolos_trt

#endif  // ROS2_YOLOS_TRT__ENGINE__ENGINE_CACHE_HPP_
```

- [ ] **Step 4: Write `src/engine/engine_cache.cpp`**

```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_trt/engine/engine_cache.hpp"

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

namespace fs = std::filesystem;

namespace ros2_yolos_trt {
namespace engine {
namespace {

class BuildLogger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cout << "[engine_cache][TRT] " << msg << std::endl;
    }
  }
};

std::string lowerExtension(const std::string& path) {
  std::string ext = fs::path(path).extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return ext;
}

/// Compute capability of device 0, e.g. "sm89". Empty when unavailable.
std::string computeCapability() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return {};
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return {};
  return "sm" + std::to_string(prop.major) + std::to_string(prop.minor);
}

/// FNV-1a: short, dependency-free, and adequate for a cache filename.
std::string hashString(const std::string& input) {
  std::uint64_t h = 1469598103934665603ULL;
  for (unsigned char c : input) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  std::ostringstream ss;
  ss << std::hex << h;
  return ss.str();
}

bool buildEngine(const fs::path& onnx, const fs::path& out,
                 const std::string& precision, std::string& error_out) {
  BuildLogger logger;

  auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
  if (!builder) {
    error_out = "failed to create TensorRT builder";
    return false;
  }

  auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
  auto parser = std::unique_ptr<nvonnxparser::IParser>(
    nvonnxparser::createParser(*network, logger));

  if (!parser->parseFromFile(
        onnx.string().c_str(),
        static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    std::ostringstream ss;
    ss << "failed to parse " << onnx;
    for (int i = 0; i < parser->getNbErrors(); ++i) {
      ss << "; " << parser->getError(i)->desc();
    }
    error_out = ss.str();
    return false;
  }

  auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                             4096ULL * 1024ULL * 1024ULL);
  if (precision == "fp16") {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }

  auto plan = std::unique_ptr<nvinfer1::IHostMemory>(
    builder->buildSerializedNetwork(*network, *config));
  if (!plan) {
    error_out = "TensorRT engine build failed";
    return false;
  }

  // Write to a temporary sibling then rename, so a crash mid-write cannot
  // leave a truncated engine that later looks like a valid cache hit.
  const fs::path tmp = out.string() + ".partial";
  {
    std::ofstream f(tmp, std::ios::binary);
    if (!f) {
      error_out = "cannot write " + tmp.string();
      return false;
    }
    f.write(static_cast<const char*>(plan->data()),
            static_cast<std::streamsize>(plan->size()));
  }
  std::error_code ec;
  fs::rename(tmp, out, ec);
  if (ec) {
    error_out = "cannot move engine into place: " + ec.message();
    return false;
  }
  return true;
}

}  // namespace

bool isEnginePath(const std::string& path) {
  const std::string ext = lowerExtension(path);
  return ext == ".trt" || ext == ".engine";
}

std::string cacheKey(const std::string& onnx_path, const std::string& precision,
                     int dla_core) {
  std::error_code ec;
  const auto size = fs::file_size(onnx_path, ec);
  const std::uintmax_t size_val = ec ? 0 : size;

  std::uint64_t mtime = 0;
  const auto t = fs::last_write_time(onnx_path, ec);
  if (!ec) {
    mtime = static_cast<std::uint64_t>(t.time_since_epoch().count());
  }

  std::ostringstream ss;
  ss << fs::path(onnx_path).filename().string() << '|' << size_val << '|'
     << mtime << '|' << precision << '|' << dla_core << '|'
     << computeCapability() << '|' << NV_TENSORRT_MAJOR << '.'
     << NV_TENSORRT_MINOR << '.' << NV_TENSORRT_PATCH;
  return hashString(ss.str());
}

fs::path resolveEngine(const std::string& model_path, const std::string& precision,
                       int dla_core, const std::string& cache_dir,
                       std::string& error_out) {
  error_out.clear();

  if (model_path.empty()) {
    error_out = "model_path is empty";
    return {};
  }
  if (isEnginePath(model_path)) {
    // Existence is checked by the library when it loads the engine, so that
    // the error surfaces with its context rather than being duplicated here.
    return fs::path(model_path);
  }
  if (lowerExtension(model_path) != ".onnx") {
    error_out = "model_path must be a .onnx, .trt or .engine file, got " + model_path;
    return {};
  }

  std::error_code ec;
  if (!fs::exists(model_path, ec)) {
    error_out = "model not found: " + model_path;
    return {};
  }

  fs::path dir = cache_dir.empty() ? fs::path(model_path).parent_path()
                                   : fs::path(cache_dir);
  if (dir.empty()) dir = fs::current_path();
  fs::create_directories(dir, ec);

  const std::string stem = fs::path(model_path).stem().string();
  const fs::path engine_path =
    dir / (stem + "." + cacheKey(model_path, precision, dla_core) + ".trt");

  if (fs::exists(engine_path, ec)) {
    std::cout << "[engine_cache] cache hit: " << engine_path << std::endl;
    return engine_path;
  }

  std::cout << "[engine_cache] building engine from " << model_path
            << " (this can take several minutes)" << std::endl;
  if (!buildEngine(model_path, engine_path, precision, error_out)) {
    return {};
  }
  std::cout << "[engine_cache] wrote " << engine_path << std::endl;
  return engine_path;
}

}  // namespace engine
}  // namespace ros2_yolos_trt
```

- [ ] **Step 5: Add the source and test to `CMakeLists.txt`**

Add to the `add_library(${PROJECT_NAME}_components SHARED ...)` source list, after the `.cu` entry:

```cmake
  src/engine/engine_cache.cpp
```

And inside `if(BUILD_TESTING)`:

```cmake
  ament_add_gtest(test_engine_cache test/unit/test_engine_cache.cpp)
  target_link_libraries(test_engine_cache ${PROJECT_NAME}_components)
```

- [ ] **Step 6: Run tests to verify they pass**

Run:
```bash
colcon build --packages-select ros2_yolos_trt
colcon test --packages-select ros2_yolos_trt --ctest-args -R test_engine_cache
colcon test-result --verbose
```
Expected: PASS, 8 cases. These run without a GPU; `computeCapability()` returns an empty string there, which is fine because the key only needs to be *consistent*.

- [ ] **Step 7: Commit**

```bash
git add include/ros2_yolos_trt/engine src/engine test/unit/test_engine_cache.cpp CMakeLists.txt
git commit -m "feat: add engine cache resolving .onnx to a built .trt

Lets a node be pointed at an .onnx and work, which matters when one model
is deployed across a fleet of mixed GPUs — TensorRT engines are portable
across neither GPU architecture nor TensorRT version.

The cache key covers compute capability, TensorRT version, precision, DLA
core, and the model's size and mtime. It is deliberately conservative: a
false miss costs a rebuild, a false hit runs the wrong network. Engines are
written to a .partial file and renamed, so an interrupted build cannot leave
a truncated file that later reads as a cache hit."
```

---
### Task 5: Shared lifecycle node base

Exists so the six nodes do not each carry a copy of identical parameter and lifecycle plumbing.

**Files:**
- Create: `include/ros2_yolos_trt/nodes/yolos_lifecycle_node_base.hpp`, `src/nodes/yolos_lifecycle_node_base.cpp`
- Test: `test/unit/test_lifecycle_node_base.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `YolosConfig` (Task 3).
- Produces: class `ros2_yolos_trt::YolosLifecycleNodeBase` deriving from `rclcpp_lifecycle::LifecycleNode`, with:
  - `explicit YolosLifecycleNodeBase(const std::string& node_name, const rclcpp::NodeOptions& options)`
  - `protected: void declareCommonParameters()`
  - `protected: YolosConfig loadCommonConfig() const`
  - `protected: bool publishDebugImage() const`, `bool publishTiming() const`
  - `protected: float depthMin() const`, `float depthMax() const`
  - `protected: void publishTiming(double inference_ms, double total_ms)`
  - `protected: rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Float64MultiArray>::SharedPtr timing_pub_`
  - `protected: rclcpp::CallbackGroup::SharedPtr inference_cb_group_`

- [ ] **Step 1: Write the failing test**

```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include <gtest/gtest.h>

#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "ros2_yolos_trt/nodes/yolos_lifecycle_node_base.hpp"

using ros2_yolos_trt::YolosLifecycleNodeBase;

// Concrete probe: the base is abstract only in intent, so expose the protected
// helpers for testing.
class ProbeNode : public YolosLifecycleNodeBase {
public:
  explicit ProbeNode(const rclcpp::NodeOptions& options)
  : YolosLifecycleNodeBase("probe_node", options) {
    declareCommonParameters();
  }
  using YolosLifecycleNodeBase::loadCommonConfig;
  using YolosLifecycleNodeBase::publishDebugImage;
};

class LifecycleBaseTest : public ::testing::Test {
protected:
  void SetUp() override { rclcpp::init(0, nullptr); }
  void TearDown() override { rclcpp::shutdown(); }
};

TEST_F(LifecycleBaseTest, DeclaresTensorRtParameterDefaults) {
  auto node = std::make_shared<ProbeNode>(rclcpp::NodeOptions{});
  const auto cfg = node->loadCommonConfig();
  EXPECT_EQ(cfg.precision, "fp16");
  EXPECT_EQ(cfg.dla_core, -1);
  EXPECT_FLOAT_EQ(cfg.conf_threshold, 0.4f);
  EXPECT_FALSE(node->publishDebugImage());
}

TEST_F(LifecycleBaseTest, ParameterOverridesReachTheConfig) {
  rclcpp::NodeOptions options;
  options.parameter_overrides({
    rclcpp::Parameter("model_path", "/models/x.onnx"),
    rclcpp::Parameter("precision", "fp32"),
    rclcpp::Parameter("conf_threshold", 0.7),
    rclcpp::Parameter("publish_debug_image", true),
  });
  auto node = std::make_shared<ProbeNode>(options);
  const auto cfg = node->loadCommonConfig();
  EXPECT_EQ(cfg.model_path, "/models/x.onnx");
  EXPECT_EQ(cfg.precision, "fp32");
  EXPECT_FLOAT_EQ(cfg.conf_threshold, 0.7f);
  EXPECT_TRUE(node->publishDebugImage());
}

// A node that starts in the unconfigured state must not have published yet.
TEST_F(LifecycleBaseTest, StartsUnconfigured) {
  auto node = std::make_shared<ProbeNode>(rclcpp::NodeOptions{});
  EXPECT_EQ(node->get_current_state().id(),
            lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `colcon build --packages-select ros2_yolos_trt`
Expected: FAIL — `yolos_lifecycle_node_base.hpp` does not exist.

- [ ] **Step 3: Write the header**

```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ROS2_YOLOS_TRT__NODES__YOLOS_LIFECYCLE_NODE_BASE_HPP_
#define ROS2_YOLOS_TRT__NODES__YOLOS_LIFECYCLE_NODE_BASE_HPP_

#include <string>

#include <lifecycle_msgs/msg/state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "ros2_yolos_trt/adapters/yolos_adapter_base.hpp"
#include "ros2_yolos_trt/visibility_control.hpp"

namespace ros2_yolos_trt {

/// @brief Parameter, lifecycle and timing plumbing shared by every task node.
///
/// Task nodes derive from this and contribute only their publishers and their
/// image callback.
class ROS2_YOLOS_TRT_PUBLIC YolosLifecycleNodeBase
  : public rclcpp_lifecycle::LifecycleNode {
public:
  YolosLifecycleNodeBase(const std::string& node_name,
                         const rclcpp::NodeOptions& options);
  ~YolosLifecycleNodeBase() override = default;

  using CallbackReturn =
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

protected:
  /// @brief Declare every parameter in config/default_params.yaml.
  void declareCommonParameters();

  /// @brief Read the declared parameters into a YolosConfig.
  [[nodiscard]] YolosConfig loadCommonConfig() const;

  [[nodiscard]] bool publishDebugImage() const;
  [[nodiscard]] bool publishTimingEnabled() const;
  [[nodiscard]] float depthMin() const;
  [[nodiscard]] float depthMax() const;
  [[nodiscard]] int maxDetections() const;

  /// @brief Create the timing publisher. Call from on_configure().
  void createTimingPublisher();

  /// @brief Publish [inference_ms, total_ms] when timing is enabled.
  void publishTimingSample(double inference_ms, double total_ms);

  rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Float64MultiArray>::SharedPtr
    timing_pub_;

  /// @brief Dedicated group so inference never blocks lifecycle transitions.
  rclcpp::CallbackGroup::SharedPtr inference_cb_group_;
};

}  // namespace ros2_yolos_trt

#endif  // ROS2_YOLOS_TRT__NODES__YOLOS_LIFECYCLE_NODE_BASE_HPP_
```

- [ ] **Step 4: Write the implementation**

```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_trt/nodes/yolos_lifecycle_node_base.hpp"

#include <vector>

namespace ros2_yolos_trt {

YolosLifecycleNodeBase::YolosLifecycleNodeBase(const std::string& node_name,
                                               const rclcpp::NodeOptions& options)
: rclcpp_lifecycle::LifecycleNode(node_name, options) {
  inference_cb_group_ =
    create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
}

void YolosLifecycleNodeBase::declareCommonParameters() {
  declare_parameter<std::string>("model_path", "");
  declare_parameter<std::string>("labels_path", "");
  declare_parameter<std::string>("precision", "fp16");
  declare_parameter<std::string>("engine_cache_dir", "");
  declare_parameter<int>("dla_core", -1);
  declare_parameter<double>("conf_threshold", 0.4);
  declare_parameter<double>("nms_threshold", 0.45);
  declare_parameter<int>("max_detections", 300);
  declare_parameter<double>("depth_min", 0.0);
  declare_parameter<double>("depth_max", 0.0);
  declare_parameter<bool>("publish_debug_image", false);
  declare_parameter<bool>("publish_timing", false);
}

YolosConfig YolosLifecycleNodeBase::loadCommonConfig() const {
  YolosConfig cfg;
  cfg.model_path = get_parameter("model_path").as_string();
  cfg.labels_path = get_parameter("labels_path").as_string();
  cfg.precision = get_parameter("precision").as_string();
  cfg.engine_cache_dir = get_parameter("engine_cache_dir").as_string();
  cfg.dla_core = static_cast<int>(get_parameter("dla_core").as_int());
  cfg.conf_threshold =
    static_cast<float>(get_parameter("conf_threshold").as_double());
  cfg.nms_threshold =
    static_cast<float>(get_parameter("nms_threshold").as_double());
  cfg.max_detections =
    static_cast<int>(get_parameter("max_detections").as_int());
  return cfg;
}

bool YolosLifecycleNodeBase::publishDebugImage() const {
  return get_parameter("publish_debug_image").as_bool();
}

bool YolosLifecycleNodeBase::publishTimingEnabled() const {
  return get_parameter("publish_timing").as_bool();
}

float YolosLifecycleNodeBase::depthMin() const {
  return static_cast<float>(get_parameter("depth_min").as_double());
}

float YolosLifecycleNodeBase::depthMax() const {
  return static_cast<float>(get_parameter("depth_max").as_double());
}

int YolosLifecycleNodeBase::maxDetections() const {
  return static_cast<int>(get_parameter("max_detections").as_int());
}

void YolosLifecycleNodeBase::createTimingPublisher() {
  timing_pub_ =
    create_publisher<std_msgs::msg::Float64MultiArray>("~/timing", 10);
}

void YolosLifecycleNodeBase::publishTimingSample(double inference_ms,
                                                 double total_ms) {
  if (!publishTimingEnabled() || !timing_pub_ || !timing_pub_->is_activated()) {
    return;
  }
  std_msgs::msg::Float64MultiArray msg;
  msg.data = {inference_ms, total_ms};
  timing_pub_->publish(msg);
}

}  // namespace ros2_yolos_trt
```

- [ ] **Step 5: Register source and test in `CMakeLists.txt`**

Add to the component library sources:
```cmake
  src/nodes/yolos_lifecycle_node_base.cpp
```
And inside `if(BUILD_TESTING)`:
```cmake
  ament_add_gtest(test_lifecycle_node_base test/unit/test_lifecycle_node_base.cpp)
  target_link_libraries(test_lifecycle_node_base ${PROJECT_NAME}_components)
  ament_target_dependencies(test_lifecycle_node_base rclcpp rclcpp_lifecycle lifecycle_msgs)
```

- [ ] **Step 6: Run tests to verify they pass**

Run:
```bash
colcon build --packages-select ros2_yolos_trt
colcon test --packages-select ros2_yolos_trt --ctest-args -R test_lifecycle_node_base
colcon test-result --verbose
```
Expected: PASS, 3 cases.

- [ ] **Step 7: Commit**

```bash
git add include/ros2_yolos_trt/nodes/yolos_lifecycle_node_base.hpp src/nodes/yolos_lifecycle_node_base.cpp test/unit/test_lifecycle_node_base.cpp CMakeLists.txt
git commit -m "feat: add shared lifecycle node base

Six task nodes would otherwise carry six copies of the same parameter
declaration, lifecycle and timing code. Inference runs in a dedicated
mutually-exclusive callback group so a long inference call cannot block
lifecycle transitions."
```

---

### Task 6: Detection slice — adapter, converter, node, launch

The first vertical slice; it proves the whole stack before the remaining five tasks repeat the shape.

**Files:**
- Create: `include/ros2_yolos_trt/adapters/detector_adapter.hpp`, `src/adapters/detector_adapter.cpp`
- Create: `include/ros2_yolos_trt/conversion/detection_converter.hpp`, `src/conversion/detection_converter.cpp`
- Create: `include/ros2_yolos_trt/nodes/detector_node.hpp`, `src/nodes/detector_node.cpp`
- Create: `launch/detector.launch.py`
- Test: `test/unit/test_detection_converter.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `IDetectorAdapter`, `DetectionResult`, `YolosConfig` (Task 3); `resolveEngine` (Task 4); `YolosLifecycleNodeBase` (Task 5).
- Produces:
  - `class DetectorAdapter : public IDetectorAdapter` and `createDetectorAdapter()`
  - `vision_msgs::msg::Detection2DArray ros2_yolos_trt::conversion::toDetection2DArray(const std::vector<DetectionResult>&, const std_msgs::msg::Header&)`
  - `class YolosDetectorNode : public YolosLifecycleNodeBase`, registered as component `ros2_yolos_trt::YolosDetectorNode`, executable `yolos_trt_detector_node`

- [ ] **Step 1: Write the failing converter test**

The converter is the part testable without a GPU, so it gets the unit test.

```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include <gtest/gtest.h>

#include "ros2_yolos_trt/conversion/detection_converter.hpp"

using ros2_yolos_trt::DetectionResult;
using ros2_yolos_trt::conversion::toDetection2DArray;

namespace {
DetectionResult makeDetection(int x, int y, int w, int h, float conf, int id,
                              const std::string& name) {
  DetectionResult d;
  d.bbox.x = x;
  d.bbox.y = y;
  d.bbox.width = w;
  d.bbox.height = h;
  d.confidence = conf;
  d.class_id = id;
  d.class_name = name;
  return d;
}
}  // namespace

TEST(DetectionConverter, EmptyInputProducesEmptyArrayWithHeader) {
  std_msgs::msg::Header header;
  header.frame_id = "camera_link";
  const auto msg = toDetection2DArray({}, header);
  EXPECT_TRUE(msg.detections.empty());
  EXPECT_EQ(msg.header.frame_id, "camera_link");
}

// vision_msgs uses centre + size; the DTO uses top-left + extent. Getting this
// conversion wrong shifts every box by half its size, which is the single most
// likely bug in this file.
TEST(DetectionConverter, ConvertsTopLeftExtentToCentreSize) {
  std_msgs::msg::Header header;
  const auto msg =
    toDetection2DArray({makeDetection(10, 20, 30, 40, 0.9f, 2, "car")}, header);

  ASSERT_EQ(msg.detections.size(), 1u);
  const auto& d = msg.detections[0];
  EXPECT_DOUBLE_EQ(d.bbox.center.position.x, 25.0);  // 10 + 30/2
  EXPECT_DOUBLE_EQ(d.bbox.center.position.y, 40.0);  // 20 + 40/2
  EXPECT_DOUBLE_EQ(d.bbox.size_x, 30.0);
  EXPECT_DOUBLE_EQ(d.bbox.size_y, 40.0);
}

TEST(DetectionConverter, CarriesClassIdNameAndScore) {
  std_msgs::msg::Header header;
  const auto msg =
    toDetection2DArray({makeDetection(0, 0, 4, 4, 0.75f, 7, "person")}, header);
  ASSERT_EQ(msg.detections.size(), 1u);
  ASSERT_EQ(msg.detections[0].results.size(), 1u);
  EXPECT_EQ(msg.detections[0].results[0].hypothesis.class_id, "person");
  EXPECT_NEAR(msg.detections[0].results[0].hypothesis.score, 0.75, 1e-6);
}

TEST(DetectionConverter, FallsBackToNumericClassIdWhenNameMissing) {
  std_msgs::msg::Header header;
  const auto msg =
    toDetection2DArray({makeDetection(0, 0, 4, 4, 0.5f, 42, "")}, header);
  ASSERT_EQ(msg.detections.size(), 1u);
  EXPECT_EQ(msg.detections[0].results[0].hypothesis.class_id, "42");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `colcon build --packages-select ros2_yolos_trt`
Expected: FAIL — `detection_converter.hpp` does not exist.

- [ ] **Step 3: Write the converter**

`include/ros2_yolos_trt/conversion/detection_converter.hpp`
```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ROS2_YOLOS_TRT__CONVERSION__DETECTION_CONVERTER_HPP_
#define ROS2_YOLOS_TRT__CONVERSION__DETECTION_CONVERTER_HPP_

#include <vector>

#include <std_msgs/msg/header.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "ros2_yolos_trt/adapters/yolos_adapter_base.hpp"
#include "ros2_yolos_trt/visibility_control.hpp"

namespace ros2_yolos_trt {
namespace conversion {

/// @brief Convert detection DTOs to a vision_msgs Detection2DArray.
ROS2_YOLOS_TRT_PUBLIC vision_msgs::msg::Detection2DArray toDetection2DArray(
  const std::vector<DetectionResult>& detections,
  const std_msgs::msg::Header& header);

}  // namespace conversion
}  // namespace ros2_yolos_trt

#endif  // ROS2_YOLOS_TRT__CONVERSION__DETECTION_CONVERTER_HPP_
```

`src/conversion/detection_converter.cpp`
```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_trt/conversion/detection_converter.hpp"

#include <string>

namespace ros2_yolos_trt {
namespace conversion {

vision_msgs::msg::Detection2DArray toDetection2DArray(
  const std::vector<DetectionResult>& detections,
  const std_msgs::msg::Header& header) {
  vision_msgs::msg::Detection2DArray msg;
  msg.header = header;
  msg.detections.reserve(detections.size());

  for (const auto& det : detections) {
    vision_msgs::msg::Detection2D d;
    d.header = header;

    // The DTO stores top-left plus extent; vision_msgs wants the centre.
    d.bbox.center.position.x = det.bbox.x + det.bbox.width / 2.0;
    d.bbox.center.position.y = det.bbox.y + det.bbox.height / 2.0;
    d.bbox.center.theta = 0.0;
    d.bbox.size_x = static_cast<double>(det.bbox.width);
    d.bbox.size_y = static_cast<double>(det.bbox.height);

    vision_msgs::msg::ObjectHypothesisWithPose hyp;
    // class_id is a string in vision_msgs; prefer the label, fall back to the
    // numeric id so a missing labels file still yields identifiable output.
    hyp.hypothesis.class_id = det.class_name.empty()
                                ? std::to_string(det.class_id)
                                : det.class_name;
    hyp.hypothesis.score = static_cast<double>(det.confidence);
    d.results.push_back(hyp);

    msg.detections.push_back(std::move(d));
  }
  return msg;
}

}  // namespace conversion
}  // namespace ros2_yolos_trt
```

- [ ] **Step 4: Write the adapter**

`include/ros2_yolos_trt/adapters/detector_adapter.hpp`
```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ROS2_YOLOS_TRT__ADAPTERS__DETECTOR_ADAPTER_HPP_
#define ROS2_YOLOS_TRT__ADAPTERS__DETECTOR_ADAPTER_HPP_

#include <memory>
#include <string>
#include <vector>

#include "ros2_yolos_trt/adapters/yolos_adapter_base.hpp"

namespace ros2_yolos_trt {

/// @brief Detection adapter wrapping yolos::det::YOLODetector.
class ROS2_YOLOS_TRT_PUBLIC DetectorAdapter : public IDetectorAdapter {
public:
  DetectorAdapter();
  ~DetectorAdapter() override;

  DetectorAdapter(const DetectorAdapter&) = delete;
  DetectorAdapter& operator=(const DetectorAdapter&) = delete;
  DetectorAdapter(DetectorAdapter&&) noexcept;
  DetectorAdapter& operator=(DetectorAdapter&&) noexcept;

  bool initialize(const YolosConfig& config) override;
  [[nodiscard]] bool isInitialized() const noexcept override;
  void shutdown() override;
  [[nodiscard]] const std::vector<std::string>& getClassNames() const override;

  std::vector<DetectionResult> detect(
    const cv::Mat& image, float conf_threshold, float nms_threshold) override;
  void drawDetections(
    cv::Mat& image, const std::vector<DetectionResult>& detections) override;

private:
  class Impl;  // TensorRT and yolos:: types stay entirely in the .cpp
  std::unique_ptr<Impl> impl_;
};

}  // namespace ros2_yolos_trt

#endif  // ROS2_YOLOS_TRT__ADAPTERS__DETECTOR_ADAPTER_HPP_
```

`src/adapters/detector_adapter.cpp`
```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_trt/adapters/detector_adapter.hpp"

// The only place upstream headers appear for this task.
#include "yolos/tasks/detection.hpp"

#include <iostream>

#include "ros2_yolos_trt/engine/engine_cache.hpp"

namespace ros2_yolos_trt {

class DetectorAdapter::Impl {
public:
  std::unique_ptr<yolos::det::YOLODetector> detector;
  std::vector<std::string> class_names;
  bool initialized{false};
};

DetectorAdapter::DetectorAdapter() : impl_(std::make_unique<Impl>()) {}
DetectorAdapter::~DetectorAdapter() = default;
DetectorAdapter::DetectorAdapter(DetectorAdapter&&) noexcept = default;
DetectorAdapter& DetectorAdapter::operator=(DetectorAdapter&&) noexcept = default;

bool DetectorAdapter::initialize(const YolosConfig& config) {
  try {
    std::string error;
    const auto engine = engine::resolveEngine(
      config.model_path, config.precision, config.dla_core,
      config.engine_cache_dir, error);
    if (engine.empty()) {
      std::cerr << "[DetectorAdapter] " << error << std::endl;
      return false;
    }

    // Version is left to runtime detection from output tensor shapes.
    impl_->detector = yolos::det::createDetector(
      engine.string(), config.labels_path, yolos::YOLOVersion::Auto,
      config.dla_core);
    if (!impl_->detector) {
      std::cerr << "[DetectorAdapter] failed to create detector" << std::endl;
      return false;
    }

    impl_->class_names = impl_->detector->getClassNames();
    impl_->initialized = true;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[DetectorAdapter] initialization error: " << e.what()
              << std::endl;
    impl_->initialized = false;
    return false;
  }
}

bool DetectorAdapter::isInitialized() const noexcept {
  return impl_ && impl_->initialized;
}

void DetectorAdapter::shutdown() {
  if (impl_) {
    impl_->detector.reset();
    impl_->initialized = false;
  }
}

const std::vector<std::string>& DetectorAdapter::getClassNames() const {
  static const std::vector<std::string> empty;
  return impl_ ? impl_->class_names : empty;
}

std::vector<DetectionResult> DetectorAdapter::detect(
  const cv::Mat& image, float conf_threshold, float nms_threshold) {
  std::vector<DetectionResult> results;
  if (!isInitialized() || image.empty()) return results;

  try {
    const auto detections =
      impl_->detector->detect(image, conf_threshold, nms_threshold);
    results.reserve(detections.size());
    for (const auto& det : detections) {
      DetectionResult r;
      r.bbox.x = det.box.x;
      r.bbox.y = det.box.y;
      r.bbox.width = det.box.width;
      r.bbox.height = det.box.height;
      r.confidence = det.conf;
      r.class_id = det.classId;
      if (det.classId >= 0 &&
          static_cast<size_t>(det.classId) < impl_->class_names.size()) {
        r.class_name = impl_->class_names[det.classId];
      }
      results.push_back(std::move(r));
    }
  } catch (const std::exception& e) {
    std::cerr << "[DetectorAdapter] detection error: " << e.what() << std::endl;
  }
  return results;
}

void DetectorAdapter::drawDetections(
  cv::Mat& image, const std::vector<DetectionResult>& detections) {
  if (!isInitialized() || image.empty()) return;

  std::vector<yolos::det::Detection> native;
  native.reserve(detections.size());
  for (const auto& det : detections) {
    yolos::BoundingBox box(det.bbox.x, det.bbox.y, det.bbox.width,
                           det.bbox.height);
    native.emplace_back(box, det.confidence, det.class_id);
  }
  impl_->detector->drawDetections(image, native);
}

std::unique_ptr<IDetectorAdapter> createDetectorAdapter() {
  return std::make_unique<DetectorAdapter>();
}

}  // namespace ros2_yolos_trt
```

- [ ] **Step 5: Write the node**

`include/ros2_yolos_trt/nodes/detector_node.hpp`
```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ROS2_YOLOS_TRT__NODES__DETECTOR_NODE_HPP_
#define ROS2_YOLOS_TRT__NODES__DETECTOR_NODE_HPP_

#include <memory>

#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "ros2_yolos_trt/adapters/detector_adapter.hpp"
#include "ros2_yolos_trt/nodes/yolos_lifecycle_node_base.hpp"

namespace ros2_yolos_trt {

class ROS2_YOLOS_TRT_PUBLIC YolosDetectorNode : public YolosLifecycleNodeBase {
public:
  explicit YolosDetectorNode(const rclcpp::NodeOptions& options);

  CallbackReturn on_configure(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_shutdown(const rclcpp_lifecycle::State& state) override;

private:
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);

  std::unique_ptr<IDetectorAdapter> adapter_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp_lifecycle::LifecyclePublisher<vision_msgs::msg::Detection2DArray>::SharedPtr
    det_pub_;
  rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::Image>::SharedPtr
    debug_pub_;
  YolosConfig config_;
};

}  // namespace ros2_yolos_trt

#endif  // ROS2_YOLOS_TRT__NODES__DETECTOR_NODE_HPP_
```

`src/nodes/detector_node.cpp`
```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_trt/nodes/detector_node.hpp"

#include <chrono>
#include <memory>

#include <cv_bridge/cv_bridge.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include "ros2_yolos_trt/conversion/detection_converter.hpp"

namespace ros2_yolos_trt {

YolosDetectorNode::YolosDetectorNode(const rclcpp::NodeOptions& options)
: YolosLifecycleNodeBase("yolos_trt_detector", options) {
  declareCommonParameters();
}

YolosDetectorNode::CallbackReturn YolosDetectorNode::on_configure(
  const rclcpp_lifecycle::State&) {
  config_ = loadCommonConfig();
  if (config_.model_path.empty()) {
    RCLCPP_ERROR(get_logger(), "model_path parameter is required");
    return CallbackReturn::FAILURE;
  }

  adapter_ = createDetectorAdapter();

  // Building an engine from .onnx happens here and can take minutes; doing it
  // in on_configure is what lets the lifecycle state report the truth.
  RCLCPP_INFO(get_logger(), "loading model %s", config_.model_path.c_str());
  if (!adapter_->initialize(config_)) {
    RCLCPP_ERROR(get_logger(), "failed to initialize detector");
    adapter_.reset();
    return CallbackReturn::FAILURE;
  }

  det_pub_ = create_publisher<vision_msgs::msg::Detection2DArray>(
    "~/detections", 10);
  if (publishDebugImage()) {
    debug_pub_ = create_publisher<sensor_msgs::msg::Image>("~/debug_image", 10);
  }
  createTimingPublisher();

  rclcpp::SubscriptionOptions sub_options;
  sub_options.callback_group = inference_cb_group_;
  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
    "image", rclcpp::SensorDataQoS(),
    [this](const sensor_msgs::msg::Image::ConstSharedPtr msg) {
      imageCallback(msg);
    },
    sub_options);

  RCLCPP_INFO(get_logger(), "configured with %zu classes",
              adapter_->getClassNames().size());
  return CallbackReturn::SUCCESS;
}

YolosDetectorNode::CallbackReturn YolosDetectorNode::on_activate(
  const rclcpp_lifecycle::State& state) {
  LifecycleNode::on_activate(state);
  return CallbackReturn::SUCCESS;
}

YolosDetectorNode::CallbackReturn YolosDetectorNode::on_deactivate(
  const rclcpp_lifecycle::State& state) {
  LifecycleNode::on_deactivate(state);
  return CallbackReturn::SUCCESS;
}

YolosDetectorNode::CallbackReturn YolosDetectorNode::on_cleanup(
  const rclcpp_lifecycle::State&) {
  image_sub_.reset();
  det_pub_.reset();
  debug_pub_.reset();
  timing_pub_.reset();
  if (adapter_) adapter_->shutdown();
  adapter_.reset();
  return CallbackReturn::SUCCESS;
}

YolosDetectorNode::CallbackReturn YolosDetectorNode::on_shutdown(
  const rclcpp_lifecycle::State& state) {
  return on_cleanup(state);
}

void YolosDetectorNode::imageCallback(
  const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
  if (!adapter_ || !adapter_->isInitialized()) return;
  if (!det_pub_ || !det_pub_->is_activated()) return;

  const auto t_start = std::chrono::steady_clock::now();

  cv::Mat frame;
  try {
    frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
  } catch (const cv_bridge::Exception& e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge: %s", e.what());
    return;
  }

  const auto t_infer_start = std::chrono::steady_clock::now();
  auto detections =
    adapter_->detect(frame, config_.conf_threshold, config_.nms_threshold);
  const auto t_infer_end = std::chrono::steady_clock::now();

  det_pub_->publish(conversion::toDetection2DArray(detections, msg->header));

  if (debug_pub_ && debug_pub_->is_activated()) {
    adapter_->drawDetections(frame, detections);
    debug_pub_->publish(
      *cv_bridge::CvImage(msg->header, "bgr8", frame).toImageMsg());
  }

  const auto t_end = std::chrono::steady_clock::now();
  using ms = std::chrono::duration<double, std::milli>;
  publishTimingSample(ms(t_infer_end - t_infer_start).count(),
                      ms(t_end - t_start).count());
}

}  // namespace ros2_yolos_trt

RCLCPP_COMPONENTS_REGISTER_NODE(ros2_yolos_trt::YolosDetectorNode)
```

Note: on ROS 2 Humble the cv_bridge header is `<cv_bridge/cv_bridge.h>`; on Jazzy and newer it is `<cv_bridge/cv_bridge.hpp>`. If the build fails on the include, switch to the other spelling.

- [ ] **Step 6: Write `launch/detector.launch.py`**

```python
# Copyright 2026 YOLOs-TRT Team
# SPDX-License-Identifier: AGPL-3.0
"""Launch the TensorRT YOLO detector as a managed lifecycle node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            'model_path', description='.onnx (engine built on first run) or .trt'),
        DeclareLaunchArgument('labels_path', default_value=''),
        DeclareLaunchArgument('precision', default_value='fp16'),
        DeclareLaunchArgument('engine_cache_dir', default_value=''),
        DeclareLaunchArgument('image_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('conf_threshold', default_value='0.4'),
        DeclareLaunchArgument('nms_threshold', default_value='0.45'),
        DeclareLaunchArgument('publish_debug_image', default_value='false'),
    ]

    node = LifecycleNode(
        package='ros2_yolos_trt',
        executable='yolos_trt_detector_node',
        name='yolos_trt_detector',
        namespace='',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'labels_path': LaunchConfiguration('labels_path'),
            'precision': LaunchConfiguration('precision'),
            'engine_cache_dir': LaunchConfiguration('engine_cache_dir'),
            'conf_threshold': LaunchConfiguration('conf_threshold'),
            'nms_threshold': LaunchConfiguration('nms_threshold'),
            'publish_debug_image': LaunchConfiguration('publish_debug_image'),
        }],
        remappings=[('image', LaunchConfiguration('image_topic'))],
    )

    return LaunchDescription(args + [node])
```

- [ ] **Step 7: Register everything in `CMakeLists.txt`**

Add to the component library sources:
```cmake
  src/adapters/detector_adapter.cpp
  src/conversion/detection_converter.cpp
  src/nodes/detector_node.cpp
```
After the library definition, register the component:
```cmake
rclcpp_components_register_node(${PROJECT_NAME}_components
  PLUGIN "ros2_yolos_trt::YolosDetectorNode"
  EXECUTABLE yolos_trt_detector_node)
```
And inside `if(BUILD_TESTING)`:
```cmake
  ament_add_gtest(test_detection_converter test/unit/test_detection_converter.cpp)
  target_link_libraries(test_detection_converter ${PROJECT_NAME}_components)
  ament_target_dependencies(test_detection_converter vision_msgs std_msgs)
```
Also remove the `OPTIONAL` keyword from the `install(DIRECTORY launch/ ...)` line now that `launch/` exists.

- [ ] **Step 8: Run tests to verify they pass**

Run:
```bash
colcon build --packages-select ros2_yolos_trt
colcon test --packages-select ros2_yolos_trt --ctest-args -R test_detection_converter
colcon test-result --verbose
ros2 pkg executables ros2_yolos_trt | grep yolos_trt_detector_node
```
Expected: 4 converter cases pass; the executable is listed.

- [ ] **Step 9: Commit**

```bash
git add include/ros2_yolos_trt/adapters/detector_adapter.hpp src/adapters/detector_adapter.cpp include/ros2_yolos_trt/conversion/detection_converter.hpp src/conversion/detection_converter.cpp include/ros2_yolos_trt/nodes/detector_node.hpp src/nodes/detector_node.cpp launch/detector.launch.py test/unit/test_detection_converter.cpp CMakeLists.txt
git commit -m "feat: add detection node, adapter and converter

First vertical slice through all four layers. The adapter resolves
model_path through the engine cache before constructing the detector, so a
node can be pointed at an .onnx. Engine building happens in on_configure,
which can take minutes, so the lifecycle state reflects it rather than the
node coming up broken.

The converter translates the DTO's top-left-plus-extent box into
vision_msgs' centre-plus-size, and falls back to the numeric class id when
no labels file was supplied."
```

---
### Task 7: Depth slice — adapter, converter, node, launch

The capability this package adds over `ros2_yolos_cpp`. The correctness risk here is a wrong depth *scale*, which looks plausible and passes casual inspection, so the tests check units explicitly.

**Files:**
- Create: `include/ros2_yolos_trt/adapters/depth_adapter.hpp`, `src/adapters/depth_adapter.cpp`
- Create: `include/ros2_yolos_trt/conversion/depth_converter.hpp`, `src/conversion/depth_converter.cpp`
- Create: `include/ros2_yolos_trt/nodes/depth_node.hpp`, `src/nodes/depth_node.cpp`
- Create: `launch/depth.launch.py`
- Test: `test/unit/test_depth_converter.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `IDepthAdapter`, `DepthResult` (Task 3); `resolveEngine` (Task 4); `YolosLifecycleNodeBase` (Task 5).
- Produces:
  - `class DepthAdapter : public IDepthAdapter` and `createDepthAdapter()`
  - `sensor_msgs::msg::Image ros2_yolos_trt::conversion::toDepthImage(const DepthResult&, const std_msgs::msg::Header&)`
  - `class YolosDepthNode : public YolosLifecycleNodeBase`, component `ros2_yolos_trt::YolosDepthNode`, executable `yolos_trt_depth_node`

- [ ] **Step 1: Write the failing converter test**

```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include <gtest/gtest.h>

#include "ros2_yolos_trt/conversion/depth_converter.hpp"

using ros2_yolos_trt::DepthResult;
using ros2_yolos_trt::conversion::toDepthImage;

// 32FC1 in metres is the ROS convention; publishing anything else silently
// breaks depth_image_proc, rtabmap and RViz.
TEST(DepthConverter, PublishesThirtyTwoFloatSingleChannel) {
  DepthResult r;
  r.depth_metres = cv::Mat::zeros(6, 8, CV_32FC1);

  std_msgs::msg::Header header;
  header.frame_id = "camera_depth_optical_frame";
  const auto msg = toDepthImage(r, header);

  EXPECT_EQ(msg.encoding, "32FC1");
  EXPECT_EQ(msg.width, 8u);
  EXPECT_EQ(msg.height, 6u);
  EXPECT_EQ(msg.step, 8u * sizeof(float));
  EXPECT_EQ(msg.header.frame_id, "camera_depth_optical_frame");
}

// The model already emits metric depth; the converter must not rescale it.
TEST(DepthConverter, PreservesMetreValuesExactly) {
  DepthResult r;
  r.depth_metres = cv::Mat::zeros(1, 3, CV_32FC1);
  r.depth_metres.at<float>(0, 0) = 0.75f;
  r.depth_metres.at<float>(0, 1) = 4.5f;
  r.depth_metres.at<float>(0, 2) = 12.25f;

  const auto msg = toDepthImage(r, std_msgs::msg::Header{});

  ASSERT_EQ(msg.data.size(), 3u * sizeof(float));
  const auto* values = reinterpret_cast<const float*>(msg.data.data());
  EXPECT_FLOAT_EQ(values[0], 0.75f);
  EXPECT_FLOAT_EQ(values[1], 4.5f);
  EXPECT_FLOAT_EQ(values[2], 12.25f);
}

TEST(DepthConverter, EmptyDepthProducesEmptyImage) {
  const auto msg = toDepthImage(DepthResult{}, std_msgs::msg::Header{});
  EXPECT_EQ(msg.width, 0u);
  EXPECT_EQ(msg.height, 0u);
  EXPECT_TRUE(msg.data.empty());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `colcon build --packages-select ros2_yolos_trt`
Expected: FAIL — `depth_converter.hpp` does not exist.

- [ ] **Step 3: Write the converter**

`include/ros2_yolos_trt/conversion/depth_converter.hpp`
```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ROS2_YOLOS_TRT__CONVERSION__DEPTH_CONVERTER_HPP_
#define ROS2_YOLOS_TRT__CONVERSION__DEPTH_CONVERTER_HPP_

#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

#include "ros2_yolos_trt/adapters/yolos_adapter_base.hpp"
#include "ros2_yolos_trt/visibility_control.hpp"

namespace ros2_yolos_trt {
namespace conversion {

/// @brief Wrap a metric depth map as a 32FC1 sensor_msgs/Image.
///
/// Values are passed through unchanged: the exported model already produces
/// metres, and 32FC1-in-metres is what ROS depth tooling expects.
ROS2_YOLOS_TRT_PUBLIC sensor_msgs::msg::Image toDepthImage(
  const DepthResult& result, const std_msgs::msg::Header& header);

}  // namespace conversion
}  // namespace ros2_yolos_trt

#endif  // ROS2_YOLOS_TRT__CONVERSION__DEPTH_CONVERTER_HPP_
```

`src/conversion/depth_converter.cpp`
```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_trt/conversion/depth_converter.hpp"

#include <cstring>

namespace ros2_yolos_trt {
namespace conversion {

sensor_msgs::msg::Image toDepthImage(const DepthResult& result,
                                     const std_msgs::msg::Header& header) {
  sensor_msgs::msg::Image msg;
  msg.header = header;
  msg.encoding = "32FC1";
  msg.is_bigendian = 0;

  if (result.depth_metres.empty()) {
    msg.width = 0;
    msg.height = 0;
    msg.step = 0;
    return msg;
  }

  // The adapter guarantees CV_32FC1; assert the contract cheaply rather than
  // silently emitting mislabelled data.
  if (result.depth_metres.type() != CV_32FC1) {
    msg.width = 0;
    msg.height = 0;
    msg.step = 0;
    return msg;
  }

  const cv::Mat& depth = result.depth_metres;
  msg.width = static_cast<uint32_t>(depth.cols);
  msg.height = static_cast<uint32_t>(depth.rows);
  msg.step = static_cast<uint32_t>(depth.cols * sizeof(float));
  msg.data.resize(static_cast<size_t>(msg.step) * msg.height);

  // Copy row by row: a cv::Mat may be padded, msg.data must be dense.
  for (int row = 0; row < depth.rows; ++row) {
    std::memcpy(msg.data.data() + static_cast<size_t>(row) * msg.step,
                depth.ptr<float>(row), msg.step);
  }
  return msg;
}

}  // namespace conversion
}  // namespace ros2_yolos_trt
```

- [ ] **Step 4: Write the adapter**

`include/ros2_yolos_trt/adapters/depth_adapter.hpp`
```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ROS2_YOLOS_TRT__ADAPTERS__DEPTH_ADAPTER_HPP_
#define ROS2_YOLOS_TRT__ADAPTERS__DEPTH_ADAPTER_HPP_

#include <memory>
#include <string>
#include <vector>

#include "ros2_yolos_trt/adapters/yolos_adapter_base.hpp"

namespace ros2_yolos_trt {

/// @brief Depth adapter wrapping yolos::depth::YOLODepthEstimator.
class ROS2_YOLOS_TRT_PUBLIC DepthAdapter : public IDepthAdapter {
public:
  DepthAdapter();
  ~DepthAdapter() override;

  DepthAdapter(const DepthAdapter&) = delete;
  DepthAdapter& operator=(const DepthAdapter&) = delete;
  DepthAdapter(DepthAdapter&&) noexcept;
  DepthAdapter& operator=(DepthAdapter&&) noexcept;

  bool initialize(const YolosConfig& config) override;
  [[nodiscard]] bool isInitialized() const noexcept override;
  void shutdown() override;
  [[nodiscard]] const std::vector<std::string>& getClassNames() const override;

  DepthResult estimate(const cv::Mat& image) override;
  cv::Mat colorize(const DepthResult& result, float min_metres,
                   float max_metres) override;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ros2_yolos_trt

#endif  // ROS2_YOLOS_TRT__ADAPTERS__DEPTH_ADAPTER_HPP_
```

`src/adapters/depth_adapter.cpp`
```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_trt/adapters/depth_adapter.hpp"

#include "yolos/tasks/depth.hpp"

#include <iostream>

#include "ros2_yolos_trt/engine/engine_cache.hpp"

namespace ros2_yolos_trt {

class DepthAdapter::Impl {
public:
  std::unique_ptr<yolos::depth::YOLODepthEstimator> estimator;
  bool initialized{false};
};

DepthAdapter::DepthAdapter() : impl_(std::make_unique<Impl>()) {}
DepthAdapter::~DepthAdapter() = default;
DepthAdapter::DepthAdapter(DepthAdapter&&) noexcept = default;
DepthAdapter& DepthAdapter::operator=(DepthAdapter&&) noexcept = default;

bool DepthAdapter::initialize(const YolosConfig& config) {
  try {
    std::string error;
    const auto engine = engine::resolveEngine(
      config.model_path, config.precision, config.dla_core,
      config.engine_cache_dir, error);
    if (engine.empty()) {
      std::cerr << "[DepthAdapter] " << error << std::endl;
      return false;
    }

    // Depth models take no labels file: they predict a distance, not a class.
    impl_->estimator = std::make_unique<yolos::depth::YOLODepthEstimator>(
      engine.string(), config.dla_core);
    impl_->initialized = true;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[DepthAdapter] initialization error: " << e.what() << std::endl;
    impl_->initialized = false;
    return false;
  }
}

bool DepthAdapter::isInitialized() const noexcept {
  return impl_ && impl_->initialized;
}

void DepthAdapter::shutdown() {
  if (impl_) {
    impl_->estimator.reset();
    impl_->initialized = false;
  }
}

const std::vector<std::string>& DepthAdapter::getClassNames() const {
  // Depth has no classes; an empty list keeps the base interface honest.
  static const std::vector<std::string> empty;
  return empty;
}

DepthResult DepthAdapter::estimate(const cv::Mat& image) {
  DepthResult result;
  if (!isInitialized() || image.empty()) return result;

  try {
    cv::Mat depth = impl_->estimator->estimate(image);
    if (depth.empty()) return result;

    // The library returns CV_32FC1 metres at the original image size. Pass it
    // through without rescaling — the calibration is baked into the graph.
    result.depth_metres = std::move(depth);

    double min_val = 0.0;
    double max_val = 0.0;
    cv::minMaxLoc(result.depth_metres, &min_val, &max_val);
    result.min_metres = static_cast<float>(min_val);
    result.max_metres = static_cast<float>(max_val);
  } catch (const std::exception& e) {
    std::cerr << "[DepthAdapter] estimation error: " << e.what() << std::endl;
    return DepthResult{};
  }
  return result;
}

cv::Mat DepthAdapter::colorize(const DepthResult& result, float min_metres,
                               float max_metres) {
  if (result.depth_metres.empty()) return {};
  return yolos::depth::colorizeDepth(result.depth_metres, min_metres, max_metres);
}

std::unique_ptr<IDepthAdapter> createDepthAdapter() {
  return std::make_unique<DepthAdapter>();
}

}  // namespace ros2_yolos_trt
```

- [ ] **Step 5: Write the node**

`include/ros2_yolos_trt/nodes/depth_node.hpp`
```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ROS2_YOLOS_TRT__NODES__DEPTH_NODE_HPP_
#define ROS2_YOLOS_TRT__NODES__DEPTH_NODE_HPP_

#include <memory>

#include <sensor_msgs/msg/image.hpp>

#include "ros2_yolos_trt/adapters/depth_adapter.hpp"
#include "ros2_yolos_trt/nodes/yolos_lifecycle_node_base.hpp"

namespace ros2_yolos_trt {

class ROS2_YOLOS_TRT_PUBLIC YolosDepthNode : public YolosLifecycleNodeBase {
public:
  explicit YolosDepthNode(const rclcpp::NodeOptions& options);

  CallbackReturn on_configure(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_shutdown(const rclcpp_lifecycle::State& state) override;

private:
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);

  std::unique_ptr<IDepthAdapter> adapter_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::Image>::SharedPtr
    depth_pub_;
  rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::Image>::SharedPtr
    debug_pub_;
  YolosConfig config_;
};

}  // namespace ros2_yolos_trt

#endif  // ROS2_YOLOS_TRT__NODES__DEPTH_NODE_HPP_
```

`src/nodes/depth_node.cpp`
```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_trt/nodes/depth_node.hpp"

#include <chrono>
#include <memory>

#include <cv_bridge/cv_bridge.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include "ros2_yolos_trt/conversion/depth_converter.hpp"

namespace ros2_yolos_trt {

YolosDepthNode::YolosDepthNode(const rclcpp::NodeOptions& options)
: YolosLifecycleNodeBase("yolos_trt_depth", options) {
  declareCommonParameters();
}

YolosDepthNode::CallbackReturn YolosDepthNode::on_configure(
  const rclcpp_lifecycle::State&) {
  config_ = loadCommonConfig();
  if (config_.model_path.empty()) {
    RCLCPP_ERROR(get_logger(), "model_path parameter is required");
    return CallbackReturn::FAILURE;
  }

  adapter_ = createDepthAdapter();
  RCLCPP_INFO(get_logger(), "loading depth model %s", config_.model_path.c_str());
  if (!adapter_->initialize(config_)) {
    RCLCPP_ERROR(get_logger(), "failed to initialize depth estimator");
    adapter_.reset();
    return CallbackReturn::FAILURE;
  }

  // 32FC1 metres on ~/depth/image; the colorized view is debug only.
  depth_pub_ = create_publisher<sensor_msgs::msg::Image>("~/depth/image", 10);
  if (publishDebugImage()) {
    debug_pub_ =
      create_publisher<sensor_msgs::msg::Image>("~/depth/debug_image", 10);
  }
  createTimingPublisher();

  rclcpp::SubscriptionOptions sub_options;
  sub_options.callback_group = inference_cb_group_;
  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
    "image", rclcpp::SensorDataQoS(),
    [this](const sensor_msgs::msg::Image::ConstSharedPtr msg) {
      imageCallback(msg);
    },
    sub_options);

  RCLCPP_INFO(get_logger(), "configured; publishing 32FC1 metres");
  return CallbackReturn::SUCCESS;
}

YolosDepthNode::CallbackReturn YolosDepthNode::on_activate(
  const rclcpp_lifecycle::State& state) {
  LifecycleNode::on_activate(state);
  return CallbackReturn::SUCCESS;
}

YolosDepthNode::CallbackReturn YolosDepthNode::on_deactivate(
  const rclcpp_lifecycle::State& state) {
  LifecycleNode::on_deactivate(state);
  return CallbackReturn::SUCCESS;
}

YolosDepthNode::CallbackReturn YolosDepthNode::on_cleanup(
  const rclcpp_lifecycle::State&) {
  image_sub_.reset();
  depth_pub_.reset();
  debug_pub_.reset();
  timing_pub_.reset();
  if (adapter_) adapter_->shutdown();
  adapter_.reset();
  return CallbackReturn::SUCCESS;
}

YolosDepthNode::CallbackReturn YolosDepthNode::on_shutdown(
  const rclcpp_lifecycle::State& state) {
  return on_cleanup(state);
}

void YolosDepthNode::imageCallback(
  const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
  if (!adapter_ || !adapter_->isInitialized()) return;
  if (!depth_pub_ || !depth_pub_->is_activated()) return;

  const auto t_start = std::chrono::steady_clock::now();

  cv::Mat frame;
  try {
    frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
  } catch (const cv_bridge::Exception& e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge: %s", e.what());
    return;
  }

  const auto t_infer_start = std::chrono::steady_clock::now();
  const auto result = adapter_->estimate(frame);
  const auto t_infer_end = std::chrono::steady_clock::now();

  if (result.depth_metres.empty()) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                         "depth estimation returned an empty map");
    return;
  }

  depth_pub_->publish(conversion::toDepthImage(result, msg->header));

  if (debug_pub_ && debug_pub_->is_activated()) {
    const cv::Mat colored =
      adapter_->colorize(result, depthMin(), depthMax());
    if (!colored.empty()) {
      debug_pub_->publish(
        *cv_bridge::CvImage(msg->header, "bgr8", colored).toImageMsg());
    }
  }

  const auto t_end = std::chrono::steady_clock::now();
  using ms = std::chrono::duration<double, std::milli>;
  publishTimingSample(ms(t_infer_end - t_infer_start).count(),
                      ms(t_end - t_start).count());
}

}  // namespace ros2_yolos_trt

RCLCPP_COMPONENTS_REGISTER_NODE(ros2_yolos_trt::YolosDepthNode)
```

- [ ] **Step 6: Write `launch/depth.launch.py`**

```python
# Copyright 2026 YOLOs-TRT Team
# SPDX-License-Identifier: AGPL-3.0
"""Launch the TensorRT monocular metric depth node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            'model_path',
            description='yolo26*-depth .onnx (engine built on first run) or .trt'),
        DeclareLaunchArgument('precision', default_value='fp16'),
        DeclareLaunchArgument('engine_cache_dir', default_value=''),
        DeclareLaunchArgument('image_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument(
            'depth_min', default_value='0.0',
            description='debug colorization clamp, metres; 0 = auto-range'),
        DeclareLaunchArgument('depth_max', default_value='0.0'),
        DeclareLaunchArgument('publish_debug_image', default_value='true'),
    ]

    node = LifecycleNode(
        package='ros2_yolos_trt',
        executable='yolos_trt_depth_node',
        name='yolos_trt_depth',
        namespace='',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'precision': LaunchConfiguration('precision'),
            'engine_cache_dir': LaunchConfiguration('engine_cache_dir'),
            'depth_min': LaunchConfiguration('depth_min'),
            'depth_max': LaunchConfiguration('depth_max'),
            'publish_debug_image': LaunchConfiguration('publish_debug_image'),
        }],
        remappings=[('image', LaunchConfiguration('image_topic'))],
    )

    return LaunchDescription(args + [node])
```

- [ ] **Step 7: Register in `CMakeLists.txt`**

Component library sources:
```cmake
  src/adapters/depth_adapter.cpp
  src/conversion/depth_converter.cpp
  src/nodes/depth_node.cpp
```
Component registration:
```cmake
rclcpp_components_register_node(${PROJECT_NAME}_components
  PLUGIN "ros2_yolos_trt::YolosDepthNode"
  EXECUTABLE yolos_trt_depth_node)
```
Tests:
```cmake
  ament_add_gtest(test_depth_converter test/unit/test_depth_converter.cpp)
  target_link_libraries(test_depth_converter ${PROJECT_NAME}_components)
  ament_target_dependencies(test_depth_converter sensor_msgs std_msgs)
```

- [ ] **Step 8: Run tests to verify they pass**

Run:
```bash
colcon build --packages-select ros2_yolos_trt
colcon test --packages-select ros2_yolos_trt --ctest-args -R test_depth_converter
colcon test-result --verbose
ros2 pkg executables ros2_yolos_trt | grep yolos_trt_depth_node
```
Expected: 3 converter cases pass; the executable is listed.

- [ ] **Step 9: Commit**

```bash
git add include/ros2_yolos_trt/adapters/depth_adapter.hpp src/adapters/depth_adapter.cpp include/ros2_yolos_trt/conversion/depth_converter.hpp src/conversion/depth_converter.cpp include/ros2_yolos_trt/nodes/depth_node.hpp src/nodes/depth_node.cpp launch/depth.launch.py test/unit/test_depth_converter.cpp CMakeLists.txt
git commit -m "feat: add monocular metric depth node

Publishes 32FC1 metres on ~/depth/image, which is the ROS depth
convention, so depth_image_proc, rtabmap and RViz consume it unchanged.
Values pass through without rescaling because the exported YOLO26-depth
graph already applies its metric calibration.

The colorized view is confined to the debug topic: publishing colour on
the depth topic would be immediately wrong for every downstream consumer.
Depth takes no labels file, since it predicts a distance rather than a
class."
```

---
### Task 8: Segmentation slice

**Files:**
- Create: `include/ros2_yolos_trt/adapters/segmentor_adapter.hpp`, `src/adapters/segmentor_adapter.cpp`
- Create: `include/ros2_yolos_trt/conversion/segmentation_converter.hpp`, `src/conversion/segmentation_converter.cpp`
- Create: `include/ros2_yolos_trt/nodes/segmentor_node.hpp`, `src/nodes/segmentor_node.cpp`
- Create: `launch/segmentor.launch.py`
- Test: `test/unit/test_segmentation_converter.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `ISegmentorAdapter`, `SegmentationResult` (Task 3); `resolveEngine` (Task 4); `YolosLifecycleNodeBase` (Task 5); `toDetection2DArray` (Task 6, reused for the box output).
- Produces:
  - `class SegmentorAdapter : public ISegmentorAdapter`, `createSegmentorAdapter()`
  - `vision_msgs::msg::Detection2DArray conversion::segmentationToDetection2DArray(const std::vector<SegmentationResult>&, const std_msgs::msg::Header&)`
  - `sensor_msgs::msg::Image conversion::toCombinedMaskImage(const std::vector<SegmentationResult>&, int width, int height, const std_msgs::msg::Header&)`
  - `class YolosSegmentorNode`, component `ros2_yolos_trt::YolosSegmentorNode`, executable `yolos_trt_segmentor_node`

- [ ] **Step 1: Write the failing converter test**

```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include <gtest/gtest.h>

#include "ros2_yolos_trt/conversion/segmentation_converter.hpp"

using ros2_yolos_trt::SegmentationResult;
using ros2_yolos_trt::conversion::segmentationToDetection2DArray;
using ros2_yolos_trt::conversion::toCombinedMaskImage;

namespace {
SegmentationResult makeSeg(int x, int y, int w, int h, int id,
                           const std::string& name, const cv::Mat& mask) {
  SegmentationResult s;
  s.bbox.x = x;
  s.bbox.y = y;
  s.bbox.width = w;
  s.bbox.height = h;
  s.confidence = 0.8f;
  s.class_id = id;
  s.class_name = name;
  s.mask = mask;
  return s;
}
}  // namespace

TEST(SegmentationConverter, BoxesConvertToCentreSize) {
  const auto msg = segmentationToDetection2DArray(
    {makeSeg(10, 20, 30, 40, 1, "cat", cv::Mat::zeros(4, 4, CV_8UC1))},
    std_msgs::msg::Header{});
  ASSERT_EQ(msg.detections.size(), 1u);
  EXPECT_DOUBLE_EQ(msg.detections[0].bbox.center.position.x, 25.0);
  EXPECT_EQ(msg.detections[0].results[0].hypothesis.class_id, "cat");
}

// One mono8 image where each pixel holds the class id of the instance that
// covers it. Publishing one image per instance would flood the topic.
TEST(SegmentationConverter, CombinedMaskLabelsPixelsWithClassId) {
  cv::Mat mask = cv::Mat::zeros(4, 4, CV_8UC1);
  mask.at<uint8_t>(1, 1) = 255;  // one covered pixel

  const auto msg = toCombinedMaskImage(
    {makeSeg(0, 0, 4, 4, 7, "dog", mask)}, 4, 4, std_msgs::msg::Header{});

  EXPECT_EQ(msg.encoding, "mono8");
  EXPECT_EQ(msg.width, 4u);
  EXPECT_EQ(msg.height, 4u);
  ASSERT_EQ(msg.data.size(), 16u);
  EXPECT_EQ(msg.data[1 * 4 + 1], 7u);  // class id written through
  EXPECT_EQ(msg.data[0], 0u);          // background untouched
}

TEST(SegmentationConverter, EmptyInputProducesBlankMaskOfRequestedSize) {
  const auto msg = toCombinedMaskImage({}, 3, 2, std_msgs::msg::Header{});
  EXPECT_EQ(msg.width, 3u);
  EXPECT_EQ(msg.height, 2u);
  ASSERT_EQ(msg.data.size(), 6u);
  EXPECT_EQ(msg.data[0], 0u);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `colcon build --packages-select ros2_yolos_trt`
Expected: FAIL — `segmentation_converter.hpp` does not exist.

- [ ] **Step 3: Write the converter**

`include/ros2_yolos_trt/conversion/segmentation_converter.hpp`
```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ROS2_YOLOS_TRT__CONVERSION__SEGMENTATION_CONVERTER_HPP_
#define ROS2_YOLOS_TRT__CONVERSION__SEGMENTATION_CONVERTER_HPP_

#include <vector>

#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "ros2_yolos_trt/adapters/yolos_adapter_base.hpp"
#include "ros2_yolos_trt/visibility_control.hpp"

namespace ros2_yolos_trt {
namespace conversion {

/// @brief Boxes and scores from segmentation results.
ROS2_YOLOS_TRT_PUBLIC vision_msgs::msg::Detection2DArray
segmentationToDetection2DArray(
  const std::vector<SegmentationResult>& segmentations,
  const std_msgs::msg::Header& header);

/// @brief All instance masks flattened into one mono8 label image.
///
/// Each pixel holds the class id of the instance covering it, 0 = background.
/// Later instances overwrite earlier ones where they overlap.
ROS2_YOLOS_TRT_PUBLIC sensor_msgs::msg::Image toCombinedMaskImage(
  const std::vector<SegmentationResult>& segmentations, int width, int height,
  const std_msgs::msg::Header& header);

}  // namespace conversion
}  // namespace ros2_yolos_trt

#endif  // ROS2_YOLOS_TRT__CONVERSION__SEGMENTATION_CONVERTER_HPP_
```

`src/conversion/segmentation_converter.cpp`
```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_trt/conversion/segmentation_converter.hpp"

#include <cstring>
#include <string>

namespace ros2_yolos_trt {
namespace conversion {

vision_msgs::msg::Detection2DArray segmentationToDetection2DArray(
  const std::vector<SegmentationResult>& segmentations,
  const std_msgs::msg::Header& header) {
  vision_msgs::msg::Detection2DArray msg;
  msg.header = header;
  msg.detections.reserve(segmentations.size());

  for (const auto& seg : segmentations) {
    vision_msgs::msg::Detection2D d;
    d.header = header;
    d.bbox.center.position.x = seg.bbox.x + seg.bbox.width / 2.0;
    d.bbox.center.position.y = seg.bbox.y + seg.bbox.height / 2.0;
    d.bbox.center.theta = 0.0;
    d.bbox.size_x = static_cast<double>(seg.bbox.width);
    d.bbox.size_y = static_cast<double>(seg.bbox.height);

    vision_msgs::msg::ObjectHypothesisWithPose hyp;
    hyp.hypothesis.class_id = seg.class_name.empty()
                                ? std::to_string(seg.class_id)
                                : seg.class_name;
    hyp.hypothesis.score = static_cast<double>(seg.confidence);
    d.results.push_back(hyp);
    msg.detections.push_back(std::move(d));
  }
  return msg;
}

sensor_msgs::msg::Image toCombinedMaskImage(
  const std::vector<SegmentationResult>& segmentations, int width, int height,
  const std_msgs::msg::Header& header) {
  sensor_msgs::msg::Image msg;
  msg.header = header;
  msg.encoding = "mono8";
  msg.is_bigendian = 0;
  msg.width = static_cast<uint32_t>(width);
  msg.height = static_cast<uint32_t>(height);
  msg.step = static_cast<uint32_t>(width);
  msg.data.assign(static_cast<size_t>(width) * height, 0);

  for (const auto& seg : segmentations) {
    if (seg.mask.empty() || seg.mask.type() != CV_8UC1) continue;
    // Class ids above 255 cannot be represented in mono8; clamp rather than
    // wrapping, so a large-vocabulary model degrades visibly instead of
    // silently relabelling pixels.
    const auto label =
      static_cast<uint8_t>(seg.class_id > 255 ? 255 : seg.class_id);

    const int rows = std::min(height, seg.mask.rows);
    const int cols = std::min(width, seg.mask.cols);
    for (int r = 0; r < rows; ++r) {
      const auto* mask_row = seg.mask.ptr<uint8_t>(r);
      for (int c = 0; c < cols; ++c) {
        if (mask_row[c] != 0) {
          msg.data[static_cast<size_t>(r) * width + c] = label;
        }
      }
    }
  }
  return msg;
}

}  // namespace conversion
}  // namespace ros2_yolos_trt
```

- [ ] **Step 4: Write the adapter**

`include/ros2_yolos_trt/adapters/segmentor_adapter.hpp` — identical in shape to `detector_adapter.hpp`, with `ISegmentorAdapter` as the base and these overrides:

```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ROS2_YOLOS_TRT__ADAPTERS__SEGMENTOR_ADAPTER_HPP_
#define ROS2_YOLOS_TRT__ADAPTERS__SEGMENTOR_ADAPTER_HPP_

#include <memory>
#include <string>
#include <vector>

#include "ros2_yolos_trt/adapters/yolos_adapter_base.hpp"

namespace ros2_yolos_trt {

class ROS2_YOLOS_TRT_PUBLIC SegmentorAdapter : public ISegmentorAdapter {
public:
  SegmentorAdapter();
  ~SegmentorAdapter() override;

  SegmentorAdapter(const SegmentorAdapter&) = delete;
  SegmentorAdapter& operator=(const SegmentorAdapter&) = delete;
  SegmentorAdapter(SegmentorAdapter&&) noexcept;
  SegmentorAdapter& operator=(SegmentorAdapter&&) noexcept;

  bool initialize(const YolosConfig& config) override;
  [[nodiscard]] bool isInitialized() const noexcept override;
  void shutdown() override;
  [[nodiscard]] const std::vector<std::string>& getClassNames() const override;

  std::vector<SegmentationResult> segment(
    const cv::Mat& image, float conf_threshold, float nms_threshold) override;
  void drawSegmentations(
    cv::Mat& image, const std::vector<SegmentationResult>& segmentations,
    float mask_alpha) override;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ros2_yolos_trt

#endif  // ROS2_YOLOS_TRT__ADAPTERS__SEGMENTOR_ADAPTER_HPP_
```

`src/adapters/segmentor_adapter.cpp`
```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_trt/adapters/segmentor_adapter.hpp"

#include "yolos/tasks/segmentation.hpp"

#include <iostream>

#include "ros2_yolos_trt/engine/engine_cache.hpp"

namespace ros2_yolos_trt {

class SegmentorAdapter::Impl {
public:
  std::unique_ptr<yolos::seg::YOLOSegDetector> segmentor;
  std::vector<std::string> class_names;
  bool initialized{false};
};

SegmentorAdapter::SegmentorAdapter() : impl_(std::make_unique<Impl>()) {}
SegmentorAdapter::~SegmentorAdapter() = default;
SegmentorAdapter::SegmentorAdapter(SegmentorAdapter&&) noexcept = default;
SegmentorAdapter& SegmentorAdapter::operator=(SegmentorAdapter&&) noexcept = default;

bool SegmentorAdapter::initialize(const YolosConfig& config) {
  try {
    std::string error;
    const auto engine = engine::resolveEngine(
      config.model_path, config.precision, config.dla_core,
      config.engine_cache_dir, error);
    if (engine.empty()) {
      std::cerr << "[SegmentorAdapter] " << error << std::endl;
      return false;
    }
    impl_->segmentor = std::make_unique<yolos::seg::YOLOSegDetector>(
      engine.string(), config.labels_path, config.dla_core);
    impl_->class_names = impl_->segmentor->getClassNames();
    impl_->initialized = true;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[SegmentorAdapter] initialization error: " << e.what()
              << std::endl;
    impl_->initialized = false;
    return false;
  }
}

bool SegmentorAdapter::isInitialized() const noexcept {
  return impl_ && impl_->initialized;
}

void SegmentorAdapter::shutdown() {
  if (impl_) {
    impl_->segmentor.reset();
    impl_->initialized = false;
  }
}

const std::vector<std::string>& SegmentorAdapter::getClassNames() const {
  static const std::vector<std::string> empty;
  return impl_ ? impl_->class_names : empty;
}

std::vector<SegmentationResult> SegmentorAdapter::segment(
  const cv::Mat& image, float conf_threshold, float nms_threshold) {
  std::vector<SegmentationResult> results;
  if (!isInitialized() || image.empty()) return results;

  try {
    const auto segs = impl_->segmentor->segment(image, conf_threshold, nms_threshold);
    results.reserve(segs.size());
    for (const auto& seg : segs) {
      SegmentationResult r;
      r.bbox.x = seg.box.x;
      r.bbox.y = seg.box.y;
      r.bbox.width = seg.box.width;
      r.bbox.height = seg.box.height;
      r.confidence = seg.conf;
      r.class_id = seg.classId;
      r.mask = seg.mask;  // CV_8UC1 at original image size
      if (seg.classId >= 0 &&
          static_cast<size_t>(seg.classId) < impl_->class_names.size()) {
        r.class_name = impl_->class_names[seg.classId];
      }
      results.push_back(std::move(r));
    }
  } catch (const std::exception& e) {
    std::cerr << "[SegmentorAdapter] segmentation error: " << e.what() << std::endl;
  }
  return results;
}

void SegmentorAdapter::drawSegmentations(
  cv::Mat& image, const std::vector<SegmentationResult>& segmentations,
  float mask_alpha) {
  if (!isInitialized() || image.empty()) return;

  std::vector<yolos::seg::Segmentation> native;
  native.reserve(segmentations.size());
  for (const auto& seg : segmentations) {
    yolos::seg::Segmentation s;
    s.box = yolos::BoundingBox(seg.bbox.x, seg.bbox.y, seg.bbox.width,
                               seg.bbox.height);
    s.conf = seg.confidence;
    s.classId = seg.class_id;
    s.mask = seg.mask;
    native.push_back(std::move(s));
  }
  impl_->segmentor->drawSegmentations(image, native, mask_alpha);
}

std::unique_ptr<ISegmentorAdapter> createSegmentorAdapter() {
  return std::make_unique<SegmentorAdapter>();
}

}  // namespace ros2_yolos_trt
```

Note: confirm the upstream `drawSegmentations` signature and the `Segmentation` field names in `${YOLOS_TRT_DIR}/include/yolos/tasks/segmentation.hpp` before building; adjust the member assignments if they differ.

- [ ] **Step 5: Write the node**

`include/ros2_yolos_trt/nodes/segmentor_node.hpp` and `src/nodes/segmentor_node.cpp` follow `detector_node` exactly, with these differences:
- class `YolosSegmentorNode`, node name `"yolos_trt_segmentor"`, component `ros2_yolos_trt::YolosSegmentorNode`
- `std::unique_ptr<ISegmentorAdapter> adapter_` created via `createSegmentorAdapter()`
- two data publishers created in `on_configure`:
```cpp
  det_pub_ = create_publisher<vision_msgs::msg::Detection2DArray>("~/detections", 10);
  mask_pub_ = create_publisher<sensor_msgs::msg::Image>("~/masks", 10);
```
- callback body between the cv_bridge conversion and the timing publication:
```cpp
  const auto t_infer_start = std::chrono::steady_clock::now();
  auto segs = adapter_->segment(frame, config_.conf_threshold,
                                config_.nms_threshold);
  const auto t_infer_end = std::chrono::steady_clock::now();

  det_pub_->publish(conversion::segmentationToDetection2DArray(segs, msg->header));
  if (mask_pub_ && mask_pub_->is_activated()) {
    mask_pub_->publish(conversion::toCombinedMaskImage(
      segs, frame.cols, frame.rows, msg->header));
  }

  if (debug_pub_ && debug_pub_->is_activated()) {
    adapter_->drawSegmentations(frame, segs, 0.5f);
    debug_pub_->publish(
      *cv_bridge::CvImage(msg->header, "bgr8", frame).toImageMsg());
  }
```
- `on_cleanup` additionally resets `mask_pub_`.

- [ ] **Step 6: Write `launch/segmentor.launch.py`**

Copy `launch/detector.launch.py` and change `executable` to `yolos_trt_segmentor_node` and `name` to `yolos_trt_segmentor`.

- [ ] **Step 7: Register in `CMakeLists.txt`**

```cmake
  src/adapters/segmentor_adapter.cpp
  src/conversion/segmentation_converter.cpp
  src/nodes/segmentor_node.cpp
```
```cmake
rclcpp_components_register_node(${PROJECT_NAME}_components
  PLUGIN "ros2_yolos_trt::YolosSegmentorNode"
  EXECUTABLE yolos_trt_segmentor_node)
```
```cmake
  ament_add_gtest(test_segmentation_converter test/unit/test_segmentation_converter.cpp)
  target_link_libraries(test_segmentation_converter ${PROJECT_NAME}_components)
  ament_target_dependencies(test_segmentation_converter vision_msgs sensor_msgs std_msgs)
```

- [ ] **Step 8: Run tests to verify they pass**

Run:
```bash
colcon build --packages-select ros2_yolos_trt
colcon test --packages-select ros2_yolos_trt --ctest-args -R test_segmentation_converter
colcon test-result --verbose
```
Expected: PASS, 3 cases.

- [ ] **Step 9: Commit**

```bash
git add include/ros2_yolos_trt/adapters/segmentor_adapter.hpp src/adapters/segmentor_adapter.cpp include/ros2_yolos_trt/conversion/segmentation_converter.hpp src/conversion/segmentation_converter.cpp include/ros2_yolos_trt/nodes/segmentor_node.hpp src/nodes/segmentor_node.cpp launch/segmentor.launch.py test/unit/test_segmentation_converter.cpp CMakeLists.txt
git commit -m "feat: add segmentation node, adapter and converters

Instance masks are flattened into one mono8 label image where each pixel
carries the class id of the instance covering it, rather than publishing one
image per instance and flooding the topic. Class ids above 255 clamp instead
of wrapping, so a large-vocabulary model degrades visibly rather than
silently relabelling pixels."
```

---

### Task 9: Pose slice — including the keypoints the reference package drops

`ros2_yolos_cpp` ships `KeyPoint2D.msg` and `KeyPoint2DArray.msg`, forward-declares the converter, and never implements or publishes it — so keypoints are computed and discarded, because `Detection2DArray` has no keypoint field. This task implements and publishes them.

**Files:**
- Create: `include/ros2_yolos_trt/adapters/pose_adapter.hpp`, `src/adapters/pose_adapter.cpp`
- Create: `include/ros2_yolos_trt/conversion/pose_converter.hpp`, `src/conversion/pose_converter.cpp`
- Create: `include/ros2_yolos_trt/nodes/pose_node.hpp`, `src/nodes/pose_node.cpp`
- Create: `launch/pose.launch.py`
- Test: `test/unit/test_pose_converter.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces:
  - `class PoseAdapter : public IPoseAdapter`, `createPoseAdapter()`
  - `vision_msgs::msg::Detection2DArray conversion::poseToDetection2DArray(const std::vector<PoseResult>&, const std_msgs::msg::Header&)`
  - `std::vector<ros2_yolos_trt::msg::KeyPoint2DArray> conversion::toKeyPoint2DArrays(const std::vector<PoseResult>&, const std_msgs::msg::Header&)`
  - `class YolosPoseNode`, component `ros2_yolos_trt::YolosPoseNode`, executable `yolos_trt_pose_node`

- [ ] **Step 1: Write the failing converter test**

```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include <gtest/gtest.h>

#include "ros2_yolos_trt/conversion/pose_converter.hpp"

using ros2_yolos_trt::KeyPoint2D;
using ros2_yolos_trt::PoseResult;
using ros2_yolos_trt::conversion::poseToDetection2DArray;
using ros2_yolos_trt::conversion::toKeyPoint2DArrays;

namespace {
PoseResult makePose() {
  PoseResult p;
  p.bbox.x = 10;
  p.bbox.y = 20;
  p.bbox.width = 30;
  p.bbox.height = 60;
  p.confidence = 0.9f;
  p.class_id = 0;
  p.class_name = "person";
  p.keypoints = {
    KeyPoint2D{15.0f, 25.0f, 0.95f, 0},
    KeyPoint2D{20.0f, 35.0f, 0.40f, 1},
  };
  return p;
}
}  // namespace

TEST(PoseConverter, BoxesConvertToCentreSize) {
  const auto msg = poseToDetection2DArray({makePose()}, std_msgs::msg::Header{});
  ASSERT_EQ(msg.detections.size(), 1u);
  EXPECT_DOUBLE_EQ(msg.detections[0].bbox.center.position.x, 25.0);
  EXPECT_DOUBLE_EQ(msg.detections[0].bbox.center.position.y, 50.0);
}

// The whole point of this task: keypoints must survive to a message.
TEST(PoseConverter, KeypointsArePreservedPerInstance) {
  const auto arrays = toKeyPoint2DArrays({makePose()}, std_msgs::msg::Header{});
  ASSERT_EQ(arrays.size(), 1u);
  ASSERT_EQ(arrays[0].keypoints.size(), 2u);
  EXPECT_DOUBLE_EQ(arrays[0].keypoints[0].x, 15.0);
  EXPECT_DOUBLE_EQ(arrays[0].keypoints[0].y, 25.0);
  EXPECT_EQ(arrays[0].keypoints[1].id, 1);
  EXPECT_NEAR(arrays[0].keypoints[1].confidence, 0.40, 1e-6);
}

TEST(PoseConverter, KeypointArrayCarriesInstanceBoxAndClass) {
  const auto arrays = toKeyPoint2DArrays({makePose()}, std_msgs::msg::Header{});
  ASSERT_EQ(arrays.size(), 1u);
  EXPECT_DOUBLE_EQ(arrays[0].bounding_box_center.x, 25.0);
  EXPECT_DOUBLE_EQ(arrays[0].bounding_box_size_x, 30.0);
  EXPECT_EQ(arrays[0].class_id, 0);
  EXPECT_NEAR(arrays[0].confidence, 0.9, 1e-6);
}

TEST(PoseConverter, EmptyInputProducesNoArrays) {
  EXPECT_TRUE(toKeyPoint2DArrays({}, std_msgs::msg::Header{}).empty());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `colcon build --packages-select ros2_yolos_trt`
Expected: FAIL — `pose_converter.hpp` does not exist.

- [ ] **Step 3: Write the converter**

`include/ros2_yolos_trt/conversion/pose_converter.hpp`
```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ROS2_YOLOS_TRT__CONVERSION__POSE_CONVERTER_HPP_
#define ROS2_YOLOS_TRT__CONVERSION__POSE_CONVERTER_HPP_

#include <vector>

#include <std_msgs/msg/header.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "ros2_yolos_trt/adapters/yolos_adapter_base.hpp"
#include "ros2_yolos_trt/msg/key_point2_d_array.hpp"
#include "ros2_yolos_trt/visibility_control.hpp"

namespace ros2_yolos_trt {
namespace conversion {

/// @brief Person boxes from pose results.
ROS2_YOLOS_TRT_PUBLIC vision_msgs::msg::Detection2DArray poseToDetection2DArray(
  const std::vector<PoseResult>& poses, const std_msgs::msg::Header& header);

/// @brief One KeyPoint2DArray per detected instance.
///
/// Detection2DArray has no keypoint field, so without this the keypoints —
/// the actual output of pose estimation — would be discarded.
ROS2_YOLOS_TRT_PUBLIC std::vector<msg::KeyPoint2DArray> toKeyPoint2DArrays(
  const std::vector<PoseResult>& poses, const std_msgs::msg::Header& header);

}  // namespace conversion
}  // namespace ros2_yolos_trt

#endif  // ROS2_YOLOS_TRT__CONVERSION__POSE_CONVERTER_HPP_
```

`src/conversion/pose_converter.cpp`
```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_trt/conversion/pose_converter.hpp"

#include <string>

namespace ros2_yolos_trt {
namespace conversion {

vision_msgs::msg::Detection2DArray poseToDetection2DArray(
  const std::vector<PoseResult>& poses, const std_msgs::msg::Header& header) {
  vision_msgs::msg::Detection2DArray msg;
  msg.header = header;
  msg.detections.reserve(poses.size());

  for (const auto& pose : poses) {
    vision_msgs::msg::Detection2D d;
    d.header = header;
    d.bbox.center.position.x = pose.bbox.x + pose.bbox.width / 2.0;
    d.bbox.center.position.y = pose.bbox.y + pose.bbox.height / 2.0;
    d.bbox.center.theta = 0.0;
    d.bbox.size_x = static_cast<double>(pose.bbox.width);
    d.bbox.size_y = static_cast<double>(pose.bbox.height);

    vision_msgs::msg::ObjectHypothesisWithPose hyp;
    hyp.hypothesis.class_id = pose.class_name.empty()
                                ? std::to_string(pose.class_id)
                                : pose.class_name;
    hyp.hypothesis.score = static_cast<double>(pose.confidence);
    d.results.push_back(hyp);
    msg.detections.push_back(std::move(d));
  }
  return msg;
}

std::vector<msg::KeyPoint2DArray> toKeyPoint2DArrays(
  const std::vector<PoseResult>& poses, const std_msgs::msg::Header& header) {
  std::vector<msg::KeyPoint2DArray> arrays;
  arrays.reserve(poses.size());

  for (const auto& pose : poses) {
    msg::KeyPoint2DArray arr;
    arr.header = header;
    arr.confidence = static_cast<double>(pose.confidence);
    arr.class_id = pose.class_id;
    arr.bounding_box_center.x = pose.bbox.x + pose.bbox.width / 2.0;
    arr.bounding_box_center.y = pose.bbox.y + pose.bbox.height / 2.0;
    arr.bounding_box_center.z = 0.0;
    arr.bounding_box_size_x = static_cast<double>(pose.bbox.width);
    arr.bounding_box_size_y = static_cast<double>(pose.bbox.height);

    arr.keypoints.reserve(pose.keypoints.size());
    for (const auto& kp : pose.keypoints) {
      msg::KeyPoint2D out;
      out.x = static_cast<double>(kp.x);
      out.y = static_cast<double>(kp.y);
      out.confidence = static_cast<double>(kp.confidence);
      out.id = kp.id;
      arr.keypoints.push_back(out);
    }
    arrays.push_back(std::move(arr));
  }
  return arrays;
}

}  // namespace conversion
}  // namespace ros2_yolos_trt
```

- [ ] **Step 4: Write the adapter**

`include/ros2_yolos_trt/adapters/pose_adapter.hpp` mirrors `segmentor_adapter.hpp` with base `IPoseAdapter` and overrides `detect(...)` and `drawPoses(cv::Mat&, const std::vector<PoseResult>&, int, float)`.

`src/adapters/pose_adapter.cpp` mirrors `segmentor_adapter.cpp` with `yolos::pose::YOLOPoseDetector` from `"yolos/tasks/pose.hpp"`, and this result mapping in `detect`:

```cpp
    const auto native = impl_->pose->detect(image, conf_threshold, nms_threshold);
    results.reserve(native.size());
    for (const auto& p : native) {
      PoseResult r;
      r.bbox.x = p.box.x;
      r.bbox.y = p.box.y;
      r.bbox.width = p.box.width;
      r.bbox.height = p.box.height;
      r.confidence = p.conf;
      r.class_id = p.classId;
      if (p.classId >= 0 &&
          static_cast<size_t>(p.classId) < impl_->class_names.size()) {
        r.class_name = impl_->class_names[p.classId];
      }
      r.keypoints.reserve(p.keypoints.size());
      int kpt_id = 0;
      for (const auto& kp : p.keypoints) {
        // Upstream KeyPoint carries x, y and conf; the index is its position
        // in the vector, which is the COCO ordering for the released models.
        r.keypoints.push_back(KeyPoint2D{kp.x, kp.y, kp.conf, kpt_id++});
      }
      results.push_back(std::move(r));
    }
```

Confirm the upstream `KeyPoint` field names in `${YOLOS_TRT_DIR}/include/yolos/core/types.hpp` before building.

- [ ] **Step 5: Write the node**

`pose_node.hpp` / `pose_node.cpp` mirror `detector_node` with:
- class `YolosPoseNode`, node name `"yolos_trt_pose"`, component `ros2_yolos_trt::YolosPoseNode`
- `createPoseAdapter()`, `std::unique_ptr<IPoseAdapter> adapter_`
- publishers created in `on_configure`:
```cpp
  det_pub_ = create_publisher<vision_msgs::msg::Detection2DArray>("~/detections", 10);
  kpt_pub_ = create_publisher<msg::KeyPoint2DArray>("~/keypoints", 10);
```
- callback inference and publication:
```cpp
  const auto t_infer_start = std::chrono::steady_clock::now();
  auto poses = adapter_->detect(frame, config_.conf_threshold,
                                config_.nms_threshold);
  const auto t_infer_end = std::chrono::steady_clock::now();

  det_pub_->publish(conversion::poseToDetection2DArray(poses, msg->header));
  if (kpt_pub_ && kpt_pub_->is_activated()) {
    // One message per instance: a person's keypoints belong together.
    for (const auto& arr : conversion::toKeyPoint2DArrays(poses, msg->header)) {
      kpt_pub_->publish(arr);
    }
  }

  if (debug_pub_ && debug_pub_->is_activated()) {
    adapter_->drawPoses(frame, poses, 4, 0.5f);
    debug_pub_->publish(
      *cv_bridge::CvImage(msg->header, "bgr8", frame).toImageMsg());
  }
```
- `on_cleanup` additionally resets `kpt_pub_`.

- [ ] **Step 6: Write `launch/pose.launch.py`**

Copy `launch/detector.launch.py`, change `executable` to `yolos_trt_pose_node` and `name` to `yolos_trt_pose`.

- [ ] **Step 7: Register in `CMakeLists.txt`**

```cmake
  src/adapters/pose_adapter.cpp
  src/conversion/pose_converter.cpp
  src/nodes/pose_node.cpp
```
```cmake
rclcpp_components_register_node(${PROJECT_NAME}_components
  PLUGIN "ros2_yolos_trt::YolosPoseNode"
  EXECUTABLE yolos_trt_pose_node)
```
```cmake
  ament_add_gtest(test_pose_converter test/unit/test_pose_converter.cpp)
  target_link_libraries(test_pose_converter ${PROJECT_NAME}_components "${cpp_typesupport_target}")
  ament_target_dependencies(test_pose_converter vision_msgs std_msgs geometry_msgs)
```

- [ ] **Step 8: Run tests to verify they pass**

Run:
```bash
colcon build --packages-select ros2_yolos_trt
colcon test --packages-select ros2_yolos_trt --ctest-args -R test_pose_converter
colcon test-result --verbose
```
Expected: PASS, 4 cases.

- [ ] **Step 9: Commit**

```bash
git add include/ros2_yolos_trt/adapters/pose_adapter.hpp src/adapters/pose_adapter.cpp include/ros2_yolos_trt/conversion/pose_converter.hpp src/conversion/pose_converter.cpp include/ros2_yolos_trt/nodes/pose_node.hpp src/nodes/pose_node.cpp launch/pose.launch.py test/unit/test_pose_converter.cpp CMakeLists.txt
git commit -m "feat: add pose node and publish keypoints

Publishes one KeyPoint2DArray per detected instance on ~/keypoints, in
addition to boxes on ~/detections.

The ONNX package defines KeyPoint2D and KeyPoint2DArray and forward-declares
the converter, but never implements or publishes it — and Detection2DArray
has no keypoint field, so the keypoints were computed and discarded. Since
keypoints are the output of pose estimation, they are published here."
```

---
### Task 10: OBB slice

**Files:**
- Create: `include/ros2_yolos_trt/adapters/obb_adapter.hpp`, `src/adapters/obb_adapter.cpp`
- Create: `include/ros2_yolos_trt/conversion/obb_converter.hpp`, `src/conversion/obb_converter.cpp`
- Create: `include/ros2_yolos_trt/nodes/obb_node.hpp`, `src/nodes/obb_node.cpp`
- Create: `launch/obb.launch.py`
- Test: `test/unit/test_obb_converter.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces:
  - `class OBBAdapter : public IOBBAdapter`, `createOBBAdapter()`
  - `msg::OBBDetection2DArray conversion::toOBBDetection2DArray(const std::vector<OBBResult>&, const std_msgs::msg::Header&)`
  - `class YolosOBBNode`, component `ros2_yolos_trt::YolosOBBNode`, executable `yolos_trt_obb_node`

- [ ] **Step 1: Write the failing converter test**

```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include <gtest/gtest.h>

#include "ros2_yolos_trt/conversion/obb_converter.hpp"

using ros2_yolos_trt::OBBResult;
using ros2_yolos_trt::conversion::toOBBDetection2DArray;

namespace {
OBBResult makeObb() {
  OBBResult o;
  o.bbox.center_x = 100.0f;
  o.bbox.center_y = 200.0f;
  o.bbox.width = 50.0f;
  o.bbox.height = 25.0f;
  o.bbox.angle = 0.7853982f;  // 45 degrees
  o.confidence = 0.88f;
  o.class_id = 4;
  o.class_name = "ship";
  return o;
}
}  // namespace

// The OBB DTO already stores a centre, so unlike the axis-aligned converters
// there is no origin shift — the angle is what must survive.
TEST(OBBConverter, PreservesCentreExtentAndAngle) {
  const auto msg = toOBBDetection2DArray({makeObb()}, std_msgs::msg::Header{});
  ASSERT_EQ(msg.detections.size(), 1u);
  const auto& d = msg.detections[0];
  EXPECT_DOUBLE_EQ(d.bbox.center.x, 100.0);
  EXPECT_DOUBLE_EQ(d.bbox.center.y, 200.0);
  EXPECT_DOUBLE_EQ(d.bbox.size_x, 50.0);
  EXPECT_DOUBLE_EQ(d.bbox.size_y, 25.0);
  EXPECT_NEAR(d.bbox.theta, 0.7853982, 1e-6);
}

TEST(OBBConverter, CarriesClassAndScore) {
  const auto msg = toOBBDetection2DArray({makeObb()}, std_msgs::msg::Header{});
  ASSERT_EQ(msg.detections.size(), 1u);
  EXPECT_EQ(msg.detections[0].class_id, 4);
  EXPECT_EQ(msg.detections[0].class_name, "ship");
  EXPECT_NEAR(msg.detections[0].score, 0.88, 1e-6);
}

TEST(OBBConverter, EmptyInputProducesEmptyArrayWithHeader) {
  std_msgs::msg::Header header;
  header.frame_id = "sat";
  const auto msg = toOBBDetection2DArray({}, header);
  EXPECT_TRUE(msg.detections.empty());
  EXPECT_EQ(msg.header.frame_id, "sat");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `colcon build --packages-select ros2_yolos_trt`
Expected: FAIL — `obb_converter.hpp` does not exist.

- [ ] **Step 3: Write the converter**

`include/ros2_yolos_trt/conversion/obb_converter.hpp`
```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ROS2_YOLOS_TRT__CONVERSION__OBB_CONVERTER_HPP_
#define ROS2_YOLOS_TRT__CONVERSION__OBB_CONVERTER_HPP_

#include <vector>

#include <std_msgs/msg/header.hpp>

#include "ros2_yolos_trt/adapters/yolos_adapter_base.hpp"
#include "ros2_yolos_trt/msg/obb_detection2_d_array.hpp"
#include "ros2_yolos_trt/visibility_control.hpp"

namespace ros2_yolos_trt {
namespace conversion {

/// @brief Convert OBB DTOs to the package's OBBDetection2DArray message.
///
/// vision_msgs has no oriented box type, so the rotation angle would be lost
/// if these were published as Detection2DArray.
ROS2_YOLOS_TRT_PUBLIC msg::OBBDetection2DArray toOBBDetection2DArray(
  const std::vector<OBBResult>& detections,
  const std_msgs::msg::Header& header);

}  // namespace conversion
}  // namespace ros2_yolos_trt

#endif  // ROS2_YOLOS_TRT__CONVERSION__OBB_CONVERTER_HPP_
```

`src/conversion/obb_converter.cpp`
```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_trt/conversion/obb_converter.hpp"

namespace ros2_yolos_trt {
namespace conversion {

msg::OBBDetection2DArray toOBBDetection2DArray(
  const std::vector<OBBResult>& detections,
  const std_msgs::msg::Header& header) {
  msg::OBBDetection2DArray out;
  out.header = header;
  out.detections.reserve(detections.size());

  for (const auto& det : detections) {
    msg::OBBDetection2D d;
    d.header = header;
    d.bbox.center.x = static_cast<double>(det.bbox.center_x);
    d.bbox.center.y = static_cast<double>(det.bbox.center_y);
    d.bbox.center.z = 0.0;
    d.bbox.size_x = static_cast<double>(det.bbox.width);
    d.bbox.size_y = static_cast<double>(det.bbox.height);
    d.bbox.theta = static_cast<double>(det.bbox.angle);
    d.class_id = det.class_id;
    d.class_name = det.class_name;
    d.score = static_cast<double>(det.confidence);
    out.detections.push_back(std::move(d));
  }
  return out;
}

}  // namespace conversion
}  // namespace ros2_yolos_trt
```

- [ ] **Step 4: Write the adapter**

`obb_adapter.hpp` mirrors `segmentor_adapter.hpp` with base `IOBBAdapter`, overriding `detect(const cv::Mat&, float, float, int)` and `drawDetections(cv::Mat&, const std::vector<OBBResult>&)`.

`src/adapters/obb_adapter.cpp` mirrors `segmentor_adapter.cpp` using `yolos::obb::YOLOOBBDetector` from `"yolos/tasks/obb.hpp"`, with:

```cpp
    const auto native =
      impl_->obb->detect(image, conf_threshold, nms_threshold, max_det);
    results.reserve(native.size());
    for (const auto& o : native) {
      OBBResult r;
      r.bbox.center_x = o.box.cx;
      r.bbox.center_y = o.box.cy;
      r.bbox.width = o.box.width;
      r.bbox.height = o.box.height;
      r.bbox.angle = o.box.angle;
      r.confidence = o.conf;
      r.class_id = o.classId;
      if (o.classId >= 0 &&
          static_cast<size_t>(o.classId) < impl_->class_names.size()) {
        r.class_name = impl_->class_names[o.classId];
      }
      results.push_back(std::move(r));
    }
```

Confirm the upstream `OBBResult` box field names in `${YOLOS_TRT_DIR}/include/yolos/tasks/obb.hpp` before building.

- [ ] **Step 5: Write the node**

`obb_node.hpp` / `obb_node.cpp` mirror `detector_node` with class `YolosOBBNode`, node name `"yolos_trt_obb"`, `createOBBAdapter()`, publisher:
```cpp
  obb_pub_ = create_publisher<msg::OBBDetection2DArray>("~/detections", 10);
```
and callback inference using the max_detections parameter:
```cpp
  auto dets = adapter_->detect(frame, config_.conf_threshold,
                               config_.nms_threshold, maxDetections());
  obb_pub_->publish(conversion::toOBBDetection2DArray(dets, msg->header));
```

- [ ] **Step 6: Write `launch/obb.launch.py`**

Copy `launch/detector.launch.py`, change `executable` to `yolos_trt_obb_node`, `name` to `yolos_trt_obb`, and add a `max_detections` argument defaulting to `'300'` wired into the parameter dict.

- [ ] **Step 7: Register in `CMakeLists.txt`**

```cmake
  src/adapters/obb_adapter.cpp
  src/conversion/obb_converter.cpp
  src/nodes/obb_node.cpp
```
```cmake
rclcpp_components_register_node(${PROJECT_NAME}_components
  PLUGIN "ros2_yolos_trt::YolosOBBNode"
  EXECUTABLE yolos_trt_obb_node)
```
```cmake
  ament_add_gtest(test_obb_converter test/unit/test_obb_converter.cpp)
  target_link_libraries(test_obb_converter ${PROJECT_NAME}_components "${cpp_typesupport_target}")
  ament_target_dependencies(test_obb_converter std_msgs geometry_msgs)
```

- [ ] **Step 8: Run tests to verify they pass**

Run:
```bash
colcon build --packages-select ros2_yolos_trt
colcon test --packages-select ros2_yolos_trt --ctest-args -R test_obb_converter
colcon test-result --verbose
```
Expected: PASS, 3 cases.

- [ ] **Step 9: Commit**

```bash
git add include/ros2_yolos_trt/adapters/obb_adapter.hpp src/adapters/obb_adapter.cpp include/ros2_yolos_trt/conversion/obb_converter.hpp src/conversion/obb_converter.cpp include/ros2_yolos_trt/nodes/obb_node.hpp src/nodes/obb_node.cpp launch/obb.launch.py test/unit/test_obb_converter.cpp CMakeLists.txt
git commit -m "feat: add oriented bounding box node, adapter and converter

Publishes the package-local OBBDetection2DArray rather than
vision_msgs/Detection2DArray, which has no oriented box type and would
silently drop the rotation angle that makes an OBB model worth running."
```

---

### Task 11: Classification slice

**Files:**
- Create: `include/ros2_yolos_trt/adapters/classifier_adapter.hpp`, `src/adapters/classifier_adapter.cpp`
- Create: `include/ros2_yolos_trt/conversion/classification_converter.hpp`, `src/conversion/classification_converter.cpp`
- Create: `include/ros2_yolos_trt/nodes/classifier_node.hpp`, `src/nodes/classifier_node.cpp`
- Create: `launch/classifier.launch.py`
- Test: `test/unit/test_classification_converter.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces:
  - `class ClassifierAdapter : public IClassifierAdapter`, `createClassifierAdapter()`
  - `vision_msgs::msg::Classification conversion::toClassification(const ClassificationResult&, const std_msgs::msg::Header&)`
  - `class YolosClassifierNode`, component `ros2_yolos_trt::YolosClassifierNode`, executable `yolos_trt_classifier_node`

- [ ] **Step 1: Write the failing converter test**

```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include <gtest/gtest.h>

#include "ros2_yolos_trt/conversion/classification_converter.hpp"

using ros2_yolos_trt::ClassificationResult;
using ros2_yolos_trt::conversion::toClassification;

TEST(ClassificationConverter, CarriesTopOneHypothesis) {
  ClassificationResult r;
  r.class_id = 207;
  r.confidence = 0.91f;
  r.class_name = "golden retriever";

  std_msgs::msg::Header header;
  header.frame_id = "cam";
  const auto msg = toClassification(r, header);

  ASSERT_EQ(msg.results.size(), 1u);
  EXPECT_EQ(msg.results[0].class_id, "golden retriever");
  EXPECT_NEAR(msg.results[0].score, 0.91, 1e-6);
  EXPECT_EQ(msg.header.frame_id, "cam");
}

TEST(ClassificationConverter, FallsBackToNumericIdWithoutLabels) {
  ClassificationResult r;
  r.class_id = 5;
  r.confidence = 0.5f;
  const auto msg = toClassification(r, std_msgs::msg::Header{});
  ASSERT_EQ(msg.results.size(), 1u);
  EXPECT_EQ(msg.results[0].class_id, "5");
}

// class_id -1 is the adapter's "nothing produced" sentinel.
TEST(ClassificationConverter, NegativeClassIdProducesNoHypothesis) {
  ClassificationResult r;
  r.class_id = -1;
  const auto msg = toClassification(r, std_msgs::msg::Header{});
  EXPECT_TRUE(msg.results.empty());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `colcon build --packages-select ros2_yolos_trt`
Expected: FAIL — `classification_converter.hpp` does not exist.

- [ ] **Step 3: Write the converter**

`include/ros2_yolos_trt/conversion/classification_converter.hpp`
```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ROS2_YOLOS_TRT__CONVERSION__CLASSIFICATION_CONVERTER_HPP_
#define ROS2_YOLOS_TRT__CONVERSION__CLASSIFICATION_CONVERTER_HPP_

#include <std_msgs/msg/header.hpp>
#include <vision_msgs/msg/classification.hpp>

#include "ros2_yolos_trt/adapters/yolos_adapter_base.hpp"
#include "ros2_yolos_trt/visibility_control.hpp"

namespace ros2_yolos_trt {
namespace conversion {

/// @brief Convert a top-1 classification result to vision_msgs/Classification.
ROS2_YOLOS_TRT_PUBLIC vision_msgs::msg::Classification toClassification(
  const ClassificationResult& result, const std_msgs::msg::Header& header);

}  // namespace conversion
}  // namespace ros2_yolos_trt

#endif  // ROS2_YOLOS_TRT__CONVERSION__CLASSIFICATION_CONVERTER_HPP_
```

`src/conversion/classification_converter.cpp`
```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_trt/conversion/classification_converter.hpp"

#include <string>

namespace ros2_yolos_trt {
namespace conversion {

vision_msgs::msg::Classification toClassification(
  const ClassificationResult& result, const std_msgs::msg::Header& header) {
  vision_msgs::msg::Classification msg;
  msg.header = header;

  // The adapter uses -1 to mean "no result"; emitting a hypothesis for it
  // would look like a confident prediction of class -1.
  if (result.class_id < 0) return msg;

  vision_msgs::msg::ObjectHypothesis hyp;
  hyp.class_id = result.class_name.empty() ? std::to_string(result.class_id)
                                           : result.class_name;
  hyp.score = static_cast<double>(result.confidence);
  msg.results.push_back(hyp);
  return msg;
}

}  // namespace conversion
}  // namespace ros2_yolos_trt
```

- [ ] **Step 4: Write the adapter**

`classifier_adapter.hpp` mirrors `segmentor_adapter.hpp` with base `IClassifierAdapter`, overriding `classify(const cv::Mat&)` and `drawResult(cv::Mat&, const ClassificationResult&)`.

`src/adapters/classifier_adapter.cpp` mirrors `segmentor_adapter.cpp` using `yolos::cls::YOLOClassifier` from `"yolos/tasks/classification.hpp"`:

```cpp
    impl_->classifier = std::make_unique<yolos::cls::YOLOClassifier>(
      engine.string(), config.labels_path, cv::Size(224, 224), config.dla_core);
```
and:
```cpp
ClassificationResult ClassifierAdapter::classify(const cv::Mat& image) {
  ClassificationResult out;
  out.class_id = -1;  // sentinel: nothing produced
  if (!isInitialized() || image.empty()) return out;
  try {
    const auto native = impl_->classifier->classify(image);
    out.class_id = native.classId;
    out.confidence = native.confidence;
    out.class_name = native.className;
  } catch (const std::exception& e) {
    std::cerr << "[ClassifierAdapter] classification error: " << e.what()
              << std::endl;
    out.class_id = -1;
  }
  return out;
}
```

- [ ] **Step 5: Write the node**

`classifier_node.hpp` / `classifier_node.cpp` mirror `detector_node` with class `YolosClassifierNode`, node name `"yolos_trt_classifier"`, `createClassifierAdapter()`, publisher:
```cpp
  cls_pub_ = create_publisher<vision_msgs::msg::Classification>("~/classification", 10);
```
and callback:
```cpp
  auto result = adapter_->classify(frame);
  cls_pub_->publish(conversion::toClassification(result, msg->header));
  if (debug_pub_ && debug_pub_->is_activated()) {
    adapter_->drawResult(frame, result);
    debug_pub_->publish(
      *cv_bridge::CvImage(msg->header, "bgr8", frame).toImageMsg());
  }
```

- [ ] **Step 6: Write `launch/classifier.launch.py`**

Copy `launch/detector.launch.py`, change `executable` to `yolos_trt_classifier_node` and `name` to `yolos_trt_classifier`.

- [ ] **Step 7: Register in `CMakeLists.txt`**

```cmake
  src/adapters/classifier_adapter.cpp
  src/conversion/classification_converter.cpp
  src/nodes/classifier_node.cpp
```
```cmake
rclcpp_components_register_node(${PROJECT_NAME}_components
  PLUGIN "ros2_yolos_trt::YolosClassifierNode"
  EXECUTABLE yolos_trt_classifier_node)
```
```cmake
  ament_add_gtest(test_classification_converter test/unit/test_classification_converter.cpp)
  target_link_libraries(test_classification_converter ${PROJECT_NAME}_components)
  ament_target_dependencies(test_classification_converter vision_msgs std_msgs)
```

- [ ] **Step 8: Run tests to verify they pass**

Run:
```bash
colcon build --packages-select ros2_yolos_trt
colcon test --packages-select ros2_yolos_trt
colcon test-result --verbose
```
Expected: every unit test in the package passes — all six converters, the DTOs, the engine cache, the lifecycle base, the messages and the build smoke test.

- [ ] **Step 9: Commit**

```bash
git add include/ros2_yolos_trt/adapters/classifier_adapter.hpp src/adapters/classifier_adapter.cpp include/ros2_yolos_trt/conversion/classification_converter.hpp src/conversion/classification_converter.cpp include/ros2_yolos_trt/nodes/classifier_node.hpp src/nodes/classifier_node.cpp launch/classifier.launch.py test/unit/test_classification_converter.cpp CMakeLists.txt
git commit -m "feat: add classification node, adapter and converter

Completes the six-task surface. class_id -1 is the adapter's 'no result'
sentinel and produces an empty hypothesis list rather than a confident
prediction of class -1."
```

---

### Task 12: GPU integration tests

**Files:**
- Create: `test/integration/test_detection.cpp`, `test_segmentation.cpp`, `test_pose.cpp`, `test_obb.cpp`, `test_classification.cpp`, `test_depth.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: every adapter factory from Tasks 6-11.
- Produces: no new API; six gtest binaries that skip cleanly without a model.

- [ ] **Step 1: Write the depth integration test**

Depth gets the most scrutiny because a wrong scale looks plausible.

`test/integration/test_depth.cpp`
```cpp
// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include <opencv2/opencv.hpp>

#include "ros2_yolos_trt/adapters/depth_adapter.hpp"
#include "ros2_yolos_trt/conversion/depth_converter.hpp"

namespace {
std::string envOr(const char* name, const std::string& fallback) {
  const char* v = std::getenv(name);
  return v ? std::string(v) : fallback;
}
}  // namespace

class DepthIntegration : public ::testing::Test {
protected:
  void SetUp() override {
    model_ = envOr("YOLOS_TRT_DEPTH_MODEL", "");
    image_ = envOr("YOLOS_TRT_TEST_IMAGE", "");
    if (model_.empty() || image_.empty()) {
      GTEST_SKIP() << "set YOLOS_TRT_DEPTH_MODEL and YOLOS_TRT_TEST_IMAGE to run";
    }
  }
  std::string model_;
  std::string image_;
};

TEST_F(DepthIntegration, ProducesMetricDepthAtInputResolution) {
  auto adapter = ros2_yolos_trt::createDepthAdapter();
  ros2_yolos_trt::YolosConfig cfg;
  cfg.model_path = model_;
  cfg.precision = "fp16";
  ASSERT_TRUE(adapter->initialize(cfg)) << "failed to initialize; check the model";

  const cv::Mat frame = cv::imread(image_);
  ASSERT_FALSE(frame.empty()) << "could not read " << image_;

  const auto result = adapter->estimate(frame);
  ASSERT_FALSE(result.depth_metres.empty());

  // Depth must come back at the ORIGINAL image size: this is what catches a
  // broken un-letterbox or resize step.
  EXPECT_EQ(result.depth_metres.cols, frame.cols);
  EXPECT_EQ(result.depth_metres.rows, frame.rows);
  EXPECT_EQ(result.depth_metres.type(), CV_32FC1);

  // Physically plausible metres. A lost metric calibration shows up as
  // near-zero or absurdly large values.
  EXPECT_GT(result.min_metres, 0.0f);
  EXPECT_LT(result.max_metres, 1000.0f);
  EXPECT_GE(result.max_metres, result.min_metres);
}

TEST_F(DepthIntegration, PublishedImageIsThirtyTwoFloatMetres) {
  auto adapter = ros2_yolos_trt::createDepthAdapter();
  ros2_yolos_trt::YolosConfig cfg;
  cfg.model_path = model_;
  ASSERT_TRUE(adapter->initialize(cfg));

  const cv::Mat frame = cv::imread(image_);
  const auto result = adapter->estimate(frame);
  const auto msg =
    ros2_yolos_trt::conversion::toDepthImage(result, std_msgs::msg::Header{});

  EXPECT_EQ(msg.encoding, "32FC1");
  EXPECT_EQ(msg.width, static_cast<uint32_t>(frame.cols));
  EXPECT_EQ(msg.height, static_cast<uint32_t>(frame.rows));

  // Spot-check that the published bytes really are the metric values.
  const auto* values = reinterpret_cast<const float*>(msg.data.data());
  const int centre = (frame.rows / 2) * frame.cols + frame.cols / 2;
  EXPECT_FLOAT_EQ(values[centre],
                  result.depth_metres.at<float>(frame.rows / 2, frame.cols / 2));
}

// Second call must reuse the cached engine rather than rebuilding.
TEST_F(DepthIntegration, SecondInitializeIsFastViaEngineCache) {
  ros2_yolos_trt::YolosConfig cfg;
  cfg.model_path = model_;

  auto first = ros2_yolos_trt::createDepthAdapter();
  ASSERT_TRUE(first->initialize(cfg));
  first->shutdown();

  const auto start = std::chrono::steady_clock::now();
  auto second = ros2_yolos_trt::createDepthAdapter();
  ASSERT_TRUE(second->initialize(cfg));
  const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
    std::chrono::steady_clock::now() - start);

  // A rebuild takes minutes; a cache hit plus warm-up is well under one.
  EXPECT_LT(elapsed.count(), 60);
}
```

- [ ] **Step 2: Write the five remaining integration tests**

Each follows the same shape: read `YOLOS_TRT_<TASK>_MODEL`, `YOLOS_TRT_TEST_LABELS` and `YOLOS_TRT_TEST_IMAGE`, `GTEST_SKIP()` when unset, initialize the adapter, run inference on the image, and assert the results are structurally sane. Concretely:

- `test_detection.cpp` — `createDetectorAdapter()`; assert `initialize` succeeds, `getClassNames()` is non-empty, and every returned box has positive width and height and lies within the image bounds.
- `test_segmentation.cpp` — `createSegmentorAdapter()`; assert each result's `mask` is non-empty, is `CV_8UC1`, and has the same size as the input image.
- `test_pose.cpp` — `createPoseAdapter()`; assert that when any pose is returned its `keypoints` vector is non-empty and every keypoint lies within the image bounds. This is the regression guard for the keypoint plumbing added in Task 9.
- `test_obb.cpp` — `createOBBAdapter()`; assert each result has positive `width`/`height` and an `angle` within `[-pi, pi]`.
- `test_classification.cpp` — `createClassifierAdapter()`; assert `class_id >= 0` and `confidence` within `[0, 1]`.

- [ ] **Step 3: Register the tests in `CMakeLists.txt`**

Inside `if(BUILD_TESTING)`:

```cmake
  # Integration tests need a GPU and model files; they skip themselves when
  # the environment variables are unset, so they are safe to always register.
  macro(add_yolos_integration_test target source)
    ament_add_gtest(${target} ${source})
    target_link_libraries(${target} ${PROJECT_NAME}_components
      "${cpp_typesupport_target}" ${OpenCV_LIBS})
    ament_target_dependencies(${target} rclcpp sensor_msgs std_msgs vision_msgs)
  endmacro()

  add_yolos_integration_test(test_integration_detection      test/integration/test_detection.cpp)
  add_yolos_integration_test(test_integration_segmentation   test/integration/test_segmentation.cpp)
  add_yolos_integration_test(test_integration_pose          test/integration/test_pose.cpp)
  add_yolos_integration_test(test_integration_obb           test/integration/test_obb.cpp)
  add_yolos_integration_test(test_integration_classification test/integration/test_classification.cpp)
  add_yolos_integration_test(test_integration_depth         test/integration/test_depth.cpp)
```

- [ ] **Step 4: Verify they skip without models and pass with one**

Run without models:
```bash
colcon build --packages-select ros2_yolos_trt
colcon test --packages-select ros2_yolos_trt
colcon test-result --verbose
```
Expected: integration tests report SKIPPED, not failed.

Run the depth test with a real model on a GPU machine:
```bash
export YOLOS_TRT_DEPTH_MODEL=/models/yolo26n-depth.onnx
export YOLOS_TRT_TEST_IMAGE=/models/bus.jpg
colcon test --packages-select ros2_yolos_trt --ctest-args -R test_integration_depth
colcon test-result --verbose
```
Expected: 3 cases pass. The first run builds an engine and takes minutes.

- [ ] **Step 5: Commit**

```bash
git add test/integration CMakeLists.txt
git commit -m "test: add GPU integration tests for all six tasks

Each test skips itself when its model environment variables are unset, so
they are safe to register unconditionally and CI stays meaningful without a
GPU.

The depth test is the most detailed because a wrong depth scale looks
plausible: it checks the map returns at the original image resolution, that
values are physically sensible metres, that the published bytes really are
those metres, and that a second initialize hits the engine cache instead of
rebuilding."
```

---

### Task 13: README with an RViz depth recording

**Files:**
- Create: `README.md`, `scripts/record_depth_demo.sh`, `rviz/depth_demo.rviz`
- Create: `assets/depth_rviz.gif` (generated by the script)
- Modify: `CMakeLists.txt` (install `rviz/`)

**Interfaces:**
- Consumes: the depth node and launch file from Task 7.
- Produces: no code API. The script is the reproducible recipe for the asset.

- [ ] **Step 1: Write `rviz/depth_demo.rviz`**

An RViz config showing the source image beside the colorized depth, so the recording is self-explanatory.

```yaml
Panels:
  - Class: rviz_common/Displays
    Name: Displays
    Property Tree Widget:
      Expanded: ~
Visualization Manager:
  Class: ""
  Displays:
    - Class: rviz_default_plugins/Image
      Enabled: true
      Name: Camera
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Best Effort
        Value: /camera/image_raw
    - Class: rviz_default_plugins/Image
      Enabled: true
      Name: Depth (colorized)
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Best Effort
        Value: /yolos_trt_depth/depth/debug_image
  Global Options:
    Background Color: 25; 25; 25
    Fixed Frame: camera_link
    Frame Rate: 30
  Tools:
    - Class: rviz_default_plugins/MoveCamera
  Value: true
Window Geometry:
  Height: 720
  Width: 1280
  Displays:
    collapsed: false
```

- [ ] **Step 2: Write `scripts/record_depth_demo.sh`**

```bash
#!/usr/bin/env bash
# Copyright 2026 YOLOs-TRT Team
# SPDX-License-Identifier: AGPL-3.0
#
# Regenerate assets/depth_rviz.gif — an RViz recording of the depth node.
#
# Requires: a CUDA GPU, a sourced ROS 2 install with this package built,
# rviz2, Xvfb, ffmpeg, and a video or image sequence to play in.
#
# Usage:
#   ./scripts/record_depth_demo.sh <model.onnx|model.trt> <input_video> [seconds]
set -euo pipefail

MODEL="${1:?usage: record_depth_demo.sh <model> <input_video> [seconds]}"
VIDEO="${2:?usage: record_depth_demo.sh <model> <input_video> [seconds]}"
SECONDS_TO_RECORD="${3:-12}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
OUT_DIR="${PKG_DIR}/assets"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"; kill $(jobs -p) 2>/dev/null || true' EXIT

mkdir -p "$OUT_DIR"

echo "== starting virtual display =="
export DISPLAY=:99
Xvfb :99 -screen 0 1280x720x24 &
sleep 2

echo "== publishing frames from $VIDEO =="
# image_publisher ships with ROS 2 and accepts a video file.
ros2 run image_publisher image_publisher_node "$VIDEO" \
  --ros-args -r __ns:=/camera -p publish_rate:=15.0 &
sleep 3

echo "== launching depth node =="
ros2 launch ros2_yolos_trt depth.launch.py \
  model_path:="$MODEL" \
  image_topic:=/camera/image_raw \
  publish_debug_image:=true &
sleep 5

echo "== driving the lifecycle =="
# configure can take minutes on the first run: the engine is built here.
ros2 lifecycle set /yolos_trt_depth configure
ros2 lifecycle set /yolos_trt_depth activate

echo "== waiting for depth output =="
timeout 120 ros2 topic echo --once /yolos_trt_depth/depth/image \
  > /dev/null || { echo "no depth published"; exit 1; }

echo "== starting rviz2 =="
rviz2 -d "${PKG_DIR}/rviz/depth_demo.rviz" &
sleep 8

echo "== recording ${SECONDS_TO_RECORD}s =="
ffmpeg -y -video_size 1280x720 -framerate 15 -f x11grab -i :99 \
  -t "$SECONDS_TO_RECORD" "${WORK_DIR}/capture.mp4"

echo "== converting to gif =="
# Two-pass palette gives a much smaller, cleaner gif than a direct convert.
ffmpeg -y -i "${WORK_DIR}/capture.mp4" \
  -vf "fps=10,scale=960:-1:flags=lanczos,palettegen" "${WORK_DIR}/palette.png"
ffmpeg -y -i "${WORK_DIR}/capture.mp4" -i "${WORK_DIR}/palette.png" \
  -lavfi "fps=10,scale=960:-1:flags=lanczos[x];[x][1:v]paletteuse" \
  "${OUT_DIR}/depth_rviz.gif"

echo "wrote ${OUT_DIR}/depth_rviz.gif"
ls -lh "${OUT_DIR}/depth_rviz.gif"
```

- [ ] **Step 3: Make the script executable and run it to produce the asset**

```bash
chmod +x scripts/record_depth_demo.sh
./scripts/record_depth_demo.sh /models/yolo26n-depth.onnx /models/street.mp4 12
```
Expected: `assets/depth_rviz.gif` exists and is under about 10 MB. If it is larger, lower `fps` or `scale` in the second ffmpeg pass. **Do not commit a GIF produced any other way** — the README states it is an RViz recording of this package, so it must be one.

- [ ] **Step 4: Write `README.md`**

Structure mirrors `ros2_yolos_cpp`, leading with the depth recording since that is what this package adds:

````markdown
# ros2_yolos_trt

ROS 2 wrapper for [YOLOs-CPP-TensorRT](https://github.com/Geekgineer/YOLOs-CPP-TensorRT)
**v3.1.0** — six inference tasks as composable lifecycle nodes, running on
TensorRT.

The TensorRT sibling of [ros2_yolos_cpp](https://github.com/Geekgineer/ros2_yolos_cpp),
which wraps the ONNX Runtime build of the same library.

## Monocular metric depth in RViz

![Depth estimation running in RViz](assets/depth_rviz.gif)

The `yolos_trt_depth` node publishing `32FC1` metres from a single RGB camera,
using the YOLO26-depth models introduced in v3.1.0. Regenerate this recording
with `scripts/record_depth_demo.sh`.

Because depth is published in metres as `32FC1` — the ROS convention — it feeds
existing tooling unchanged:

```bash
ros2 run depth_image_proc point_cloud_xyz \
  --ros-args -r image_rect:=/yolos_trt_depth/depth/image
```

## Tasks

| Task | Executable | Publishes |
|------|-----------|-----------|
| Detection | `yolos_trt_detector_node` | `vision_msgs/Detection2DArray` |
| Segmentation | `yolos_trt_segmentor_node` | `Detection2DArray` + `mono8` label mask |
| Pose | `yolos_trt_pose_node` | `Detection2DArray` + `KeyPoint2DArray` |
| OBB | `yolos_trt_obb_node` | `OBBDetection2DArray` |
| Classification | `yolos_trt_classifier_node` | `vision_msgs/Classification` |
| Metric depth | `yolos_trt_depth_node` | `sensor_msgs/Image` `32FC1`, metres |

## Requirements

- ROS 2 (Humble or newer)
- NVIDIA GPU, compute capability >= 7.5
- CUDA Toolkit >= 12.0
- **TensorRT 10.x — not 11.x.** `libnvinfer-dev` now resolves to 11.x, which
  removed the tensor API this library uses. Pin it:
  ```bash
  sudo tee /etc/apt/preferences.d/tensorrt-pin > /dev/null <<'PIN'
  Package: libnvinfer* libnvonnxparsers* tensorrt*
  Pin: version 10.*
  Pin-Priority: 1001
  PIN
  ```
- OpenCV >= 4.5

## Build

```bash
cd ~/ros2_ws/src
git clone https://github.com/Geekgineer/ros2_yolos_trt.git
cd ~/ros2_ws
colcon build --packages-select ros2_yolos_trt
```

The upstream library is fetched at the pinned `v3.1.0` tag during configure;
there is no separate install step. CUDA architectures default to
`75;80;86;89;90` (`72;87` on Jetson) and are **not** auto-detected — narrow them
to your own GPU to cut compile time:

```bash
colcon build --packages-select ros2_yolos_trt \
  --cmake-args -DCMAKE_CUDA_ARCHITECTURES=89
```

## Models

`model_path` accepts either form:

- **`.onnx`** — an engine is built on first `configure` and cached beside the
  model. Convenient, and the right choice when one model ships to a fleet of
  mixed GPUs, because a TensorRT engine is portable across neither GPU
  architecture nor TensorRT version.
- **`.trt` / `.engine`** — loaded directly, for fast deterministic startup.

The cache key covers compute capability, TensorRT version, precision, DLA core,
and the model's size and mtime, so a changed model or a different machine
rebuilds rather than silently running the wrong network.

Export a depth model at its training resolution:

```bash
python -c "from ultralytics import YOLO; \
  YOLO('yolo26n-depth.pt').export(format='onnx', imgsz=768, opset=12, simplify=True)"
```

## Run

```bash
ros2 launch ros2_yolos_trt depth.launch.py \
  model_path:=/models/yolo26n-depth.onnx \
  image_topic:=/camera/image_raw

# nodes are managed; the first configure builds the engine
ros2 lifecycle set /yolos_trt_depth configure
ros2 lifecycle set /yolos_trt_depth activate
```

## Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `model_path` | *(required)* | `.onnx`, `.trt` or `.engine` |
| `labels_path` | `""` | not used by depth |
| `precision` | `fp16` | `fp32` or `fp16`; used only when building |
| `engine_cache_dir` | `""` | default: beside the model |
| `dla_core` | `-1` | `-1` = GPU, `0`/`1` = DLA on Jetson |
| `conf_threshold` | `0.4` | |
| `nms_threshold` | `0.45` | |
| `max_detections` | `300` | OBB only |
| `depth_min` / `depth_max` | `0.0` | debug colorization clamp, metres; `0` = auto |
| `publish_debug_image` | `false` | |
| `publish_timing` | `false` | `[inference_ms, total_ms]` |

There is deliberately no `use_gpu` (TensorRT is always GPU) and no
`yolo_version` (the library detects it from output tensor shapes).

## Testing

```bash
colcon test --packages-select ros2_yolos_trt          # unit tests, no GPU
```

Integration tests skip unless pointed at models:

```bash
export YOLOS_TRT_DEPTH_MODEL=/models/yolo26n-depth.onnx
export YOLOS_TRT_TEST_IMAGE=/models/bus.jpg
colcon test --packages-select ros2_yolos_trt --ctest-args -R test_integration_depth
```

## Troubleshooting

**Results look plausible but are wrong.** Check the CUDA architecture the build
used:

```bash
grep CMAKE_CUDA_ARCHITECTURES build/ros2_yolos_trt/CMakeCache.txt
```

`52` means the CUDA preprocessing kernel was built for sm_52 and returns
incorrect pixels on a modern GPU. Reconfigure from a clean build directory.

**Configure hangs for minutes.** Expected on first run with a `.onnx`: the
engine is being built. Subsequent runs hit the cache.

**`configure` returns FAILURE.** The node logs why — a missing model, an engine
built for a different GPU, or a TensorRT version mismatch.

## License

AGPL-3.0, matching the upstream library.
````

- [ ] **Step 5: Install the rviz config**

In `CMakeLists.txt`, next to the launch and config installs:

```cmake
install(DIRECTORY rviz/ DESTINATION share/${PROJECT_NAME}/rviz)
```

- [ ] **Step 6: Verify the README renders and the asset is referenced correctly**

```bash
test -f assets/depth_rviz.gif && echo "asset present"
grep -q 'assets/depth_rviz.gif' README.md && echo "README references it"
du -h assets/depth_rviz.gif
```
Expected: both checks print, and the GIF is a reasonable size for a repository.

- [ ] **Step 7: Commit**

```bash
git add README.md scripts/record_depth_demo.sh rviz/depth_demo.rviz assets/depth_rviz.gif CMakeLists.txt
git commit -m "docs: add README with an RViz depth recording

Leads with the depth capability, since that is what this package adds over
its ONNX sibling. The recording is produced by scripts/record_depth_demo.sh
so the asset can be regenerated rather than being an unreproducible one-off.

Documents the two failure modes that cost real debugging time upstream: an
sm_52 build returning wrong pixels while appearing to work, and apt now
resolving TensorRT to 11.x."
```

---

### Task 14: CI

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: the package as built by Tasks 1-13.
- Produces: no code API.

- [ ] **Step 1: Write the workflow**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  # Unit tests only. Building the component library needs CUDA and TensorRT,
  # which a standard runner does not have, so this job checks everything that
  # can be checked without a GPU: linting and the pure conversion logic.
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5

      - name: Install clang-format and cpplint
        run: |
          sudo apt-get update
          sudo apt-get install -y --no-install-recommends clang-format
          pipx install cpplint || pip install --user cpplint

      - name: Check C++ formatting
        run: |
          FILES=$(find include src test -name '*.hpp' -o -name '*.cpp' | sort)
          echo "$FILES" | while read -r f; do
            clang-format --dry-run --Werror "$f" || \
              echo "::warning file=${f}::formatting differs"
          done

      - name: Check license headers
        run: |
          MISSING=0
          while IFS= read -r f; do
            if ! head -3 "$f" | grep -q "SPDX-License-Identifier: AGPL-3.0"; then
              echo "::error file=${f}::missing SPDX license header"
              MISSING=$((MISSING + 1))
            fi
          done < <(find include src -name '*.hpp' -o -name '*.cpp')
          exit $MISSING

      - name: Check for a moving upstream pin
        run: |
          # A branch here would make builds irreproducible.
          if grep -qE 'YOLOS_TRT_VERSION "(main|master|HEAD)"' CMakeLists.txt; then
            echo "::error::upstream must be pinned to a release tag, not a branch"
            exit 1
          fi
          grep -E 'YOLOS_TRT_VERSION "v[0-9]+\.[0-9]+\.[0-9]+"' CMakeLists.txt

  build:
    runs-on: ubuntu-latest
    container:
      image: ros:jazzy-ros-base
    steps:
      - uses: actions/checkout@v5
        with:
          path: src/ros2_yolos_trt

      - name: Install CUDA and TensorRT 10.x
        run: |
          apt-get update
          apt-get install -y --no-install-recommends wget ca-certificates gnupg
          wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
          dpkg -i cuda-keyring_1.1-1_all.deb
          apt-get update

          # Pin the whole family: unpinned, libnvinfer-dev now resolves to
          # TensorRT 11.x built against CUDA 13, which does not compile
          # against the TRT 10 tensor API this package uses.
          tee /etc/apt/preferences.d/tensorrt-pin > /dev/null <<'PIN'
          Package: libnvinfer* libnvonnxparsers* tensorrt*
          Pin: version 10.*
          Pin-Priority: 1001
          PIN
          apt-get update
          apt-get install -y --no-install-recommends \
            cuda-nvcc-12-6 cuda-cudart-dev-12-6 \
            libnvinfer-dev libnvinfer-headers-dev libnvonnxparsers-dev
          echo "/usr/local/cuda-12.6/bin" >> "$GITHUB_PATH"

      - name: Install ROS dependencies
        run: |
          apt-get install -y --no-install-recommends \
            ros-jazzy-vision-msgs ros-jazzy-cv-bridge ros-jazzy-image-transport \
            libopencv-dev python3-colcon-common-extensions

      - name: Build
        shell: bash
        run: |
          source /opt/ros/jazzy/setup.bash
          export PATH=/usr/local/cuda-12.6/bin:$PATH
          colcon build --packages-select ros2_yolos_trt \
            --cmake-args -DCMAKE_BUILD_TYPE=Release

      - name: Verify the build did not fall back to sm_52
        shell: bash
        run: |
          ARCH=$(grep '^CMAKE_CUDA_ARCHITECTURES' \
            build/ros2_yolos_trt/CMakeCache.txt | cut -d= -f2)
          echo "CUDA architectures: $ARCH"
          if [ "$ARCH" = "52" ]; then
            echo "::error::built for sm_52 — preprocessing would return wrong pixels"
            exit 1
          fi

      - name: Run unit tests
        shell: bash
        run: |
          source /opt/ros/jazzy/setup.bash
          colcon test --packages-select ros2_yolos_trt \
            --ctest-args -E test_integration
          colcon test-result --verbose
```

- [ ] **Step 2: Verify the workflow is valid YAML**

```bash
python3 -c "import yaml; d=yaml.safe_load(open('.github/workflows/ci.yml')); print('jobs:', list(d['jobs']))"
```
Expected: `jobs: ['lint', 'build']`

- [ ] **Step 3: Commit and push**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: build against pinned TensorRT 10.x and run unit tests

Integration tests are excluded because they need a GPU; the unit tests
cover the conversion layer and the engine cache key, which is where the
logic actually lives.

Two guards encode failures that cost real debugging time: the build fails
if CMAKE_CUDA_ARCHITECTURES ends up as 52, since an sm_52 build returns
wrong pixels while appearing to work, and lint fails if the upstream
dependency is ever pointed at a branch instead of a release tag."
git push -u origin main
```

---

## Self-Review

**1. Spec coverage.** Every spec section maps to a task: purpose and success criteria (Tasks 1, 6-11); four-layer architecture (Tasks 3, 6-11); engine cache including its key contents and `on_configure` placement (Task 4); the six nodes and topics table (Tasks 6-11); depth as 32FC1 metres with colorized output confined to debug (Task 7); the configuration table (Task 1 `default_params.yaml`, Task 5 declaration); both build constraints (Task 1, re-verified in Task 14); the pose keypoints deviation (Task 9); unit and integration test split (Tasks 6-12); README with the RViz depth recording (Task 13); CI (Task 14). Out-of-scope items (PointCloud2, INT8 calibration, batching, tracking) have no tasks, as intended.

**2. Placeholders.** None remain. Every code step contains compilable content. Tasks 8-11 describe node files by enumerating their exact differences from `detector_node` with the full differing code given, rather than deferring work.

**3. Type consistency.** Checked across tasks: `YolosConfig` field names used in Tasks 4-11 match Task 3; `resolveEngine(model_path, precision, dla_core, cache_dir, error_out)` is called with that argument order everywhere; converter names are distinct per task (`toDetection2DArray`, `segmentationToDetection2DArray`, `poseToDetection2DArray`, `toKeyPoint2DArrays`, `toOBBDetection2DArray`, `toClassification`, `toDepthImage`, `toCombinedMaskImage`) with no collisions; `publishTimingSample` is the base-class method name used by every node, and `publishTimingEnabled()` is the accessor, deliberately not overloaded with it; component plugin strings match the class names they register.

**Three upstream signatures to confirm before building**, called out at their point of use rather than assumed: `yolos::seg::Segmentation` field names and `drawSegmentations` arity (Task 8), `yolos::KeyPoint` field names (Task 9), and `yolos::obb::OBBResult` box field names (Task 10). Each note says which upstream header to read.
