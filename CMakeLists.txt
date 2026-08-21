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

# TensorRT 10.x defines NV_TENSORRT_MAJOR indirectly, as
#   #define TRT_MAJOR_ENTERPRISE 10
#   #define NV_TENSORRT_MAJOR TRT_MAJOR_ENTERPRISE
# so matching NV_TENSORRT_MAJOR against a digit finds nothing. Try the direct
# literal first (older layouts and some builds), then the indirection.
file(READ "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" _trt_version_header)
set(TENSORRT_MAJOR "")
if("${_trt_version_header}" MATCHES "define NV_TENSORRT_MAJOR ([0-9]+)")
  set(TENSORRT_MAJOR "${CMAKE_MATCH_1}")
elseif("${_trt_version_header}" MATCHES "define TRT_MAJOR_ENTERPRISE ([0-9]+)")
  set(TENSORRT_MAJOR "${CMAKE_MATCH_1}")
endif()
if(TENSORRT_MAJOR STREQUAL "")
  message(FATAL_ERROR
    "Could not determine the TensorRT major version from "
    "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h. Refusing to build blind: this "
    "package requires 10.x and 11.x removes the API it uses.")
endif()
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
  src/engine/engine_cache.cpp
  src/nodes/yolos_lifecycle_node_base.cpp
  src/adapters/detector_adapter.cpp
  src/adapters/segmentor_adapter.cpp
  src/adapters/pose_adapter.cpp
  src/adapters/obb_adapter.cpp
  src/adapters/classifier_adapter.cpp
  src/adapters/depth_adapter.cpp
  src/conversion/detection_converter.cpp
  src/conversion/segmentation_converter.cpp
  src/conversion/pose_converter.cpp
  src/conversion/obb_converter.cpp
  src/conversion/classification_converter.cpp
  src/conversion/depth_converter.cpp
  src/nodes/detector_node.cpp
  src/nodes/segmentor_node.cpp
  src/nodes/pose_node.cpp
  src/nodes/obb_node.cpp
  src/nodes/classifier_node.cpp
  src/nodes/depth_node.cpp
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
# Components
# -----------------------------------------------------------------------------
rclcpp_components_register_node(${PROJECT_NAME}_components
  PLUGIN "ros2_yolos_trt::YolosDetectorNode"   EXECUTABLE yolos_trt_detector_node)
rclcpp_components_register_node(${PROJECT_NAME}_components
  PLUGIN "ros2_yolos_trt::YolosSegmentorNode"  EXECUTABLE yolos_trt_segmentor_node)
rclcpp_components_register_node(${PROJECT_NAME}_components
  PLUGIN "ros2_yolos_trt::YolosPoseNode"       EXECUTABLE yolos_trt_pose_node)
rclcpp_components_register_node(${PROJECT_NAME}_components
  PLUGIN "ros2_yolos_trt::YolosOBBNode"        EXECUTABLE yolos_trt_obb_node)
rclcpp_components_register_node(${PROJECT_NAME}_components
  PLUGIN "ros2_yolos_trt::YolosClassifierNode" EXECUTABLE yolos_trt_classifier_node)
rclcpp_components_register_node(${PROJECT_NAME}_components
  PLUGIN "ros2_yolos_trt::YolosDepthNode"      EXECUTABLE yolos_trt_depth_node)

# -----------------------------------------------------------------------------
# Install
# -----------------------------------------------------------------------------
install(TARGETS ${PROJECT_NAME}_components
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY rviz/  DESTINATION share/${PROJECT_NAME}/rviz)

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_build_smoke test/unit/test_build_smoke.cpp)
  target_link_libraries(test_build_smoke ${PROJECT_NAME}_components)

  ament_add_gtest(test_messages test/unit/test_messages.cpp)
  target_link_libraries(test_messages ${PROJECT_NAME}_components "${cpp_typesupport_target}")

  ament_add_gtest(test_dtos test/unit/test_dtos.cpp)
  target_link_libraries(test_dtos ${PROJECT_NAME}_components)

  ament_add_gtest(test_engine_cache test/unit/test_engine_cache.cpp)
  target_link_libraries(test_engine_cache ${PROJECT_NAME}_components)

  ament_add_gtest(test_lifecycle_node_base test/unit/test_lifecycle_node_base.cpp)
  target_link_libraries(test_lifecycle_node_base ${PROJECT_NAME}_components)
  ament_target_dependencies(test_lifecycle_node_base rclcpp rclcpp_lifecycle lifecycle_msgs)

  ament_add_gtest(test_detection_converter test/unit/test_detection_converter.cpp)
  target_link_libraries(test_detection_converter ${PROJECT_NAME}_components)
  ament_target_dependencies(test_detection_converter vision_msgs std_msgs)

  ament_add_gtest(test_segmentation_converter test/unit/test_segmentation_converter.cpp)
  target_link_libraries(test_segmentation_converter ${PROJECT_NAME}_components)
  ament_target_dependencies(test_segmentation_converter vision_msgs sensor_msgs std_msgs)

  ament_add_gtest(test_pose_converter test/unit/test_pose_converter.cpp)
  target_link_libraries(test_pose_converter ${PROJECT_NAME}_components "${cpp_typesupport_target}")
  ament_target_dependencies(test_pose_converter vision_msgs std_msgs geometry_msgs)

  ament_add_gtest(test_obb_converter test/unit/test_obb_converter.cpp)
  target_link_libraries(test_obb_converter ${PROJECT_NAME}_components "${cpp_typesupport_target}")
  ament_target_dependencies(test_obb_converter std_msgs geometry_msgs)

  ament_add_gtest(test_classification_converter test/unit/test_classification_converter.cpp)
  target_link_libraries(test_classification_converter ${PROJECT_NAME}_components)
  ament_target_dependencies(test_classification_converter vision_msgs std_msgs)

  ament_add_gtest(test_depth_converter test/unit/test_depth_converter.cpp)
  target_link_libraries(test_depth_converter ${PROJECT_NAME}_components)
  ament_target_dependencies(test_depth_converter sensor_msgs std_msgs)

  # Integration tests need a GPU and model files; they skip themselves when
  # the environment variables are unset, so they are safe to always register.
  # A generous timeout: the first run with a .onnx builds a TensorRT engine,
  # which takes minutes. ctest's 60 s default kills it mid-build.
  macro(add_yolos_integration_test target source)
    ament_add_gtest(${target} ${source} TIMEOUT 900)
    target_link_libraries(${target} ${PROJECT_NAME}_components
      "${cpp_typesupport_target}" ${OpenCV_LIBS})
    ament_target_dependencies(${target} rclcpp sensor_msgs std_msgs vision_msgs)
  endmacro()

  add_yolos_integration_test(test_integration_detection      test/integration/test_detection.cpp)
  add_yolos_integration_test(test_integration_segmentation   test/integration/test_segmentation.cpp)
  add_yolos_integration_test(test_integration_pose           test/integration/test_pose.cpp)
  add_yolos_integration_test(test_integration_obb            test/integration/test_obb.cpp)
  add_yolos_integration_test(test_integration_classification test/integration/test_classification.cpp)
  add_yolos_integration_test(test_integration_depth          test/integration/test_depth.cpp)
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
