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
