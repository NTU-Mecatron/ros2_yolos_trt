// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_trt/adapters/depth_adapter.hpp"

// The only place upstream headers appear for this task.
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
  return yolos::depth::colorizeDepth(result.depth_metres, min_metres,
                                     max_metres);
}

std::unique_ptr<IDepthAdapter> createDepthAdapter() {
  return std::make_unique<DepthAdapter>();
}

}  // namespace ros2_yolos_trt
