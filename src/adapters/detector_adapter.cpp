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
    std::cerr << "[DetectorAdapter] initialization error: " << e.what() << std::endl;
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
    const auto native =
      impl_->detector->detect(image, conf_threshold, nms_threshold);
    results.reserve(native.size());
    for (const auto& det : native) {
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
