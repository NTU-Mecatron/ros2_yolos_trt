// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_trt/adapters/obb_adapter.hpp"

// The only place upstream headers appear for this task.
#include "yolos/tasks/obb.hpp"

#include <iostream>

#include "ros2_yolos_trt/engine/engine_cache.hpp"

namespace ros2_yolos_trt {

class OBBAdapter::Impl {
public:
  std::unique_ptr<yolos::obb::YOLOOBBDetector> obb;
  std::vector<std::string> class_names;
  bool initialized{false};
};

OBBAdapter::OBBAdapter() : impl_(std::make_unique<Impl>()) {}
OBBAdapter::~OBBAdapter() = default;
OBBAdapter::OBBAdapter(OBBAdapter&&) noexcept = default;
OBBAdapter& OBBAdapter::operator=(OBBAdapter&&) noexcept = default;

bool OBBAdapter::initialize(const YolosConfig& config) {
  try {
    std::string error;
    const auto engine = engine::resolveEngine(
      config.model_path, config.precision, config.dla_core,
      config.engine_cache_dir, error);
    if (engine.empty()) {
      std::cerr << "[OBBAdapter] " << error << std::endl;
      return false;
    }

    impl_->obb = std::make_unique<yolos::obb::YOLOOBBDetector>(
      engine.string(), config.labels_path, config.dla_core);
    impl_->class_names = impl_->obb->getClassNames();
    impl_->initialized = true;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[OBBAdapter] initialization error: " << e.what() << std::endl;
    impl_->initialized = false;
    return false;
  }
}

bool OBBAdapter::isInitialized() const noexcept {
  return impl_ && impl_->initialized;
}

void OBBAdapter::shutdown() {
  if (impl_) {
    impl_->obb.reset();
    impl_->initialized = false;
  }
}

const std::vector<std::string>& OBBAdapter::getClassNames() const {
  static const std::vector<std::string> empty;
  return impl_ ? impl_->class_names : empty;
}

std::vector<OBBResult> OBBAdapter::detect(
  const cv::Mat& image, float conf_threshold, float nms_threshold,
  int max_det) {
  std::vector<OBBResult> results;
  if (!isInitialized() || image.empty()) return results;

  try {
    const auto native =
      impl_->obb->detect(image, conf_threshold, nms_threshold, max_det);
    results.reserve(native.size());
    for (const auto& o : native) {
      OBBResult r;
      // Upstream OrientedBoundingBox stores the centre in x/y.
      r.bbox.center_x = o.box.x;
      r.bbox.center_y = o.box.y;
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
  } catch (const std::exception& e) {
    std::cerr << "[OBBAdapter] obb error: " << e.what() << std::endl;
  }
  return results;
}

void OBBAdapter::drawDetections(
  cv::Mat& image, const std::vector<OBBResult>& detections) {
  if (!isInitialized() || image.empty()) return;

  std::vector<yolos::obb::OBBResult> native;
  native.reserve(detections.size());
  for (const auto& det : detections) {
    yolos::OrientedBoundingBox box(det.bbox.center_x, det.bbox.center_y,
                                   det.bbox.width, det.bbox.height,
                                   det.bbox.angle);
    native.emplace_back(box, det.confidence, det.class_id);
  }
  impl_->obb->drawDetections(image, native);
}

std::unique_ptr<IOBBAdapter> createOBBAdapter() {
  return std::make_unique<OBBAdapter>();
}

}  // namespace ros2_yolos_trt
