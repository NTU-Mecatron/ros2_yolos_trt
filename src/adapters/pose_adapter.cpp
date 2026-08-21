// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_trt/adapters/pose_adapter.hpp"

// The only place upstream headers appear for this task.
#include "yolos/tasks/pose.hpp"

#include <iostream>

#include "ros2_yolos_trt/engine/engine_cache.hpp"

namespace ros2_yolos_trt {

class PoseAdapter::Impl {
public:
  std::unique_ptr<yolos::pose::YOLOPoseDetector> pose;
  std::vector<std::string> class_names;
  bool initialized{false};
};

PoseAdapter::PoseAdapter() : impl_(std::make_unique<Impl>()) {}
PoseAdapter::~PoseAdapter() = default;
PoseAdapter::PoseAdapter(PoseAdapter&&) noexcept = default;
PoseAdapter& PoseAdapter::operator=(PoseAdapter&&) noexcept = default;

bool PoseAdapter::initialize(const YolosConfig& config) {
  try {
    std::string error;
    const auto engine = engine::resolveEngine(
      config.model_path, config.precision, config.dla_core,
      config.engine_cache_dir, error);
    if (engine.empty()) {
      std::cerr << "[PoseAdapter] " << error << std::endl;
      return false;
    }

    impl_->pose = std::make_unique<yolos::pose::YOLOPoseDetector>(
      engine.string(), config.labels_path, config.dla_core);
    impl_->class_names = impl_->pose->getClassNames();
    impl_->initialized = true;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[PoseAdapter] initialization error: " << e.what() << std::endl;
    impl_->initialized = false;
    return false;
  }
}

bool PoseAdapter::isInitialized() const noexcept {
  return impl_ && impl_->initialized;
}

void PoseAdapter::shutdown() {
  if (impl_) {
    impl_->pose.reset();
    impl_->initialized = false;
  }
}

const std::vector<std::string>& PoseAdapter::getClassNames() const {
  static const std::vector<std::string> empty;
  return impl_ ? impl_->class_names : empty;
}

std::vector<PoseResult> PoseAdapter::detect(
  const cv::Mat& image, float conf_threshold, float nms_threshold) {
  std::vector<PoseResult> results;
  if (!isInitialized() || image.empty()) return results;

  try {
    const auto native =
      impl_->pose->detect(image, conf_threshold, nms_threshold);
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
      // Upstream KeyPoint carries only x, y and confidence. The keypoint
      // index is its position in the vector, which is the COCO ordering for
      // the released pose models.
      r.keypoints.reserve(p.keypoints.size());
      int kpt_id = 0;
      for (const auto& kp : p.keypoints) {
        KeyPoint2D out;
        out.x = kp.x;
        out.y = kp.y;
        out.confidence = kp.confidence;
        out.id = kpt_id++;
        r.keypoints.push_back(out);
      }
      results.push_back(std::move(r));
    }
  } catch (const std::exception& e) {
    std::cerr << "[PoseAdapter] pose error: " << e.what() << std::endl;
  }
  return results;
}

void PoseAdapter::drawPoses(
  cv::Mat& image, const std::vector<PoseResult>& poses, int kpt_radius,
  float kpt_threshold) {
  if (!isInitialized() || image.empty()) return;

  std::vector<yolos::pose::PoseResult> native;
  native.reserve(poses.size());
  for (const auto& pose : poses) {
    std::vector<yolos::KeyPoint> kpts;
    kpts.reserve(pose.keypoints.size());
    for (const auto& kp : pose.keypoints) {
      yolos::KeyPoint out;
      out.x = kp.x;
      out.y = kp.y;
      out.confidence = kp.confidence;
      kpts.push_back(out);
    }
    yolos::BoundingBox box(pose.bbox.x, pose.bbox.y, pose.bbox.width,
                           pose.bbox.height);
    native.emplace_back(box, pose.confidence, pose.class_id, kpts);
  }
  impl_->pose->drawPoses(image, native, kpt_radius, kpt_threshold);
}

std::unique_ptr<IPoseAdapter> createPoseAdapter() {
  return std::make_unique<PoseAdapter>();
}

}  // namespace ros2_yolos_trt
