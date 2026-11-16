// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_trt/adapters/segmentor_adapter.hpp"

// The only place upstream headers appear for this task.
#include "yolos/tasks/segmentation.hpp"

#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <iostream>

#include "ros2_yolos_trt/engine/engine_cache.hpp"
#include "ros2_yolos_trt/labels/label_map.hpp"

namespace fs = std::filesystem;

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

    // Labels files may carry a trailing `#rrggbb` per line. Upstream does not
    // know that format, so hand it a names-only copy and keep our own names.
    LabelMap labels;
    std::string labels_for_upstream = config.labels_path;
    fs::path stripped;
    if (!config.labels_path.empty()) {
      if (!loadLabelMap(config.labels_path, false, labels, error)) {
        std::cerr << "[SegmentorAdapter] " << error << std::endl;
        return false;
      }
      stripped = fs::temp_directory_path() /
                 ("yolos_labels_" + std::to_string(::getpid()) + ".txt");
      {
        std::ofstream f(stripped);
        for (const auto& n : labels.names) f << n << '\n';
        if (!f) {
          std::cerr << "[SegmentorAdapter] cannot write " << stripped << std::endl;
          return false;
        }
      }
      labels_for_upstream = stripped.string();
    }

    impl_->segmentor = std::make_unique<yolos::seg::YOLOSegDetector>(
      engine.string(), labels_for_upstream, config.dla_core);
    if (!stripped.empty()) {
      std::error_code ec;
      fs::remove(stripped, ec);  // upstream reads labels in its constructor
    }
    impl_->class_names = labels.names.empty() ? impl_->segmentor->getClassNames()
                                              : labels.names;
    impl_->initialized = true;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[SegmentorAdapter] initialization error: " << e.what() << std::endl;
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
    const auto native =
      impl_->segmentor->segment(image, conf_threshold, nms_threshold);
    results.reserve(native.size());
    for (const auto& seg : native) {
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
    std::cerr << "[SegmentorAdapter] segmentation error: " << e.what()
              << std::endl;
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
    yolos::BoundingBox box(seg.bbox.x, seg.bbox.y, seg.bbox.width,
                           seg.bbox.height);
    native.emplace_back(box, seg.confidence, seg.class_id, seg.mask);
  }
  impl_->segmentor->drawSegmentations(image, native, mask_alpha);
}

std::unique_ptr<ISegmentorAdapter> createSegmentorAdapter() {
  return std::make_unique<SegmentorAdapter>();
}

}  // namespace ros2_yolos_trt
