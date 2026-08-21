// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_trt/adapters/classifier_adapter.hpp"

// The only place upstream headers appear for this task.
#include "yolos/tasks/classification.hpp"

#include <iostream>

#include "ros2_yolos_trt/engine/engine_cache.hpp"

namespace ros2_yolos_trt {

class ClassifierAdapter::Impl {
public:
  std::unique_ptr<yolos::cls::YOLOClassifier> classifier;
  std::vector<std::string> class_names;
  bool initialized{false};
};

ClassifierAdapter::ClassifierAdapter() : impl_(std::make_unique<Impl>()) {}
ClassifierAdapter::~ClassifierAdapter() = default;
ClassifierAdapter::ClassifierAdapter(ClassifierAdapter&&) noexcept = default;
ClassifierAdapter& ClassifierAdapter::operator=(ClassifierAdapter&&) noexcept = default;

bool ClassifierAdapter::initialize(const YolosConfig& config) {
  try {
    std::string error;
    const auto engine = engine::resolveEngine(
      config.model_path, config.precision, config.dla_core,
      config.engine_cache_dir, error);
    if (engine.empty()) {
      std::cerr << "[ClassifierAdapter] " << error << std::endl;
      return false;
    }

    impl_->classifier = std::make_unique<yolos::cls::YOLOClassifier>(
      engine.string(), config.labels_path, cv::Size(224, 224),
      config.dla_core);
    impl_->class_names = impl_->classifier->getClassNames();
    impl_->initialized = true;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[ClassifierAdapter] initialization error: " << e.what() << std::endl;
    impl_->initialized = false;
    return false;
  }
}

bool ClassifierAdapter::isInitialized() const noexcept {
  return impl_ && impl_->initialized;
}

void ClassifierAdapter::shutdown() {
  if (impl_) {
    impl_->classifier.reset();
    impl_->initialized = false;
  }
}

const std::vector<std::string>& ClassifierAdapter::getClassNames() const {
  static const std::vector<std::string> empty;
  return impl_ ? impl_->class_names : empty;
}

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

void ClassifierAdapter::drawResult(cv::Mat& image,
                                   const ClassificationResult& result) {
  if (!isInitialized() || image.empty() || result.class_id < 0) return;
  yolos::cls::ClassificationResult native(result.class_id, result.confidence,
                                          result.class_name);
  impl_->classifier->drawResult(image, native);
}

std::unique_ptr<IClassifierAdapter> createClassifierAdapter() {
  return std::make_unique<ClassifierAdapter>();
}

}  // namespace ros2_yolos_trt
