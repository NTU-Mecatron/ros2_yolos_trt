// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ROS2_YOLOS_TRT__ADAPTERS__CLASSIFIER_ADAPTER_HPP_
#define ROS2_YOLOS_TRT__ADAPTERS__CLASSIFIER_ADAPTER_HPP_

#include <memory>
#include <string>
#include <vector>

#include "ros2_yolos_trt/adapters/yolos_adapter_base.hpp"

namespace ros2_yolos_trt {

/// @brief Classifier adapter wrapping yolos::cls::YOLOClassifier.
class ROS2_YOLOS_TRT_PUBLIC ClassifierAdapter : public IClassifierAdapter {
public:
  ClassifierAdapter();
  ~ClassifierAdapter() override;

  ClassifierAdapter(const ClassifierAdapter&) = delete;
  ClassifierAdapter& operator=(const ClassifierAdapter&) = delete;
  ClassifierAdapter(ClassifierAdapter&&) noexcept;
  ClassifierAdapter& operator=(ClassifierAdapter&&) noexcept;

  bool initialize(const YolosConfig& config) override;
  [[nodiscard]] bool isInitialized() const noexcept override;
  void shutdown() override;
  [[nodiscard]] const std::vector<std::string>& getClassNames() const override;

  ClassificationResult classify(const cv::Mat& image) override;
  void drawResult(cv::Mat& image, const ClassificationResult& result) override;

private:
  class Impl;  // TensorRT and yolos:: types stay entirely in the .cpp
  std::unique_ptr<Impl> impl_;
};

}  // namespace ros2_yolos_trt

#endif  // ROS2_YOLOS_TRT__ADAPTERS__CLASSIFIER_ADAPTER_HPP_
