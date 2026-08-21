// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ROS2_YOLOS_TRT__ADAPTERS__DETECTOR_ADAPTER_HPP_
#define ROS2_YOLOS_TRT__ADAPTERS__DETECTOR_ADAPTER_HPP_

#include <memory>
#include <string>
#include <vector>

#include "ros2_yolos_trt/adapters/yolos_adapter_base.hpp"

namespace ros2_yolos_trt {

/// @brief Detector adapter wrapping yolos::det::YOLODetector.
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
