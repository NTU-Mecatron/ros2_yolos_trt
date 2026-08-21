// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ROS2_YOLOS_TRT__ADAPTERS__DEPTH_ADAPTER_HPP_
#define ROS2_YOLOS_TRT__ADAPTERS__DEPTH_ADAPTER_HPP_

#include <memory>
#include <string>
#include <vector>

#include "ros2_yolos_trt/adapters/yolos_adapter_base.hpp"

namespace ros2_yolos_trt {

/// @brief Depth adapter wrapping yolos::depth::YOLODepthEstimator.
class ROS2_YOLOS_TRT_PUBLIC DepthAdapter : public IDepthAdapter {
public:
  DepthAdapter();
  ~DepthAdapter() override;

  DepthAdapter(const DepthAdapter&) = delete;
  DepthAdapter& operator=(const DepthAdapter&) = delete;
  DepthAdapter(DepthAdapter&&) noexcept;
  DepthAdapter& operator=(DepthAdapter&&) noexcept;

  bool initialize(const YolosConfig& config) override;
  [[nodiscard]] bool isInitialized() const noexcept override;
  void shutdown() override;
  [[nodiscard]] const std::vector<std::string>& getClassNames() const override;

  DepthResult estimate(const cv::Mat& image) override;
  cv::Mat colorize(const DepthResult& result, float min_metres,
                   float max_metres) override;

private:
  class Impl;  // TensorRT and yolos:: types stay entirely in the .cpp
  std::unique_ptr<Impl> impl_;
};

}  // namespace ros2_yolos_trt

#endif  // ROS2_YOLOS_TRT__ADAPTERS__DEPTH_ADAPTER_HPP_
