// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ROS2_YOLOS_TRT__CONVERSION__DETECTION_CONVERTER_HPP_
#define ROS2_YOLOS_TRT__CONVERSION__DETECTION_CONVERTER_HPP_

#include <vector>

#include <std_msgs/msg/header.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "ros2_yolos_trt/adapters/yolos_adapter_base.hpp"
#include "ros2_yolos_trt/visibility_control.hpp"

namespace ros2_yolos_trt {
namespace conversion {

/// @brief Convert detection DTOs to a vision_msgs Detection2DArray.
ROS2_YOLOS_TRT_PUBLIC vision_msgs::msg::Detection2DArray toDetection2DArray(
  const std::vector<DetectionResult>& detections,
  const std_msgs::msg::Header& header);

}  // namespace conversion
}  // namespace ros2_yolos_trt

#endif  // ROS2_YOLOS_TRT__CONVERSION__DETECTION_CONVERTER_HPP_
