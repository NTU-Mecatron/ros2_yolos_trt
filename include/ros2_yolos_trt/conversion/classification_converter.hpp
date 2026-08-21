// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ROS2_YOLOS_TRT__CONVERSION__CLASSIFICATION_CONVERTER_HPP_
#define ROS2_YOLOS_TRT__CONVERSION__CLASSIFICATION_CONVERTER_HPP_

#include <std_msgs/msg/header.hpp>
#include <vision_msgs/msg/classification.hpp>

#include "ros2_yolos_trt/adapters/yolos_adapter_base.hpp"
#include "ros2_yolos_trt/visibility_control.hpp"

namespace ros2_yolos_trt {
namespace conversion {

/// @brief Convert a top-1 classification result to vision_msgs/Classification.
ROS2_YOLOS_TRT_PUBLIC vision_msgs::msg::Classification toClassification(
  const ClassificationResult& result, const std_msgs::msg::Header& header);

}  // namespace conversion
}  // namespace ros2_yolos_trt

#endif  // ROS2_YOLOS_TRT__CONVERSION__CLASSIFICATION_CONVERTER_HPP_
