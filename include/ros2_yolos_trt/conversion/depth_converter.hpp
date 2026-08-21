// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ROS2_YOLOS_TRT__CONVERSION__DEPTH_CONVERTER_HPP_
#define ROS2_YOLOS_TRT__CONVERSION__DEPTH_CONVERTER_HPP_

#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

#include "ros2_yolos_trt/adapters/yolos_adapter_base.hpp"
#include "ros2_yolos_trt/visibility_control.hpp"

namespace ros2_yolos_trt {
namespace conversion {

/// @brief Wrap a metric depth map as a 32FC1 sensor_msgs/Image.
///
/// Values are passed through unchanged: the exported model already produces
/// metres, and 32FC1-in-metres is what ROS depth tooling expects.
ROS2_YOLOS_TRT_PUBLIC sensor_msgs::msg::Image toDepthImage(
  const DepthResult& result, const std_msgs::msg::Header& header);

}  // namespace conversion
}  // namespace ros2_yolos_trt

#endif  // ROS2_YOLOS_TRT__CONVERSION__DEPTH_CONVERTER_HPP_
