// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ROS2_YOLOS_TRT__CONVERSION__POSE_CONVERTER_HPP_
#define ROS2_YOLOS_TRT__CONVERSION__POSE_CONVERTER_HPP_

#include <vector>

#include <std_msgs/msg/header.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "ros2_yolos_trt/adapters/yolos_adapter_base.hpp"
#include "ros2_yolos_trt/msg/key_point2_d_array.hpp"
#include "ros2_yolos_trt/visibility_control.hpp"

namespace ros2_yolos_trt {
namespace conversion {

/// @brief Person boxes from pose results.
ROS2_YOLOS_TRT_PUBLIC vision_msgs::msg::Detection2DArray poseToDetection2DArray(
  const std::vector<PoseResult>& poses, const std_msgs::msg::Header& header);

/// @brief One KeyPoint2DArray per detected instance.
///
/// Detection2DArray has no keypoint field, so without this the keypoints —
/// the actual output of pose estimation — would be discarded.
ROS2_YOLOS_TRT_PUBLIC std::vector<msg::KeyPoint2DArray> toKeyPoint2DArrays(
  const std::vector<PoseResult>& poses, const std_msgs::msg::Header& header);

}  // namespace conversion
}  // namespace ros2_yolos_trt

#endif  // ROS2_YOLOS_TRT__CONVERSION__POSE_CONVERTER_HPP_
