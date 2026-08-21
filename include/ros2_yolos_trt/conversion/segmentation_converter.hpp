// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ROS2_YOLOS_TRT__CONVERSION__SEGMENTATION_CONVERTER_HPP_
#define ROS2_YOLOS_TRT__CONVERSION__SEGMENTATION_CONVERTER_HPP_

#include <vector>

#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "ros2_yolos_trt/adapters/yolos_adapter_base.hpp"
#include "ros2_yolos_trt/visibility_control.hpp"

namespace ros2_yolos_trt {
namespace conversion {

/// @brief Boxes and scores from segmentation results.
ROS2_YOLOS_TRT_PUBLIC vision_msgs::msg::Detection2DArray
segmentationToDetection2DArray(
  const std::vector<SegmentationResult>& segmentations,
  const std_msgs::msg::Header& header);

/// @brief All instance masks flattened into one mono8 label image.
///
/// Each pixel holds the class id of the instance covering it, 0 = background.
/// Later instances overwrite earlier ones where they overlap.
ROS2_YOLOS_TRT_PUBLIC sensor_msgs::msg::Image toCombinedMaskImage(
  const std::vector<SegmentationResult>& segmentations, int width, int height,
  const std_msgs::msg::Header& header);

}  // namespace conversion
}  // namespace ros2_yolos_trt

#endif  // ROS2_YOLOS_TRT__CONVERSION__SEGMENTATION_CONVERTER_HPP_
