// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_trt/conversion/depth_converter.hpp"

#include <cstring>

namespace ros2_yolos_trt {
namespace conversion {

sensor_msgs::msg::Image toDepthImage(const DepthResult& result,
                                     const std_msgs::msg::Header& header) {
  sensor_msgs::msg::Image msg;
  msg.header = header;
  msg.encoding = "32FC1";
  msg.is_bigendian = 0;

  // The adapter guarantees CV_32FC1; check the contract cheaply rather than
  // silently emitting mislabelled data.
  if (result.depth_metres.empty() || result.depth_metres.type() != CV_32FC1) {
    msg.width = 0;
    msg.height = 0;
    msg.step = 0;
    return msg;
  }

  const cv::Mat& depth = result.depth_metres;
  msg.width = static_cast<uint32_t>(depth.cols);
  msg.height = static_cast<uint32_t>(depth.rows);
  msg.step = static_cast<uint32_t>(depth.cols * sizeof(float));
  msg.data.resize(static_cast<size_t>(msg.step) * msg.height);

  // Copy row by row: a cv::Mat may be padded, msg.data must be dense.
  for (int row = 0; row < depth.rows; ++row) {
    std::memcpy(msg.data.data() + static_cast<size_t>(row) * msg.step,
                depth.ptr<float>(row), msg.step);
  }
  return msg;
}

}  // namespace conversion
}  // namespace ros2_yolos_trt
