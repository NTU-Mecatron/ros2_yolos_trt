// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_trt/conversion/obb_converter.hpp"

namespace ros2_yolos_trt {
namespace conversion {

msg::OBBDetection2DArray toOBBDetection2DArray(
  const std::vector<OBBResult>& detections,
  const std_msgs::msg::Header& header) {
  msg::OBBDetection2DArray out;
  out.header = header;
  out.detections.reserve(detections.size());

  for (const auto& det : detections) {
    msg::OBBDetection2D d;
    d.header = header;
    d.bbox.center.x = static_cast<double>(det.bbox.center_x);
    d.bbox.center.y = static_cast<double>(det.bbox.center_y);
    d.bbox.center.z = 0.0;
    d.bbox.size_x = static_cast<double>(det.bbox.width);
    d.bbox.size_y = static_cast<double>(det.bbox.height);
    d.bbox.theta = static_cast<double>(det.bbox.angle);
    d.class_id = det.class_id;
    d.class_name = det.class_name;
    d.score = static_cast<double>(det.confidence);
    out.detections.push_back(std::move(d));
  }
  return out;
}

}  // namespace conversion
}  // namespace ros2_yolos_trt
