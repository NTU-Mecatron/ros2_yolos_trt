// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_trt/conversion/detection_converter.hpp"

#include <string>

namespace ros2_yolos_trt {
namespace conversion {

vision_msgs::msg::Detection2DArray toDetection2DArray(
  const std::vector<DetectionResult>& detections,
  const std_msgs::msg::Header& header) {
  vision_msgs::msg::Detection2DArray msg;
  msg.header = header;
  msg.detections.reserve(detections.size());

  for (const auto& det : detections) {
    vision_msgs::msg::Detection2D d;
    d.header = header;

    // The DTO stores top-left plus extent; vision_msgs wants the centre.
    d.bbox.center.position.x = det.bbox.x + det.bbox.width / 2.0;
    d.bbox.center.position.y = det.bbox.y + det.bbox.height / 2.0;
    d.bbox.center.theta = 0.0;
    d.bbox.size_x = static_cast<double>(det.bbox.width);
    d.bbox.size_y = static_cast<double>(det.bbox.height);

    vision_msgs::msg::ObjectHypothesisWithPose hyp;
    // class_id is a string in vision_msgs; prefer the label, fall back to the
    // numeric id so a missing labels file still yields identifiable output.
    hyp.hypothesis.class_id = det.class_name.empty()
                                ? std::to_string(det.class_id)
                                : det.class_name;
    hyp.hypothesis.score = static_cast<double>(det.confidence);
    d.results.push_back(hyp);

    msg.detections.push_back(std::move(d));
  }
  return msg;
}

}  // namespace conversion
}  // namespace ros2_yolos_trt
