// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_trt/conversion/pose_converter.hpp"

#include <string>

namespace ros2_yolos_trt {
namespace conversion {

vision_msgs::msg::Detection2DArray poseToDetection2DArray(
  const std::vector<PoseResult>& poses, const std_msgs::msg::Header& header) {
  vision_msgs::msg::Detection2DArray msg;
  msg.header = header;
  msg.detections.reserve(poses.size());

  for (const auto& pose : poses) {
    vision_msgs::msg::Detection2D d;
    d.header = header;
    d.bbox.center.position.x = pose.bbox.x + pose.bbox.width / 2.0;
    d.bbox.center.position.y = pose.bbox.y + pose.bbox.height / 2.0;
    d.bbox.center.theta = 0.0;
    d.bbox.size_x = static_cast<double>(pose.bbox.width);
    d.bbox.size_y = static_cast<double>(pose.bbox.height);

    vision_msgs::msg::ObjectHypothesisWithPose hyp;
    hyp.hypothesis.class_id = pose.class_name.empty()
                                ? std::to_string(pose.class_id)
                                : pose.class_name;
    hyp.hypothesis.score = static_cast<double>(pose.confidence);
    d.results.push_back(hyp);
    msg.detections.push_back(std::move(d));
  }
  return msg;
}

std::vector<msg::KeyPoint2DArray> toKeyPoint2DArrays(
  const std::vector<PoseResult>& poses, const std_msgs::msg::Header& header) {
  std::vector<msg::KeyPoint2DArray> arrays;
  arrays.reserve(poses.size());

  for (const auto& pose : poses) {
    msg::KeyPoint2DArray arr;
    arr.header = header;
    arr.confidence = static_cast<double>(pose.confidence);
    arr.class_id = pose.class_id;
    arr.bounding_box_center.x = pose.bbox.x + pose.bbox.width / 2.0;
    arr.bounding_box_center.y = pose.bbox.y + pose.bbox.height / 2.0;
    arr.bounding_box_center.z = 0.0;
    arr.bounding_box_size_x = static_cast<double>(pose.bbox.width);
    arr.bounding_box_size_y = static_cast<double>(pose.bbox.height);

    arr.keypoints.reserve(pose.keypoints.size());
    for (const auto& kp : pose.keypoints) {
      msg::KeyPoint2D out;
      out.x = static_cast<double>(kp.x);
      out.y = static_cast<double>(kp.y);
      out.confidence = static_cast<double>(kp.confidence);
      out.id = kp.id;
      arr.keypoints.push_back(out);
    }
    arrays.push_back(std::move(arr));
  }
  return arrays;
}

}  // namespace conversion
}  // namespace ros2_yolos_trt
