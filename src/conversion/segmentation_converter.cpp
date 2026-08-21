// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_trt/conversion/segmentation_converter.hpp"

#include <algorithm>
#include <string>

namespace ros2_yolos_trt {
namespace conversion {

vision_msgs::msg::Detection2DArray segmentationToDetection2DArray(
  const std::vector<SegmentationResult>& segmentations,
  const std_msgs::msg::Header& header) {
  vision_msgs::msg::Detection2DArray msg;
  msg.header = header;
  msg.detections.reserve(segmentations.size());

  for (const auto& seg : segmentations) {
    vision_msgs::msg::Detection2D d;
    d.header = header;
    d.bbox.center.position.x = seg.bbox.x + seg.bbox.width / 2.0;
    d.bbox.center.position.y = seg.bbox.y + seg.bbox.height / 2.0;
    d.bbox.center.theta = 0.0;
    d.bbox.size_x = static_cast<double>(seg.bbox.width);
    d.bbox.size_y = static_cast<double>(seg.bbox.height);

    vision_msgs::msg::ObjectHypothesisWithPose hyp;
    hyp.hypothesis.class_id = seg.class_name.empty()
                                ? std::to_string(seg.class_id)
                                : seg.class_name;
    hyp.hypothesis.score = static_cast<double>(seg.confidence);
    d.results.push_back(hyp);
    msg.detections.push_back(std::move(d));
  }
  return msg;
}

sensor_msgs::msg::Image toCombinedMaskImage(
  const std::vector<SegmentationResult>& segmentations, int width, int height,
  const std_msgs::msg::Header& header) {
  sensor_msgs::msg::Image msg;
  msg.header = header;
  msg.encoding = "mono8";
  msg.is_bigendian = 0;
  msg.width = static_cast<uint32_t>(width);
  msg.height = static_cast<uint32_t>(height);
  msg.step = static_cast<uint32_t>(width);
  msg.data.assign(static_cast<size_t>(width) * height, 0);

  for (const auto& seg : segmentations) {
    if (seg.mask.empty() || seg.mask.type() != CV_8UC1) continue;
    // Class ids above 255 cannot be represented in mono8; clamp rather than
    // wrapping, so a large-vocabulary model degrades visibly instead of
    // silently relabelling pixels.
    const auto label =
      static_cast<uint8_t>(seg.class_id > 255 ? 255 : seg.class_id);

    const int rows = std::min(height, seg.mask.rows);
    const int cols = std::min(width, seg.mask.cols);
    for (int r = 0; r < rows; ++r) {
      const auto* mask_row = seg.mask.ptr<uint8_t>(r);
      for (int c = 0; c < cols; ++c) {
        if (mask_row[c] != 0) {
          msg.data[static_cast<size_t>(r) * width + c] = label;
        }
      }
    }
  }
  return msg;
}

}  // namespace conversion
}  // namespace ros2_yolos_trt
