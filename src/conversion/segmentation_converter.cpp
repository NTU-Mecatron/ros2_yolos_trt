// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include "ros2_yolos_trt/conversion/segmentation_converter.hpp"

#include <algorithm>
#include <string>

#include <opencv2/imgproc.hpp>

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

namespace {

foxglove_msgs::msg::Color toFoxgloveColor(const cv::Scalar& bgr, double alpha) {
  foxglove_msgs::msg::Color c;
  c.r = bgr[2] / 255.0;
  c.g = bgr[1] / 255.0;
  c.b = bgr[0] / 255.0;
  c.a = alpha;
  return c;
}

}  // namespace

foxglove_msgs::msg::ImageAnnotations segmentationToImageAnnotations(
  const std::vector<SegmentationResult>& segmentations,
  const std::vector<cv::Scalar>& colors_bgr,
  const std_msgs::msg::Header& header) {
  constexpr double kFontSize = 14.0;
  constexpr double kSimplifyEpsilonPx = 1.5;

  foxglove_msgs::msg::ImageAnnotations out;
  for (const auto& seg : segmentations) {
    const cv::Scalar color =
      (seg.class_id >= 0 && static_cast<size_t>(seg.class_id) < colors_bgr.size())
        ? colors_bgr[seg.class_id]
        : cv::Scalar(0, 255, 0);

    if (!seg.mask.empty() && seg.mask.type() == CV_8UC1) {
      std::vector<std::vector<cv::Point>> contours;
      cv::findContours(seg.mask, contours, cv::RETR_EXTERNAL,
                       cv::CHAIN_APPROX_SIMPLE);
      for (const auto& contour : contours) {
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, kSimplifyEpsilonPx, true);
        if (approx.size() < 3) continue;

        foxglove_msgs::msg::PointsAnnotation pa;
        pa.timestamp = header.stamp;
        pa.type = foxglove_msgs::msg::PointsAnnotation::LINE_LOOP;
        pa.outline_color = toFoxgloveColor(color, 1.0);
        pa.fill_color = toFoxgloveColor(color, 0.0);
        pa.thickness = 2.0;
        pa.points.reserve(approx.size());
        for (const auto& p : approx) {
          foxglove_msgs::msg::Point2 fp;
          fp.x = p.x;
          fp.y = p.y;
          pa.points.push_back(fp);
        }
        out.points.push_back(std::move(pa));
      }
    }

    // Foxglove text positions use the text baseline, not the top-left corner.
    foxglove_msgs::msg::TextAnnotation ta;
    ta.timestamp = header.stamp;
    double y = static_cast<double>(seg.bbox.y) + kFontSize;
    if (seg.mask.rows > 0) y = std::min(y, static_cast<double>(seg.mask.rows - 1));
    ta.position.x = std::max(0, seg.bbox.x);
    ta.position.y = std::max(kFontSize, y);
    ta.text = seg.class_name.empty() ? std::to_string(seg.class_id) : seg.class_name;
    ta.font_size = kFontSize;
    ta.text_color = toFoxgloveColor(color, 1.0);
    ta.background_color = toFoxgloveColor(cv::Scalar(0, 0, 0), 0.4);
    out.texts.push_back(std::move(ta));
  }
  return out;
}

}  // namespace conversion
}  // namespace ros2_yolos_trt
