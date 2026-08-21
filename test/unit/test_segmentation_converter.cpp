// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include <gtest/gtest.h>

#include "ros2_yolos_trt/conversion/segmentation_converter.hpp"

using ros2_yolos_trt::SegmentationResult;
using ros2_yolos_trt::conversion::segmentationToDetection2DArray;
using ros2_yolos_trt::conversion::toCombinedMaskImage;

namespace {
SegmentationResult makeSeg(int x, int y, int w, int h, int id,
                           const std::string& name, const cv::Mat& mask) {
  SegmentationResult s;
  s.bbox.x = x;
  s.bbox.y = y;
  s.bbox.width = w;
  s.bbox.height = h;
  s.confidence = 0.8f;
  s.class_id = id;
  s.class_name = name;
  s.mask = mask;
  return s;
}
}  // namespace

TEST(SegmentationConverter, BoxesConvertToCentreSize) {
  const auto msg = segmentationToDetection2DArray(
    {makeSeg(10, 20, 30, 40, 1, "cat", cv::Mat::zeros(4, 4, CV_8UC1))},
    std_msgs::msg::Header{});
  ASSERT_EQ(msg.detections.size(), 1u);
  EXPECT_DOUBLE_EQ(msg.detections[0].bbox.center.position.x, 25.0);
  EXPECT_EQ(msg.detections[0].results[0].hypothesis.class_id, "cat");
}

// One mono8 image where each pixel holds the class id of the instance that
// covers it. Publishing one image per instance would flood the topic.
TEST(SegmentationConverter, CombinedMaskLabelsPixelsWithClassId) {
  cv::Mat mask = cv::Mat::zeros(4, 4, CV_8UC1);
  mask.at<uint8_t>(1, 1) = 255;

  const auto msg = toCombinedMaskImage(
    {makeSeg(0, 0, 4, 4, 7, "dog", mask)}, 4, 4, std_msgs::msg::Header{});

  EXPECT_EQ(msg.encoding, "mono8");
  EXPECT_EQ(msg.width, 4u);
  EXPECT_EQ(msg.height, 4u);
  ASSERT_EQ(msg.data.size(), 16u);
  EXPECT_EQ(msg.data[1 * 4 + 1], 7u);
  EXPECT_EQ(msg.data[0], 0u);
}

TEST(SegmentationConverter, EmptyInputProducesBlankMaskOfRequestedSize) {
  const auto msg = toCombinedMaskImage({}, 3, 2, std_msgs::msg::Header{});
  EXPECT_EQ(msg.width, 3u);
  EXPECT_EQ(msg.height, 2u);
  ASSERT_EQ(msg.data.size(), 6u);
  EXPECT_EQ(msg.data[0], 0u);
}
