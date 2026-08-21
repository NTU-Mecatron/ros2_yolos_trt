// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include <gtest/gtest.h>

#include "ros2_yolos_trt/conversion/detection_converter.hpp"

using ros2_yolos_trt::DetectionResult;
using ros2_yolos_trt::conversion::toDetection2DArray;

namespace {
DetectionResult makeDetection(int x, int y, int w, int h, float conf, int id,
                              const std::string& name) {
  DetectionResult d;
  d.bbox.x = x;
  d.bbox.y = y;
  d.bbox.width = w;
  d.bbox.height = h;
  d.confidence = conf;
  d.class_id = id;
  d.class_name = name;
  return d;
}
}  // namespace

TEST(DetectionConverter, EmptyInputProducesEmptyArrayWithHeader) {
  std_msgs::msg::Header header;
  header.frame_id = "camera_link";
  const auto msg = toDetection2DArray({}, header);
  EXPECT_TRUE(msg.detections.empty());
  EXPECT_EQ(msg.header.frame_id, "camera_link");
}

// vision_msgs uses centre + size; the DTO uses top-left + extent. Getting this
// wrong shifts every box by half its size.
TEST(DetectionConverter, ConvertsTopLeftExtentToCentreSize) {
  std_msgs::msg::Header header;
  const auto msg =
    toDetection2DArray({makeDetection(10, 20, 30, 40, 0.9f, 2, "car")}, header);

  ASSERT_EQ(msg.detections.size(), 1u);
  const auto& d = msg.detections[0];
  EXPECT_DOUBLE_EQ(d.bbox.center.position.x, 25.0);
  EXPECT_DOUBLE_EQ(d.bbox.center.position.y, 40.0);
  EXPECT_DOUBLE_EQ(d.bbox.size_x, 30.0);
  EXPECT_DOUBLE_EQ(d.bbox.size_y, 40.0);
}

TEST(DetectionConverter, CarriesClassIdNameAndScore) {
  std_msgs::msg::Header header;
  const auto msg =
    toDetection2DArray({makeDetection(0, 0, 4, 4, 0.75f, 7, "person")}, header);
  ASSERT_EQ(msg.detections.size(), 1u);
  ASSERT_EQ(msg.detections[0].results.size(), 1u);
  EXPECT_EQ(msg.detections[0].results[0].hypothesis.class_id, "person");
  EXPECT_NEAR(msg.detections[0].results[0].hypothesis.score, 0.75, 1e-6);
}

TEST(DetectionConverter, FallsBackToNumericClassIdWhenNameMissing) {
  std_msgs::msg::Header header;
  const auto msg =
    toDetection2DArray({makeDetection(0, 0, 4, 4, 0.5f, 42, "")}, header);
  ASSERT_EQ(msg.detections.size(), 1u);
  EXPECT_EQ(msg.detections[0].results[0].hypothesis.class_id, "42");
}
