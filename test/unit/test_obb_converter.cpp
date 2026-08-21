// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include <gtest/gtest.h>

#include "ros2_yolos_trt/conversion/obb_converter.hpp"

using ros2_yolos_trt::OBBResult;
using ros2_yolos_trt::conversion::toOBBDetection2DArray;

namespace {
OBBResult makeObb() {
  OBBResult o;
  o.bbox.center_x = 100.0f;
  o.bbox.center_y = 200.0f;
  o.bbox.width = 50.0f;
  o.bbox.height = 25.0f;
  o.bbox.angle = 0.7853982f;  // 45 degrees
  o.confidence = 0.88f;
  o.class_id = 4;
  o.class_name = "ship";
  return o;
}
}  // namespace

// The OBB DTO already stores a centre, so unlike the axis-aligned converters
// there is no origin shift — the angle is what must survive.
TEST(OBBConverter, PreservesCentreExtentAndAngle) {
  const auto msg = toOBBDetection2DArray({makeObb()}, std_msgs::msg::Header{});
  ASSERT_EQ(msg.detections.size(), 1u);
  const auto& d = msg.detections[0];
  EXPECT_DOUBLE_EQ(d.bbox.center.x, 100.0);
  EXPECT_DOUBLE_EQ(d.bbox.center.y, 200.0);
  EXPECT_DOUBLE_EQ(d.bbox.size_x, 50.0);
  EXPECT_DOUBLE_EQ(d.bbox.size_y, 25.0);
  EXPECT_NEAR(d.bbox.theta, 0.7853982, 1e-6);
}

TEST(OBBConverter, CarriesClassAndScore) {
  const auto msg = toOBBDetection2DArray({makeObb()}, std_msgs::msg::Header{});
  ASSERT_EQ(msg.detections.size(), 1u);
  EXPECT_EQ(msg.detections[0].class_id, 4);
  EXPECT_EQ(msg.detections[0].class_name, "ship");
  EXPECT_NEAR(msg.detections[0].score, 0.88, 1e-6);
}

TEST(OBBConverter, EmptyInputProducesEmptyArrayWithHeader) {
  std_msgs::msg::Header header;
  header.frame_id = "sat";
  const auto msg = toOBBDetection2DArray({}, header);
  EXPECT_TRUE(msg.detections.empty());
  EXPECT_EQ(msg.header.frame_id, "sat");
}
