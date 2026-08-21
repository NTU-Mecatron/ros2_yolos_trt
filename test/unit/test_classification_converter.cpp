// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include <gtest/gtest.h>

#include "ros2_yolos_trt/conversion/classification_converter.hpp"

using ros2_yolos_trt::ClassificationResult;
using ros2_yolos_trt::conversion::toClassification;

TEST(ClassificationConverter, CarriesTopOneHypothesis) {
  ClassificationResult r;
  r.class_id = 207;
  r.confidence = 0.91f;
  r.class_name = "golden retriever";

  std_msgs::msg::Header header;
  header.frame_id = "cam";
  const auto msg = toClassification(r, header);

  ASSERT_EQ(msg.results.size(), 1u);
  EXPECT_EQ(msg.results[0].class_id, "golden retriever");
  EXPECT_NEAR(msg.results[0].score, 0.91, 1e-6);
  EXPECT_EQ(msg.header.frame_id, "cam");
}

TEST(ClassificationConverter, FallsBackToNumericIdWithoutLabels) {
  ClassificationResult r;
  r.class_id = 5;
  r.confidence = 0.5f;
  const auto msg = toClassification(r, std_msgs::msg::Header{});
  ASSERT_EQ(msg.results.size(), 1u);
  EXPECT_EQ(msg.results[0].class_id, "5");
}

// class_id -1 is the adapter's "nothing produced" sentinel.
TEST(ClassificationConverter, NegativeClassIdProducesNoHypothesis) {
  ClassificationResult r;
  r.class_id = -1;
  const auto msg = toClassification(r, std_msgs::msg::Header{});
  EXPECT_TRUE(msg.results.empty());
}
