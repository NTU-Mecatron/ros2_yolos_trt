// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#include <gtest/gtest.h>

#include <memory>

#include <lifecycle_msgs/msg/state.hpp>
#include <rclcpp/rclcpp.hpp>

#include "ros2_yolos_trt/nodes/yolos_lifecycle_node_base.hpp"

using ros2_yolos_trt::YolosLifecycleNodeBase;

// Concrete probe exposing the protected helpers for testing.
class ProbeNode : public YolosLifecycleNodeBase {
public:
  explicit ProbeNode(const rclcpp::NodeOptions& options)
  : YolosLifecycleNodeBase("probe_node", options) {
    declareCommonParameters();
  }
  using YolosLifecycleNodeBase::loadCommonConfig;
  using YolosLifecycleNodeBase::publishDebugImage;
};

class LifecycleBaseTest : public ::testing::Test {
protected:
  void SetUp() override { rclcpp::init(0, nullptr); }
  void TearDown() override { rclcpp::shutdown(); }
};

TEST_F(LifecycleBaseTest, DeclaresTensorRtParameterDefaults) {
  auto node = std::make_shared<ProbeNode>(rclcpp::NodeOptions{});
  const auto cfg = node->loadCommonConfig();
  EXPECT_EQ(cfg.precision, "fp16");
  EXPECT_EQ(cfg.dla_core, -1);
  EXPECT_FLOAT_EQ(cfg.conf_threshold, 0.4f);
  EXPECT_FALSE(node->publishDebugImage());
}

TEST_F(LifecycleBaseTest, ParameterOverridesReachTheConfig) {
  rclcpp::NodeOptions options;
  options.parameter_overrides({
    rclcpp::Parameter("model_path", "/models/x.onnx"),
    rclcpp::Parameter("precision", "fp32"),
    rclcpp::Parameter("conf_threshold", 0.7),
    rclcpp::Parameter("publish_debug_image", true),
  });
  auto node = std::make_shared<ProbeNode>(options);
  const auto cfg = node->loadCommonConfig();
  EXPECT_EQ(cfg.model_path, "/models/x.onnx");
  EXPECT_EQ(cfg.precision, "fp32");
  EXPECT_FLOAT_EQ(cfg.conf_threshold, 0.7f);
  EXPECT_TRUE(node->publishDebugImage());
}

TEST_F(LifecycleBaseTest, StartsUnconfigured) {
  auto node = std::make_shared<ProbeNode>(rclcpp::NodeOptions{});
  EXPECT_EQ(node->get_current_state().id(),
            lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
}
