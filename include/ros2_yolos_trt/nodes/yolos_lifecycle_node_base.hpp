// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ROS2_YOLOS_TRT__NODES__YOLOS_LIFECYCLE_NODE_BASE_HPP_
#define ROS2_YOLOS_TRT__NODES__YOLOS_LIFECYCLE_NODE_BASE_HPP_

#include <string>

#include <lifecycle_msgs/msg/state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "ros2_yolos_trt/adapters/yolos_adapter_base.hpp"
#include "ros2_yolos_trt/visibility_control.hpp"

namespace ros2_yolos_trt {

/// @brief Parameter, lifecycle and timing plumbing shared by every task node.
///
/// Task nodes derive from this and contribute only their publishers and their
/// image callback.
class ROS2_YOLOS_TRT_PUBLIC YolosLifecycleNodeBase
  : public rclcpp_lifecycle::LifecycleNode {
public:
  YolosLifecycleNodeBase(const std::string& node_name,
                         const rclcpp::NodeOptions& options);
  ~YolosLifecycleNodeBase() override = default;

  using CallbackReturn =
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

protected:
  /// @brief Declare every parameter in config/default_params.yaml.
  void declareCommonParameters();

  /// @brief Read the declared parameters into a YolosConfig.
  [[nodiscard]] YolosConfig loadCommonConfig() const;

  [[nodiscard]] bool publishDebugImage() const;
  [[nodiscard]] bool publishTimingEnabled() const;
  [[nodiscard]] float depthMin() const;
  [[nodiscard]] float depthMax() const;
  [[nodiscard]] int maxDetections() const;

  /// @brief Create the timing publisher. Call from on_configure().
  void createTimingPublisher();

  /// @brief Publish [inference_ms, total_ms] when timing is enabled.
  void publishTimingSample(double inference_ms, double total_ms);

  rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Float64MultiArray>::SharedPtr
    timing_pub_;

  /// @brief Dedicated group so inference never blocks lifecycle transitions.
  rclcpp::CallbackGroup::SharedPtr inference_cb_group_;
};

}  // namespace ros2_yolos_trt

#endif  // ROS2_YOLOS_TRT__NODES__YOLOS_LIFECYCLE_NODE_BASE_HPP_
