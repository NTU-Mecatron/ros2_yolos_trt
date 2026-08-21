// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ROS2_YOLOS_TRT__NODES__CLASSIFIER_NODE_HPP_
#define ROS2_YOLOS_TRT__NODES__CLASSIFIER_NODE_HPP_

#include <memory>

#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/classification.hpp>

#include "ros2_yolos_trt/adapters/classifier_adapter.hpp"
#include "ros2_yolos_trt/nodes/yolos_lifecycle_node_base.hpp"

namespace ros2_yolos_trt {

class ROS2_YOLOS_TRT_PUBLIC YolosClassifierNode : public YolosLifecycleNodeBase {
public:
  explicit YolosClassifierNode(const rclcpp::NodeOptions& options);

  CallbackReturn on_configure(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_shutdown(const rclcpp_lifecycle::State& state) override;

private:
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);

  std::unique_ptr<IClassifierAdapter> adapter_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp_lifecycle::LifecyclePublisher<vision_msgs::msg::Classification>::SharedPtr
    cls_pub_;
  rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::Image>::SharedPtr
    debug_pub_;
  YolosConfig config_;
};

}  // namespace ros2_yolos_trt

#endif  // ROS2_YOLOS_TRT__NODES__CLASSIFIER_NODE_HPP_
