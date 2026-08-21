// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ROS2_YOLOS_TRT__NODES__OBB_NODE_HPP_
#define ROS2_YOLOS_TRT__NODES__OBB_NODE_HPP_

#include <memory>

#include <sensor_msgs/msg/image.hpp>
#include "ros2_yolos_trt/msg/obb_detection2_d_array.hpp"

#include "ros2_yolos_trt/adapters/obb_adapter.hpp"
#include "ros2_yolos_trt/nodes/yolos_lifecycle_node_base.hpp"

namespace ros2_yolos_trt {

class ROS2_YOLOS_TRT_PUBLIC YolosOBBNode : public YolosLifecycleNodeBase {
public:
  explicit YolosOBBNode(const rclcpp::NodeOptions& options);

  CallbackReturn on_configure(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_shutdown(const rclcpp_lifecycle::State& state) override;

private:
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);

  std::unique_ptr<IOBBAdapter> adapter_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp_lifecycle::LifecyclePublisher<msg::OBBDetection2DArray>::SharedPtr
    obb_pub_;
  rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::Image>::SharedPtr
    debug_pub_;
  YolosConfig config_;
};

}  // namespace ros2_yolos_trt

#endif  // ROS2_YOLOS_TRT__NODES__OBB_NODE_HPP_
