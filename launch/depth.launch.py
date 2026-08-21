# Copyright 2026 YOLOs-TRT Team
# SPDX-License-Identifier: AGPL-3.0
"""Launch the TensorRT YOLO depth node as a managed lifecycle node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            'model_path',
            description='.onnx (engine built on first run) or .trt/.engine'),
        DeclareLaunchArgument('precision', default_value='fp16'),
        DeclareLaunchArgument('engine_cache_dir', default_value=''),
        DeclareLaunchArgument('image_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument(
            'depth_min', default_value='0.0',
            description='debug colorization clamp, metres; 0 = auto-range'),
        DeclareLaunchArgument('depth_max', default_value='0.0'),
        DeclareLaunchArgument('publish_debug_image', default_value='true'),
        DeclareLaunchArgument('publish_timing', default_value='false'),
    ]

    node = LifecycleNode(
        package='ros2_yolos_trt',
        executable='yolos_trt_depth_node',
        name='yolos_trt_depth',
        namespace='',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'precision': LaunchConfiguration('precision'),
            'engine_cache_dir': LaunchConfiguration('engine_cache_dir'),
            'depth_min': LaunchConfiguration('depth_min'),
            'depth_max': LaunchConfiguration('depth_max'),
            'publish_debug_image': LaunchConfiguration('publish_debug_image'),
            'publish_timing': LaunchConfiguration('publish_timing'),
        }],
        remappings=[('image', LaunchConfiguration('image_topic'))],
    )

    return LaunchDescription(args + [node])
