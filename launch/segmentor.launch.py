# Copyright 2026 YOLOs-TRT Team
# SPDX-License-Identifier: AGPL-3.0
"""Launch the TensorRT YOLO segmentor node as a managed lifecycle node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            'model_path',
            description='.onnx (engine built on first run) or .trt/.engine'),
        DeclareLaunchArgument('labels_path', default_value=''),
        DeclareLaunchArgument('precision', default_value='fp16'),
        DeclareLaunchArgument('engine_cache_dir', default_value=''),
        DeclareLaunchArgument('image_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('conf_threshold', default_value='0.4'),
        DeclareLaunchArgument('nms_threshold', default_value='0.45'),
        DeclareLaunchArgument('publish_debug_image', default_value='false'),
        DeclareLaunchArgument('publish_timing', default_value='false'),
    ]

    node = LifecycleNode(
        package='ros2_yolos_trt',
        executable='yolos_trt_segmentor_node',
        name='yolos_trt_segmentor',
        namespace='',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'labels_path': LaunchConfiguration('labels_path'),
            'precision': LaunchConfiguration('precision'),
            'engine_cache_dir': LaunchConfiguration('engine_cache_dir'),
            'conf_threshold': LaunchConfiguration('conf_threshold'),
            'nms_threshold': LaunchConfiguration('nms_threshold'),
            'publish_debug_image': LaunchConfiguration('publish_debug_image'),
            'publish_timing': LaunchConfiguration('publish_timing'),
        }],
        remappings=[('image', LaunchConfiguration('image_topic'))],
    )

    return LaunchDescription(args + [node])
