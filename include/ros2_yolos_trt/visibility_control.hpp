// Copyright 2026 YOLOs-TRT Team
// SPDX-License-Identifier: AGPL-3.0

#ifndef ROS2_YOLOS_TRT__VISIBILITY_CONTROL_HPP_
#define ROS2_YOLOS_TRT__VISIBILITY_CONTROL_HPP_

#if defined _WIN32 || defined __CYGWIN__
  #ifdef __GNUC__
    #define ROS2_YOLOS_TRT_EXPORT __attribute__((dllexport))
    #define ROS2_YOLOS_TRT_IMPORT __attribute__((dllimport))
  #else
    #define ROS2_YOLOS_TRT_EXPORT __declspec(dllexport)
    #define ROS2_YOLOS_TRT_IMPORT __declspec(dllimport)
  #endif
  #ifdef ROS2_YOLOS_TRT_BUILDING_DLL
    #define ROS2_YOLOS_TRT_PUBLIC ROS2_YOLOS_TRT_EXPORT
  #else
    #define ROS2_YOLOS_TRT_PUBLIC ROS2_YOLOS_TRT_IMPORT
  #endif
  #define ROS2_YOLOS_TRT_LOCAL
#else
  #define ROS2_YOLOS_TRT_EXPORT __attribute__((visibility("default")))
  #define ROS2_YOLOS_TRT_IMPORT
  #define ROS2_YOLOS_TRT_PUBLIC __attribute__((visibility("default")))
  #define ROS2_YOLOS_TRT_LOCAL  __attribute__((visibility("hidden")))
#endif

#endif  // ROS2_YOLOS_TRT__VISIBILITY_CONTROL_HPP_
