| Task | Model | Inference mean | p50 | p95 | p99 | Inference FPS | Publish rate |
|------|-------|---------------:|----:|----:|----:|--------------:|-------------:|
| Detection | `yolo11n.onnx` | 2.782 ms | 2.873 ms | 3.453 ms | 3.932 ms | **359.5** | 201.3 Hz |
| Segmentation | `yolo11n-seg.onnx` | 10.393 ms | 5.718 ms | 49.208 ms | 63.443 ms | **96.2** | 83.8 Hz |
| Pose | `yolo11n-pose.onnx` | 0.946 ms | 0.945 ms | 0.988 ms | 1.011 ms | **1056.7** | 203.0 Hz |
| Depth | `yolo26n-depth.onnx` | 2.251 ms | 1.752 ms | 2.0 ms | 24.374 ms | **444.3** | 146.6 Hz |
