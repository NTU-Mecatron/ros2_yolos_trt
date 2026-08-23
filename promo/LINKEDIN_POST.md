# LinkedIn post copy — ros2_yolos_trt

Assets in this folder:

| File | Use |
|------|-----|
| `ros2_yolos_trt_promo_portrait.mp4` | 1080×1350 (4:5), 32.6s — **the one to post.** Tallest allowed in the feed. |
| `ros2_yolos_trt_promo_square.mp4` | 1080×1080 re-cut, same content |
| `poster_portrait.png` / `poster_square.png` | Upload as the thumbnail so the hook card is the still frame |

Silent by design — LinkedIn autoplays muted, so every claim is burned in as text.
Regenerate either cut with `./scripts/make_promo.py [--aspect square]`.

---

## Version A — the main post

> Most ROS 2 vision nodes spend more time moving pixels than running the network.
>
> So I built **ros2_yolos_trt**: six YOLO tasks as composable ROS 2 lifecycle nodes on a TensorRT pipeline where the CPU never touches a pixel after `cv_bridge`.
>
> Letterbox, BGR→RGB and normalize all happen in one CUDA kernel that writes straight into the TensorRT input buffer. Inference dispatch is a captured CUDA graph, replayed per frame. Point a node at an `.onnx` and it builds the engine on first configure, then caches it keyed on GPU architecture, TensorRT version and precision — so one model file ships to a fleet of mixed GPUs.
>
> RTX 4090, fp16, mean inference latency reported by the nodes themselves:
> → detection 2.78 ms
> → pose 0.95 ms
> → depth 2.25 ms
> → segmentation 10.39 ms — with a wide tail I documented instead of hiding. Mask postprocessing is still on the CPU, so its cost scales with instance count. That's the next thing to move onto the GPU.
>
> The part I'm most pleased with: monocular **metric** depth, published as `32FC1` in metres. Not a normalised 0–1 map — real distance, in the unit ROS already expects. Which means this needs no glue code at all:
>
> `ros2 run depth_image_proc point_cloud_xyz --ros-args -r image_rect:=/yolos_trt_depth/depth/image`
>
> A point cloud out of one RGB camera.
>
> AGPL-3.0, Humble and Jazzy, TensorRT 10.x. The benchmark table in the README is generated from a real run by a script in the repo — not typed by hand.
>
> Repo link in the comments 👇
>
> #ROS2 #Robotics #TensorRT #ComputerVision #CUDA

## Version B — short cut

> Six YOLO tasks. One ROS 2 package. Zero CPU preprocessing.
>
> `ros2_yolos_trt` runs detection, segmentation, pose, OBB, classification and monocular metric depth as composable ROS 2 lifecycle nodes. Letterbox + BGR→RGB + normalize in a single CUDA kernel, straight into the TensorRT input buffer. CUDA graph replay for dispatch. Point it at an `.onnx` and the engine builds and caches itself.
>
> 2.78 ms mean inference for detection on an RTX 4090 (fp16, node-reported).
>
> The bit that doesn't exist in the ONNX sibling: depth published as `32FC1` **in metres**, so `depth_image_proc` gives you a point cloud from one RGB camera with no glue code.
>
> AGPL-3.0. Link in the comments.
>
> #ROS2 #Robotics #TensorRT #ComputerVision

---

## Posting notes

- **Put the repo URL in the first comment, not the post body.** Widely practised on the theory that LinkedIn down-ranks outbound links — commonly reported, not something I can measure. Harmless either way.
- **Upload the poster PNG as the video thumbnail.** Otherwise LinkedIn picks a random frame, and a mid-crossfade frame is a bad first impression.
- Reply to every comment in the first hour — early engagement drives distribution more than anything in the copy.
- Worth tagging in a comment rather than the body: the ROS Discourse crowd, and Ultralytics if you want reach beyond robotics.
