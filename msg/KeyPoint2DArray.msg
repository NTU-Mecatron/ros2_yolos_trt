# Keypoints belonging to one detected instance.

std_msgs/Header header

ros2_yolos_trt/KeyPoint2D[] keypoints

# Bounding box of the instance these keypoints belong to.
geometry_msgs/Point bounding_box_center
float64 bounding_box_size_x
float64 bounding_box_size_y

# Detection confidence for the instance.
float64 confidence

# Class id (0 = person for the released pose models).
int32 class_id
