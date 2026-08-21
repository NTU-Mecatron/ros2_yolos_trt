# Oriented (rotated) bounding box in 2D image space.

# Centre point of the box (z unused).
geometry_msgs/Point center

# Extent before rotation, pixels.
float64 size_x
float64 size_y

# Rotation, radians, counter-clockwise from the x-axis.
float64 theta
