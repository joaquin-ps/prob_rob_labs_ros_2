import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64
from prob_rob_msgs.msg import Point2DArrayStamped
from sensor_msgs.msg import CameraInfo

import math

heartbeat_period = 0.1

class EstimateObjectProperties(Node):

    def __init__(self):
        super().__init__('estimate_object_properties')
        self.log = self.get_logger()

        # Camera Info:
        self.camera_subscriber = self.create_subscription(
            CameraInfo, '/camera/camera_info',
            self.camera_callback, 1
        )
        self.f_x, self.f_y, self.c_x, self.c_y = None, None, None, None

        # Color:
        colors = ["yellow", "red", "magenta", "green", "cyan"]

        self.declare_parameter('landmark_color', 'cyan')
        self.current_color = self.get_parameter('landmark_color').get_parameter_value().string_value

        assert self.current_color in colors, f"Choose one of the following colors: {colors}"
        
        self.get_logger().info(f"Tracking {self.current_color} obstacle")

        # Landmarks:
        self.declare_parameter('landmark_height', 0.5)
        self.h = self.get_parameter('landmark_height').get_parameter_value().double_value

        assert self.h > 0, "Height must be higher than 0."

        self.get_logger().info(f"Landmark Height: {self.h}")

        # Validity Requirements:
        self.min_points = 2
        self.max_width = 115

        # Corners subscriber
        self.points_subscriber = self.create_subscription(
            Point2DArrayStamped, f'/vision_{self.current_color}/corners',
            self.corners_callback, 1
        )
        self.coords_dict = {color: (None, None) for color in colors} # Meant for when we need to track all colors.

        self.get_logger().info(f"Colors {self.coords_dict}")

        # Result publishers: 
        self.distance_publisher = self.create_publisher(
            Float64, '/distance', 1
        )
        self.distance_msg = Float64()

        self.bearing_publisher = self.create_publisher(
            Float64, '/bearing', 1
        )
        self.bearing_msg = Float64()

        self.timer = self.create_timer(heartbeat_period, self.heartbeat)

    # Camera:

    def camera_callback(self, msg):
        k = msg.k
        self.f_x, self.f_y, self.c_x, self.c_y = k[0], k[4], k[2], k[5]
        # self.get_logger().info(f"f_x = {self.f_x}, f_y = {self.f_y}, c_x = {self.c_x}, c_y = {self.c_y}")
    
    # Landmarks:

    def corners_callback(self, msg):
        corners = msg.points

        x_coords = [point.x for point in msg.points]
        y_coords = [point.y for point in msg.points]

        self.coords_dict[self.current_color] = (x_coords, y_coords)

        # self.get_logger().info(f"y = {y_coords}")
        # self.get_logger().info(f"x = {x_coords}")

    def get_width(self, x_coords):
        return max(x_coords) - min(x_coords)

    def get_height(self, y_coords):
        return max(y_coords) - min(y_coords)
        
    def get_vertical_symmetry_axis(self, x_coords, width):
        return min(x_coords) + width / 2

    def calculate_bearing(self, x_p):
        return math.atan2(self.c_x - x_p, self.f_x)

    def calculate_distance(self, theta, delta_y):
        return self.h * self.f_y / (delta_y * math.cos(theta))

    def heartbeat(self):
        x_coords, y_coords = self.coords_dict[self.current_color]

        if y_coords is not None:
            num_points = len(x_coords)
            
            if num_points < self.min_points:
                self.log.info(f'Not enough points (min {self.min_points}).')
                return
        
            width = self.get_width(x_coords)

            # If we are too close we cannot see the top (which is when the width is really high)
            if width > self.max_width:
                self.log.info('Too close to landmark.')
                return

            height = self.get_height(y_coords)
            x_p = self.get_vertical_symmetry_axis(x_coords, width)

            bearing = self.calculate_bearing(x_p = x_p)
            distance = self.calculate_distance(theta = bearing, delta_y = height)                

            self.log.info(f'Current Width: {width}')
            self.log.info(f'Current Height: {height}')
            self.log.info(f'Vertical Symmetry Axis: {x_p}')
            
            self.publish_distance_bearing(bearing, distance)

    def publish_distance_bearing(self, bearing, distance):
        self.log.info(f'Bearing: {bearing}')
        self.log.info(f'Distance: {distance}')

        self.bearing_msg.data = bearing
        self.distance_msg.data = distance

        self.bearing_publisher.publish(self.bearing_msg)
        self.distance_publisher.publish(self.distance_msg)

    def spin(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    estimate_object_properties = EstimateObjectProperties()
    estimate_object_properties.spin()
    estimate_object_properties.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
