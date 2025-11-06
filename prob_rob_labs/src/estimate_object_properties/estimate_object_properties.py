import rclpy
from rclpy.node import Node
from prob_rob_msgs.msg import Point2DArrayStamped

heartbeat_period = 0.1

class EstimateObjectProperties(Node):

    def __init__(self):
        super().__init__('estimate_object_properties')
        self.log = self.get_logger()

        self.subscriber = self.create_subscription(
            Point2DArrayStamped, '/vision_cyan/corners',
            self.corners_callback, 1
        )
        self.x_coords = None
        self.y_coords = None

        self.timer = self.create_timer(heartbeat_period, self.heartbeat)

    def corners_callback(self, msg):
        corners = msg.points
        self.y_coords = [point.y for point in msg.points]
        self.x_coords = [point.x for point in msg.points]

        self.get_logger().info(f"y = {self.y_coords}")
        self.get_logger().info(f"x = {self.x_coords}")

    def get_width(self):
        return max(self.x_coords) - min(self.x_coords)

    def get_height(self):
        return max(self.y_coords) - min(self.y_coords)
        
    def get_vertical_symmetry_axis(self):
        return min(self.x_coords) + self.get_width() / 2

    def heartbeat(self):
        if self.y_coords is not None:
            if len(self.y_coords) > 1:
                self.log.info(f'Current Height: {self.get_height()}')
                self.log.info(f'Vertical Symmetry Axis: {self.get_vertical_symmetry_axis()}')

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
