import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class MotionPublisher(Node):

    def __init__(self):
        super().__init__('motion_publisher')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = Twist()
        
        msg.linear.x, msg.linear.y, msg.linear.z = 1.0, 0.0, 0.0
        msg.angular.x, msg.angular.y, msg.angular.z = 0.0, 0.0, 0.5
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)

    motion_publisher = MotionPublisher()

    rclpy.spin(motion_publisher)

    motion_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
