import rclpy
from rclpy.node import Node

from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Header

class Turtlebot3GroundTruth(Node):

    def __init__(self):
        super().__init__('turtlebot3_ground_truth')
        self.log = self.get_logger()

        # Subscribe to ground truth from gazebo
        self.link_states_sub = self.create_subscription(
            LinkStates, '/gazebo/link_states', self.link_states_callback, 1
        )

        # Publishers
        self.pose_pub = self.create_publisher(
            PoseStamped, '/tb3/ground_truth/pose', 1
        )
        self.twist_pub = self.create_publisher(
            TwistStamped, '/tb3/ground_truth/twist', 1
        )

        # Reference frame and link name:
        self.log.info("="*50)
        self.declare_parameter('reference_frame', 'odom')
        self.reference_frame = self.get_parameter('reference_frame').get_parameter_value().string_value
        self.log.info(f"Reference frame set to {self.reference_frame}")
        
        self.declare_parameter('link_name', 'base_link')
        link_name = self.get_parameter('link_name').get_parameter_value().string_value
        
        self.link_name = f'waffle_pi::{link_name}'
        self.log.info(f"Publishing pose and twist for {self.link_name}")

        # Publisher
        self.pose = None
        self.twist = None

    def link_states_callback(self, msg):

        timestamp = self.get_clock().now().to_msg()
        
        try:
            link_index = msg.name.index(self.link_name)

            self.pose = msg.pose[link_index]
            self.twist = msg.twist[link_index]
            
            self._pub_ground_truth(timestamp)

        except ValueError:
            self.log.warn(f"{self.link_name} not being published")
            pass
    
    def _pub_ground_truth(self, timestamp):
        if self.pose is not None and self.twist is not None:
            pose_msg, twist_msg = PoseStamped(), TwistStamped()

            header = Header()
            header.stamp = timestamp
            header.frame_id = self.reference_frame

            pose_msg.pose, twist_msg.twist = self.pose, self.twist
            pose_msg.header, twist_msg.header = header, header

            self.pose_pub.publish(pose_msg)
            self.twist_pub.publish(twist_msg)

    def spin(self):
        rclpy.spin(self)

def main():
    rclpy.init()
    turtlebot3_ground_truth = Turtlebot3GroundTruth()
    turtlebot3_ground_truth.spin()
    turtlebot3_ground_truth.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
