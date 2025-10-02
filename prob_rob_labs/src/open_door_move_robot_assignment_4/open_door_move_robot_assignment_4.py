import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64
from geometry_msgs.msg import Twist

heartbeat_period = 0.1

class OpenDoorMoveRobot(Node):

    def __init__(self):
        super().__init__('open_door_move_robot')
        self.log = self.get_logger()
        self.timer = self.create_timer(heartbeat_period, self.heartbeat)

        # Command publishers:
        self.torque_pub = self.create_publisher(
            Float64, '/hinged_glass_door/torque', 1)
        self.robot_cmd_pub = self.create_publisher(
            Twist, '/cmd_vel', 1)
        
        # Door state:
        self.feature_mean_sub = self.create_subscription(
            Float64, '/feature_mean', self.feature_mean_callback, 1)
        self.feature_mean_threshold = 280.0
        self.feature_mean = None
        self.door_state = 'unknown'  # 'open', 'closed', 'unknown'

        # State machine variables:
        self.counter = 0
        self.state = 'init'
        
        self.declare_parameter('velocity', 50.0)
        self.velocity = self.get_parameter('velocity').get_parameter_value().double_value

        self.log.info("="*40)
        self.log.info(f'Robot velocity set to {self.velocity}')

        self.time_to_open = 2
        self.time_to_enter = 4

    def heartbeat(self): 
        self.log.info('heartbeat')
        self.log.info('++++ state machine state: ' + self.state)
        if self.state == 'init':    
            self.state = 'open_door'
        elif self.state == 'open_door':
            self.open_door()
            if self.door_state == 'open':
                self.state = 'move_robot'
        elif self.state == 'move_robot':
            self.move_robot()
            if self.counter * heartbeat_period == self.time_to_enter:
                self.state = 'close_door'
                self.counter = 0
                self.stop_robot()
            self.counter += 1
        elif self.state == 'close_door':
            self.close_door()
            if self.counter * heartbeat_period == self.time_to_open:
                self.state = 'done'
                self.counter = 0
            self.counter += 1
        elif self.state == 'done':
            self.log.info('done')
            self.timer.cancel()   
            
    
    def feature_mean_callback(self, msg):
        self.feature_mean = msg.data
        self.check_door_state()

    def check_door_state(self):
        if self.feature_mean is None:
            self.door_state = 'unknown'
        elif self.feature_mean > self.feature_mean_threshold:
            self.door_state = 'closed'
        elif self.feature_mean <= self.feature_mean_threshold:
            self.door_state = 'open'
        else:
            self.door_state = 'unknown'
        # self.log.info(f'door state: {self.door_state}')

    def open_door(self):
        torque = Float64()
        torque.data = 5.0
        self.torque_pub.publish(torque)
        self.log.info('applying torque to door')

    def close_door(self):
        torque = Float64()
        torque.data = -5.0
        self.torque_pub.publish(torque)
        self.log.info('applying reverse torque to door')

    def move_robot(self):
        twist = Twist()
        twist.linear.x = self.velocity
        twist.angular.z = 0.0
        self.robot_cmd_pub.publish(twist)
        self.log.info('moving robot forward')

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.robot_cmd_pub.publish(twist)
        self.log.info('stopping robot')

    def spin(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    open_door_move_robot = OpenDoorMoveRobot()
    open_door_move_robot.spin()
    open_door_move_robot.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
