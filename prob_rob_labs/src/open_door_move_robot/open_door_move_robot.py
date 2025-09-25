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
        
        self.torque_pub = self.create_publisher(
            Float64, '/hinged_glass_door/torque', 1)
        self.robot_cmd_pub = self.create_publisher(
            Twist, '/cmd_vel', 1)
        self.counter = 0
        self.state = 'init'

        self.time_to_open = 3
        self.time_to_enter = 3

    def heartbeat(self): 
        self.log.info('heartbeat')
        if self.state == 'init':    
            self.counter = 0
            self.state = 'open_door'
        elif self.state == 'open_door':
            self.open_door()
            if self.counter * heartbeat_period == self.time_to_open:
                self.state = 'move_robot'
                self.counter = 0
            self.counter += 1
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
        twist.linear.x = 50.0
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
