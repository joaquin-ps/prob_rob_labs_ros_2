import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64
from geometry_msgs.msg import Twist

import numpy as np

heartbeat_period = 3

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
        self.feature_mean_threshold = 236.5
        self.feature_mean = None
        self.door_state = 'unknown'  # 'open', 'closed', 'unknown'

        # Initial conditions:
        self.belief = np.array([[0.5], [0.5]])  # P(door_open), P(door_closed)
        # self.belief_bar = np.zeros((2,1))

        # Measurement Probabilities: 
        self.P_z_open_given_door_open = 0.8465
        self.P_z_closed_given_door_open = 0.1535
        self.P_z_open_given_door_closed = 0.0075
        self.P_z_closed_given_door_closed = 0.9925

        # Measurement Model: 
        self.measurement_model = np.array(
            [[self.P_z_open_given_door_open, self.P_z_open_given_door_closed],
            [self.P_z_closed_given_door_open, self.P_z_closed_given_door_closed]]
        )

        # State machine variables:
        self.counter = 0
        self.state = 'init'
        
        self.declare_parameter('velocity', 50.0)
        self.velocity = self.get_parameter('velocity').get_parameter_value().double_value

        self.log.info("="*40)
        self.log.info(f'Robot velocity set to {self.velocity}')

        self.time_to_open = 2
        self.time_to_enter = 4

        self.move_threshold = 0.99999

    def _innovation(self, measurement):
        m = measurement
        unnormalized_posterior = \
            (self.measurement_model * np.repeat(self.belief, 2, axis=1).T)[m]
        posterior = unnormalized_posterior / np.sum(unnormalized_posterior)
        self.belief = posterior.reshape(2,1)

    def _update_action_state_machine(self):
        if self.belief[0] < self.move_threshold:
            self.state = "open_door"
        else:
            self.state = "move_robot"

    def heartbeat(self): 
        self.log.info('heartbeat')
        self.log.info('++++ state machine state: ' + self.state)

        if self.state == 'init':    
            self.state = 'open_door'
        elif self.state == 'open_door':
            self.log.info(f'P(door open) = {self.belief[0,0]}')
            self.open_door()
            measurement = 0 if self.door_state == 'open' else 1 
            self.log.info(f'Measurement = {measurement}')
            self._innovation(measurement)
            self.log.info(f'Posterior P(door open) = {self.belief[0,0]}')
            self._update_action_state_machine()
            
        elif self.state == 'move_robot':
            self.move_robot()
            if self.counter * heartbeat_period >= self.time_to_enter:
                self.state = 'close_door'
                self.counter = 0
                self.stop_robot()
            self.counter += 1
        elif self.state == 'close_door':
            self.close_door()
            if self.counter * heartbeat_period >= self.time_to_open:
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
