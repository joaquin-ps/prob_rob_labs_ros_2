"""
Make sure to run lab3/reset_world.sh while running this node. 
"""

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64

heartbeat_period = 0.1

class Lab3EstimateProbabilities(Node):

    def __init__(self):
        super().__init__('lab3_estimate_probabilities')
        self.log = self.get_logger()
        self.timer = self.create_timer(heartbeat_period, self.heartbeat)

        self.threshold = 0.5

        # Command publishers:
        self.torque_pub = self.create_publisher(
            Float64, '/hinged_glass_door/torque', 1)
        self.torque = 20.0

        # Conditional Probabilities: 
        self.P_z_open_given_door_open = 0.0
        self.P_z_open_given_door_closed = 0.0
        self.P_z_closed_given_door_open = 0.0
        self.P_z_closed_given_door_closed = 0.0

        # Door state:
        self.feature_mean_sub = self.create_subscription(
            Float64, '/feature_mean', self.feature_mean_callback, 1)
        self.feature_mean_threshold = 236.5
        self.feature_mean = None
        self.door_measurement = 'unknown'  # 'open', 'closed', 'unknown
        
        # State machine variables:
        self.state = 'init'  # 'init', 'wait_open', 'open', 'wait_close', 'closed', 'display'

        # Timer variables
        self.counter = 0
        self.sample_counts = 1000
        self.wait_for_door = 5.0

    def feature_mean_callback(self, msg):
        self.feature_mean = msg.data
        # self.log.info(f'feature mean: {self.feature_mean:.1f}')
        self.measure_door_state()

    def measure_door_state(self):
        if self.feature_mean is None:
            self.door_measurement = 'unknown'
        elif self.feature_mean > self.feature_mean_threshold:
            self.door_measurement = 'closed'
        else:
            self.door_measurement = 'open'
        # self.log.info(f'door measurement: {self.door_measurement}')

    def open_door(self):
        torque = Float64()
        torque.data = self.torque
        self.torque_pub.publish(torque)

    def close_door(self):
        torque = Float64()
        torque.data = -self.torque
        self.torque_pub.publish(torque)

    def display_probabilities(self):
        self.log.info(f'P(z=open|door=open) = {self.P_z_open_given_door_open}')
        self.log.info(f'P(z=closed|door=open) = {self.P_z_closed_given_door_open}')
        self.log.info(f'P(z=open|door=closed) = {self.P_z_open_given_door_closed}')
        self.log.info(f'P(z=closed|door=closed) = {self.P_z_closed_given_door_closed}')

    def heartbeat(self):
        self.log.info('heartbeat')
        
        if self.state == 'init':
            self.open_door()
            self.state = 'wait_open'
            self.counter = 0
            self.log.info('State: wait')
        elif self.state == 'wait_open':
            self.log.info('State: wait_open, Progress: ' + str(self.counter * heartbeat_period) + ' / ' + str(self.wait_for_door))
            if self.counter * heartbeat_period >= self.wait_for_door:
                self.state = 'open'
                self.counter = 0
                self.log.info('State: open')
            self.counter += 1
        elif self.state == 'open':
            self.log.info('State: open, Progress: ' + str(self.counter) + ' / ' + str(self.sample_counts))
            if self.feature_mean is not None:
                if self.counter < self.sample_counts:    
                    if self.door_measurement == 'open':
                        self.P_z_open_given_door_open += 1
                    else:
                        self.P_z_closed_given_door_open += 1
                else:
                    self.counter = 0 
                    self.close_door()
                    self.state = 'wait_close'
            self.counter += 1
        elif self.state == 'wait_close':
            self.log.info('State: wait_close, Progress: ' + str(self.counter * heartbeat_period) + ' / ' + str(self.wait_for_door))
            if self.counter * heartbeat_period >= self.wait_for_door:
                self.state = 'closed'
                self.counter = 0
                self.log.info('State: close')
            self.counter += 1
        elif self.state == 'closed':
            self.log.info('State: closed, Progress: ' + str(self.counter) + ' / ' + str(self.sample_counts))
            if self.feature_mean is not None:
                if self.counter < self.sample_counts:    
                    if self.door_measurement == 'open':
                        self.P_z_open_given_door_closed += 1
                    else:
                        self.P_z_closed_given_door_closed += 1
                else:
                    self.counter = 0
                    self.state = 'display'
            self.counter += 1
        else: 
            self.log.info('State: display')
            self.P_z_open_given_door_open /= self.sample_counts
            self.P_z_closed_given_door_open /= self.sample_counts
            self.P_z_open_given_door_closed /= self.sample_counts
            self.P_z_closed_given_door_closed /= self.sample_counts
            self.display_probabilities()
            self.timer.cancel()   

    def spin(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    lab3_estimate_probabilities = Lab3EstimateProbabilities()
    lab3_estimate_probabilities.spin()
    lab3_estimate_probabilities.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
