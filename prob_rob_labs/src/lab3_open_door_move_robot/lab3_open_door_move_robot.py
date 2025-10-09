import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64, Empty
from geometry_msgs.msg import Twist

import numpy as np

heartbeat_period = 5 # Slow down from 0.1 to 0.5 for lab 3 assignment 8

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
        self.door_open_pub = self.create_publisher(
            Empty, '/door_open', 1)
        
        # Door state:
        self.feature_mean_sub = self.create_subscription(
            Float64, '/feature_mean', self.feature_mean_callback, 1)
        self.feature_mean_threshold = 236.5
        self.feature_mean = None
        self.door_state = 'unknown'  # 'open', 'closed', 'unknown'

        # Initial conditions:
        self.belief = np.array([[0.5], [0.5]])  # P(door_open), P(door_closed)
        self.belief_bar = np.zeros((2,1))

        # Measurement Probabilities: 
        self.P_z_open_given_door_open = 0.8465
        self.P_z_closed_given_door_open = 0.1535
        self.P_z_open_given_door_closed = 0.0075
        self.P_z_closed_given_door_closed = 0.9925

        # Prediction Probabilities:
        self.P_x_next_open_x_prev_open_given_u_open = 1.0
        self.P_x_next_closed_x_prev_open_given_u_open = 0.0
        self.P_x_next_open_x_prev_closed_given_u_open = 0.2
        self.P_x_next_closed_x_prev_closed_given_u_open = 0.8

        # Measurement Model: 
        self.measurement_model = np.array(
            [[self.P_z_open_given_door_open, self.P_z_open_given_door_closed],
            [self.P_z_closed_given_door_open, self.P_z_closed_given_door_closed]]
        )

        # Prediction Model:
        self.action = 'open' # 'open', 'move'
        self.move_threshold = 0.99
        
        self.pred_model_u_open = np.array(
            [[self.P_x_next_open_x_prev_open_given_u_open, self.P_x_next_open_x_prev_closed_given_u_open],
            [self.P_x_next_closed_x_prev_open_given_u_open, self.P_x_next_closed_x_prev_closed_given_u_open]]
        )
        self.pred_model_u_move_robot = np.array(
            [[1.0, 0.0],
             [0.0, 1.0]]
        )

        # State machine variables:
        self.counter = 0
        self.state = 'init'
        
        # Data collection for experiment results:
        self.experiment_data = []
        
        self.declare_parameter('velocity', 50.0)
        self.velocity = self.get_parameter('velocity').get_parameter_value().double_value

        self.log.info("="*40)
        self.log.info(f'Robot velocity set to {self.velocity}')

        self.time_to_open = 2
        self.time_to_enter = 4

    def _predict(self, action):
        if action == 'open':
            pred_model = self.pred_model_u_open
        else:
            pred_model = self.pred_model_u_move_robot
        self.belief_bar = pred_model @ self.belief

    def _innovation(self, measurement):
        m = measurement
        unnormalized_posterior = \
            (self.measurement_model * np.repeat(self.belief_bar, 2, axis=1).T)[m]
        posterior = unnormalized_posterior / np.sum(unnormalized_posterior)
        self.belief = posterior.reshape(2,1)

    def _update_action_state_machine(self):
        if self.belief[0] < self.move_threshold:
            self.action = 'open'
            self.state = 'open_door'
        else:
            self.action = 'move'
            self.state = 'move_robot'
    
    def bayes_filter_step(self, measurement):
        prior = self.belief.copy()
        
        # Choose action:
        self._update_action_state_machine()
        action = self.action

        # Predict:
        self._predict(action)
        prediction = self.belief_bar.copy()

        # Update:
        self._innovation(measurement)
        posterior = self.belief.copy()

        return prior, action, prediction, posterior

    def heartbeat(self): 
        self.log.info('heartbeat')
        self.log.info('++++ state machine state: ' + self.state)

        if self.state == 'init':    
            self.state = 'open_door'
        elif self.state == 'open_door':
            self.open_door()
            measurement = self._measure_door_state()
            prior, action, prediction, posterior = self.bayes_filter_step(measurement)
            self.log.info(f'measurement: {"open" if measurement == 0 else "closed"}')
            self.log.info(f'prior: {prior[0,0]}')
            self.log.info(f'action: {action}')
            self.log.info(f'prediction: {prediction[0,0]}')
            self.log.info(f'posterior: {posterior[0,0]}')
            
            # Collect data for experiment results
            self.experiment_data.append({
                'index': len(self.experiment_data),
                'prior': prior[0,0],
                'action': action,
                'prediction': prediction[0,0],
                'posterior': posterior[0,0],
                'measurement': measurement,
                'door_state': self.door_state
            })

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
            self._print_experiment_results()
            self.timer.cancel()   

    def _measure_door_state(self):
        if self.door_state == 'open':
            return 0
        elif self.door_state == 'closed':
            return 1
        else:
            return None

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
        # Use flaky door opener
        msg = Empty()
        self.door_open_pub.publish(msg)
        self.log.info("Sent door open command")

    def close_door(self):
        # Use torque command directly
        torque = Float64()
        torque.data = -20.0
        self.torque_pub.publish(torque)
        self.log.info('applying reverse torque to door')

    def move_robot(self):
        self.pub_twist(self.velocity, 0.0)
        self.log.info('moving robot forward')

    def stop_robot(self):
        self.pub_twist(0.0, 0.0)
        self.log.info('stopping robot')

    def pub_twist(self, linear_x, angular_z):
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.robot_cmd_pub.publish(twist)

    def spin(self):
        rclpy.spin(self)

    """ Auxilliary code """
    def _print_experiment_results(self):
        """ Code for printing results in table format, AI assistance used for formatting."""
        if not self.experiment_data:
            self.log.info("No experiment data collected.")
            return
            
        self.log.info("="*80)
        self.log.info("EXPERIMENT RESULTS TABLE")
        self.log.info("="*80)
        
        # Print header
        header = f"{'Index':<6} {'Prior':<8} {'Action':<8} {'Prediction':<12} {'Posterior':<10} {'Measurement':<12} {'Door State':<12}"
        self.log.info(header)
        self.log.info("-" * 80)
        
        # Print data rows
        for data in self.experiment_data:
            measurement_str = str(data['measurement']) if data['measurement'] is not None else 'None'
            row = f"{data['index']:<6} {data['prior']:<8.4f} {data['action']:<8} {data['prediction']:<12.4f} {data['posterior']:<10.4f} {measurement_str:<12} {data['door_state']:<12}"
            self.log.info(row)
        
        self.log.info("="*80)
        self.log.info(f"Total measurements collected: {len(self.experiment_data)}")
        self.log.info("="*80)


def main():
    rclpy.init()
    open_door_move_robot = OpenDoorMoveRobot()
    open_door_move_robot.spin()
    open_door_move_robot.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
