import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Empty
from gazebo_msgs.msg import LinkStates
import math

heartbeat_period = 0.1

class EstimateProbDoorOpen(Node):

    def __init__(self):
        super().__init__('estimate_prob_door_open')
        self.log = self.get_logger()
        self.timer = self.create_timer(heartbeat_period, self.heartbeat)

        # Subscribe to link states for door state ground truth
        self.link_states_sub = self.create_subscription(
            LinkStates, '/gazebo/link_states', self.link_states_callback, 1
        )
        
        # Publisher of flaky door open command
        self.flaky_open_pub = self.create_publisher(
            Empty, '/door_open', 1
        )
        
        # Publisher to close door
        self.torque_pub = self.create_publisher(
            Float64, '/hinged_glass_door/torque', 1
        )
        
        # Variables for data collection
        self.num_samples = 100
        self.num_total = 0
        self.num_success = 0
        self.num_failure = 0
        
        # State machine variables
        self.state = 'close_door'  # 'start_trial', 'wait_for_result', 'closing_door', 'repeat_commands', 'done'
        self.state_counter = 0
        self.trial_time = 2.0  
        self.close_time = 2.0  
        
        # Command sending variables
        self.hold_torque_steps = 11 # 11 because 10 hold steps + 1 reset step in flaky opener logic
        
        # Door state tracking
        self.door_angle = 0.0
        self.door_open_threshold = 2.5  

    def link_states_callback(self, msg):
        # Extract door hinge angle
        door_link_name = "hinged_glass_door::door"
        try:
            door_index = msg.name.index(door_link_name)
            orientation = msg.pose[door_index].orientation
            
            w, x, y, z = orientation.w, orientation.x, orientation.y, orientation.z
            door_angle = math.atan2(y, x)
            self.door_angle = abs(door_angle)
            
        except ValueError:
            self.log.warn("Door link not being published")
            pass

    def calculate_prob(self):
        self.log.info(f"Successes: {self.num_success}")
        self.log.info(f"Failures: {self.num_failure}")
        self.log.info(f"Total trials: {self.num_total}")
        if self.num_total > 0:
            prob_x_open_given_u_open = self.num_success / self.num_total
            return prob_x_open_given_u_open
        return 0.0

    def is_door_open(self):
        return self.door_angle > self.door_open_threshold
    
    def is_door_closed(self):
        return self.door_angle < self.door_open_threshold

    def close_door(self):
        torque_msg = Float64()
        torque_msg.data = -100.0
        self.torque_pub.publish(torque_msg)
        self.log.info("Applied closing torque to door")

    def stop_door(self):
        torque_msg = Float64()
        torque_msg.data = 0.0
        self.torque_pub.publish(torque_msg)
        self.log.info("Stopped applying torque to door")

    def send_door_open_command(self):
        msg = Empty()
        self.flaky_open_pub.publish(msg)
        self.log.info("Sent door open command")

    def heartbeat(self):
        
        if self.state == 'close_door':
            self.close_door()
            self.log.info("Closing door..." + str(self.state_counter * heartbeat_period) + " / " + str(self.close_time))
            self.state_counter += 1
            if self.state_counter * heartbeat_period >= self.close_time:
                self.stop_door()
                self.state = 'start_trial'
                self.log.info("+++ Closed door +++")
                self.state_counter = 0

        elif self.state == 'start_trial':
            if self.num_total >= self.num_samples:
                self.state = 'done'
            else:
                self.log.info("Starting trial " + str(self.num_total + 1))
                self.send_door_open_command()
                self.state = 'wait_for_result'
                self.state_counter = 0

        elif self.state == 'wait_for_result':
            self.state_counter += 1
            if self.state_counter * heartbeat_period <= self.trial_time:
                self.log.info("Waiting for result..." + str(self.state_counter * heartbeat_period) + " / " + str(self.trial_time))
            else:
                if self.is_door_open():
                    self.log.info("Door opened")
                    self.state = 'repeat_commands'
                    self.state_counter = 0
                    self.num_success += 1
                else:
                    self.log.info("Door not opened")
                    self.state = 'start_trial'
                    self.num_failure += 1
                self.num_total += 1

        elif self.state == 'repeat_commands':
            self.log.info("Resetting flaky door opener...")
            if self.state_counter < self.hold_torque_steps:
                self.send_door_open_command()
                self.state_counter += 1
            else:
                self.state = 'close_door'
                self.state_counter = 0

        elif self.state == 'done':
            self.log.info("Finished collecting data")
            prob = self.calculate_prob()
            self.log.info(f"P(x=open|u=open) = {prob}")
            self.timer.cancel()

    def spin(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    estimate_prob_door_open = EstimateProbDoorOpen()
    estimate_prob_door_open.spin()
    estimate_prob_door_open.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
