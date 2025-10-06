import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Empty

heartbeat_period = 0.1

class EstimateProbDoorOpen(Node):

    def __init__(self):
        super().__init__('estimate_prob_door_open')
        self.log = self.get_logger()
        self.timer = self.create_timer(heartbeat_period, self.heartbeat)

        # Listen to /hinged_glass_door/torque to determine whether the door has been commanded to close:
        self.torque_sub =  self.create_subscription(
            Float64, '/hinged_glass_door/torque', self.torque_sub_callback, 1)
        
        # Publisher of flaky door open command: 
        self.flaky_open_pub = self.create_publisher(
            Empty, '/door_open', 1
        )
        
        # Variables for data collection:
        self.num_samples = 100
        self.num_total = 0
        self.num_success = 0
        self.num_failure = 0

    def torque_sub_callback(self, msg):
        torque = msg.data
        if torque == 5.0:
            self.log.info("Success")
            self.num_success += 1
        elif torque == 0.0:
            self.log.info("Failure")
            self.num_failure += 1
        else: 
            self.log.info("Unrecognized value")

    def calculate_prob(self):
        self.log.info(str(self.num_success))
        self.log.info(str(self.num_failure))
        self.log.info(str(self.num_total))
        prob_x_open_given_u_open = self.num_success / self.num_total
        return prob_x_open_given_u_open

    def heartbeat(self):
        self.log.info('heartbeat')
        if self.num_total < self.num_samples:
            msg = Empty()
            self.log.info(f"Published command: {self.num_total}")
            self.flaky_open_pub.publish(msg)
            self.num_total += 1
        else:
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
