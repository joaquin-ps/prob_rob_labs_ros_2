import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from sensor_msgs.msg import JointState
from message_filters import Subscriber, ApproximateTimeSynchronizer

from copy import copy

from geometry_msgs.msg import Twist

from nav_msgs.msg import Odometry

import numpy as np

heartbeat_period = 0.05

default_var = 0.25

class EkfOdometry(Node):

    def __init__(self):
        super().__init__('ekf_odometry')
        self.log = self.get_logger()
        
        # ==============================================
        # ROS 2 - Data Collection
        # ==============================================

        # Subscribers
        self.imu_subscriber = Subscriber(self, Imu, '/imu')
        self.joint_state_subscriber = Subscriber(self, JointState, '/joint_states')

        self.cmd_vel_subscriber = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # Synchronizer
        self.ts = ApproximateTimeSynchronizer([self.imu_subscriber, self.joint_state_subscriber], queue_size=10, slop=0.015)
        self.ts.registerCallback(self.data_callback)
        
        # Odometry publisher
        self.ekf_odom_publisher = self.create_publisher(Odometry, '/ekf_odom', 10)

        # Timer for heartbeat
        self.timer = self.create_timer(heartbeat_period, self.heartbeat)

        # Data storage
        self.latest_data = {
            'imu': {
                'timestamp': None,
                'orientation': None,
                'angular_velocity': None,
                'linear_acceleration': None,
                'orientation_covariance': None,
                'angular_velocity_covariance': None,
                'linear_acceleration_covariance': None
            },
            'joint_states': {
                'omega_r': None,
                'omega_l': None   
            }
        }
        self.latest_cmd_vel = (0.0, 0.0)

        # ==============================================
        # EKF Variables
        # ==============================================
        
        # State
        self.state_len = 5
        self.input_len = 2
        self.measurement_len = 3

        self.state = np.zeros((self.state_len, 1))
        self.state_cov = np.identity(self.state_len) * default_var

        self.input = np.zeros((self.input_len, 1))
        self.input_cov = np.identity(self.input_len) * default_var

        self.measurement = np.zeros((self.measurement_len, 1))
        self.measurement_cov = np.identity(self.measurement_len) * default_var

        self.predicted_state = np.zeros((self.state_len, 1))
        self.predicted_state_cov = np.identity(self.state_len) * default_var

        # Parameters 
        self.delta_t = heartbeat_period

        self.r_w = 0.033  # m - Wheel radius
        self.R = 0.1435/2 # m - 1/2 Wheel separation
        
        self.G_v = 1.000448 # Gain linear velocity
        self.G_w = 1.005156 # Gain angular velocity

        self.tau_v = 0.49 # Time constant - linear
        self.tau_w = 0.65 # Time constant - angular

        self.a_v = 0.1 ** (self.delta_t/self.tau_v) # Forgetting factor - linear
        self.a_w = 0.1 ** (self.delta_t/self.tau_w) # Forgetting factor - angular

        # Populate Covariance
        sigma_uv = 0.01
        sigma_uw = 0.01

        self.input_cov[0,0] = sigma_uv
        self.input_cov[1,1] = sigma_uw

        sigma_wr = 0.01
        sigma_wl = 0.01

        self.measurement_cov[0,0] = sigma_wr
        self.measurement_cov[1,1] = sigma_wl

    def get_input(self):
        # Input
        u = np.zeros((self.input_len, 1))

        u_v, u_w  = self.latest_cmd_vel

        u[0, 0] = u_v
        u[1, 0] = u_w

        # Covariance
        u_cov = self.input_cov

        return u, u_cov
        
    def get_measurement(self):
        # Measurement
        z = np.zeros((self.measurement_len, 1))

        omega_r = self.latest_data['joint_states']['omega_r']
        omega_l = self.latest_data['joint_states']['omega_l']
        ang_vel = self.latest_data['imu']['angular_velocity']
        omega_g = ang_vel[2] if ang_vel is not None else 0.0

        z[0, 0] = 0.0 if omega_r is None else omega_r
        z[1, 0] = 0.0 if omega_l is None else omega_l
        z[2, 0] = omega_g

        # Covariance
        z_cov = copy(self.measurement_cov)

        sigma_w_g = self.latest_data["imu"]["angular_velocity_covariance"]
        z_cov[2,2] = 0.0 if sigma_w_g is None else sigma_w_g[2*2] 
        
        return z, z_cov

    def G_x_matrix(self):
        delta_t = self.delta_t

        theta = self.state[0, 0]
        v = self.state[3, 0]

        G_x = np.identity(self.state_len)

        # θ row
        G_x[0, 4] = delta_t

        # x row
        G_x[1, 0] = -v * delta_t * np.sin(theta)
        G_x[1, 3] =  delta_t * np.cos(theta)

        # y row
        G_x[2, 0] =  v * delta_t * np.cos(theta)
        G_x[2, 3] =  delta_t * np.sin(theta)

        # v and ω dynamics (first-order with forgetting factors)
        G_x[3, 3] = self.a_v
        G_x[4, 4] = self.a_w

        return G_x

    def G_u_matrix(self):
        delta_t = self.delta_t

        G_u = np.zeros((self.state_len, 2))

        G_u[3, 0] = self.G_v * (1.0 - self.a_v)
        G_u[4, 1] = self.G_w * (1.0 - self.a_w)
        return G_u

    def H_matrix(self):
        delta_t = self.delta_t
        rw = self.r_w
        R  = self.R

        H = np.zeros((self.measurement_len, self.state_len))

        # ω_r = v/rw + (R/rw)*ω
        H[0, 3] = 1.0 / rw
        H[0, 4] = R / rw

        # ω_l = v/rw - (R/rw)*ω
        H[1, 3] = 1.0 / rw
        H[1, 4] = -R / rw

        # ω_g = ω
        H[2, 4] = 1.0

        return H
    
    def _prediction_model(self):
        x = self.state
        u = self.input

        sigma_x = self.state_cov
        sigma_u = self.input_cov

        self.predicted_state = self.G_x_matrix() @ \
            x + self.G_u_matrix() @ u 

        self.predicted_state_cov = \
            self.G_x_matrix() @ sigma_x @ \
            self.G_x_matrix().transpose() + \
            self.G_u_matrix() @ sigma_u @ \
            self.G_u_matrix().transpose()

    def _measurement_model(self):
        z = self.measurement

        sigma_z = self.measurement_cov

        SigmaXHT =  self.predicted_state_cov @ self.H_matrix().transpose()
        HSigmaXHT = self.H_matrix() @ SigmaXHT
        K = SigmaXHT @ np.linalg.inv(HSigmaXHT + sigma_z)
        self.state = self.predicted_state + K @ \
            (z - self.H_matrix() @ self.predicted_state)
        self.state_cov = (np.identity(self.state_len) - K @ self.H_matrix()) @ \
            self.predicted_state_cov

    def advance_filter(self):
        # Get state
        self.input, self.input_cov = self.get_input()
        self.measurement, self.measurement_cov = self.get_measurement()
        self.get_logger().info(f"Prior: {self.state}")
        
        # Step prediction model
        self._prediction_model()
        self.get_logger().info(f"Prediction: {self.predicted_state}")
        
        # Step measurement model
        self._measurement_model()
        self.get_logger().info(f"Innovation: {self.state}")

    def yaw_to_quat(self, yaw):
        half_yaw = yaw * 0.5
        qx = 0.0
        qy = 0.0
        qz = np.sin(half_yaw)
        qw = np.cos(half_yaw)
        return (qx, qy, qz, qw)

    def heartbeat(self):
        self.log.info('heartbeat')
        self.get_logger().info("*" * 80)
        self.get_logger().info(f"{self.latest_data}")
        
        lin, ang = self.latest_cmd_vel
        self.get_logger().info("*" * 80)
        self.get_logger().info(f"Linear vel: {lin}, Angular vel: {ang}")

        self.get_logger().info("*" * 80)
        self.advance_filter()

        # Publish EKF odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_footprint"

        # Populate message:
        theta = float(self.state[0])
        odom_msg.pose.pose.position.x = float(self.state[1])
        odom_msg.pose.pose.position.y = float(self.state[2])
        odom_msg.twist.twist.linear.x = float(self.state[3])
        odom_msg.twist.twist.angular.z = float(self.state[4])

        qx, qy, qz, qw = self.yaw_to_quat(theta)
        odom_msg.pose.pose.orientation.x = qx
        odom_msg.pose.pose.orientation.y = qy
        odom_msg.pose.pose.orientation.z = qz
        odom_msg.pose.pose.orientation.w = qw

        # Publish
        self.ekf_odom_publisher.publish(odom_msg)
        self.get_logger().info("Published /ekf_odom message")

    def data_callback(self, imu_msg, joint_state_msg):
        # IMU data
        self.latest_data['imu']['timestamp'] = imu_msg.header.stamp
        self.latest_data['imu']['orientation'] = (imu_msg.orientation.x, imu_msg.orientation.y, imu_msg.orientation.z, imu_msg.orientation.w)
        self.latest_data['imu']['angular_velocity'] = (imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z)
        self.latest_data['imu']['linear_acceleration'] = (imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z)
        self.latest_data['imu']['orientation_covariance'] = imu_msg.orientation_covariance
        self.latest_data['imu']['angular_velocity_covariance'] = imu_msg.angular_velocity_covariance
        self.latest_data['imu']['linear_acceleration_covariance'] = imu_msg.linear_acceleration_covariance
        
        # Wheel velocity
        if len(joint_state_msg.velocity) >= 2:
            self.latest_data['joint_states']['omega_r'] = joint_state_msg.velocity[1] 
            self.latest_data['joint_states']['omega_l'] = joint_state_msg.velocity[0]  

    def cmd_vel_callback(self, msg):
        # Store latest velocity command
        self.latest_cmd_vel = (msg.linear.x, msg.angular.z)

    def spin(self):
        rclpy.spin(self)

def main():
    rclpy.init()
    ekf_odometry = EkfOdometry()
    ekf_odometry.spin()
    ekf_odometry.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
