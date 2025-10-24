#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64

from message_filters import Subscriber, ApproximateTimeSynchronizer


def yaw_from_quaternion(qx, qy, qz, qw) -> float:
    s = 2.0 * (qw * qz + qx * qy)
    c = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.arctan2(s, c)  # radians


def wrap_to_pi(angle: float) -> float:
    # Wrap to [-pi, pi]
    return np.arctan2(np.sin(angle), np.cos(angle))


class EkfOdometryMetrics(Node):

    def __init__(self):
        super().__init__('ekf_odometry_metrics')
        self.log = self.get_logger()

        # Publishers
        self.pos_err_pub = self.create_publisher(Float64, '/odometry_error/position', 10)
        self.yaw_err_rad_pub = self.create_publisher(Float64, '/odometry_error/yaw_rad', 10)
        self.yaw_err_deg_pub = self.create_publisher(Float64, '/odometry_error/yaw_deg', 10)

        # Time-synced subscribers
        self.gt_sub = Subscriber(self, Odometry, '/odom')
        self.est_sub = Subscriber(self, Odometry, '/ekf_odom')
        self.ts = ApproximateTimeSynchronizer([self.gt_sub, self.est_sub],
                                              queue_size=50, slop=0.05)
        self.ts.registerCallback(self.synced_cb)

        self.log.info('Subscribed to /odom and /ekf_odom; publishing errors under /odometry_error/*')

    def synced_cb(self, gt_msg: Odometry, ekf_msg: Odometry):
        # Position error
        g_pos = np.array([gt_msg.pose.pose.position.x,
                          gt_msg.pose.pose.position.y], dtype=float)
        e_pos = np.array([ekf_msg.pose.pose.position.x,
                          ekf_msg.pose.pose.position.y], dtype=float)
        pos_err = float(np.linalg.norm(e_pos - g_pos))  # meters

        # Yaw error
        gq = gt_msg.pose.pose.orientation
        eq = ekf_msg.pose.pose.orientation
        yaw_gt = yaw_from_quaternion(gq.x, gq.y, gq.z, gq.w)
        yaw_ekf = yaw_from_quaternion(eq.x, eq.y, eq.z, eq.w)

        yaw_err_rad = float(wrap_to_pi(yaw_ekf - yaw_gt))
        yaw_err_deg = float(np.degrees(yaw_err_rad))

        # Publish
        self.pos_err_pub.publish(Float64(data=pos_err))
        self.yaw_err_rad_pub.publish(Float64(data=yaw_err_rad))
        self.yaw_err_deg_pub.publish(Float64(data=yaw_err_deg))

    def spin(self):
        rclpy.spin(self)


def main():
    rclpy.init()
    ekf_odometry_metrics = EkfOdometryMetrics()
    ekf_odometry_metrics.spin()
    ekf_odometry_metrics.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
