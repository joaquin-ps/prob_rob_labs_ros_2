import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped

import math

import numpy as np 

heartbeat_period = 0.1

class ErrorCharacterizationDistBear(Node):

    def __init__(self):
        super().__init__('error_characterization_dist_bear')

        ### Ground Truth Values ###
        self.gt_distance = 0.0
        self.gt_bearing = 0.0

        # tb3 pose
        self.tb3_gt_x = None
        self.tb3_gt_y = None
        self.tb3_gt_quat = None

        self.tb3_pose_sub = self.create_subscription(
            PoseStamped, '/tb3/ground_truth/pose', self.tb3_gt_pose_callback, 1
        )

        # landmarks
        self.red_landmark_pos =     np.array([8.5,    -5])
        self.green_landmark_pos =   np.array([8.5,    5])
        self.yellow_landmark_pos =  np.array([-11.5,  5])
        self.magenta_landmark_pos = np.array([-11.5,  -5])
        self.cyan_landmark_pos =    np.array([0,      0])

        ### Estimated Values ###
        self.est_distance = None
        self.est_bearing = None

        self.distance_sub = self.create_subscription(
            Float64, '/distance', self.estimated_distance_callback, 1
        )
        self.bearing_sub = self.create_subscription(
            Float64, '/bearing', self.estimated_bearing_callback, 1
        )
        
        self.log = self.get_logger()
        self.timer = self.create_timer(heartbeat_period, self.heartbeat)

    def tb3_gt_pose_callback(self, msg):
        pose = msg.pose
        
        self.tb3_gt_x = pose.position.x
        self.tb3_gt_y = pose.position.y
        self.tb3_gt_quat = pose.orientation

    def get_gt_distance(self):
        tb3_pos = np.array([self.tb3_gt_x, self.tb3_gt_y])
        landmark_pos = self.cyan_landmark_pos

        return np.linalg.norm(tb3_pos - landmark_pos)

    def get_gt_bearing(self):
        # Landmark to robot
        dx = self.cyan_landmark_pos[0] - self.tb3_gt_x
        dy = self.cyan_landmark_pos[1] - self.tb3_gt_y
        phi = math.atan2(dy, dx)

        # Robot orientation
        q = self.tb3_gt_quat
        w, x, y, z = q.w, q.x, q.y, q.z
        yaw_robot = yaw_from_quat(w, x, y, z)

        # Bearing
        return wrap_to_pi(phi - yaw_robot)

    def estimated_distance_callback(self, msg):
        self.est_distance = msg.data
    
    def estimated_bearing_callback(self, msg):
        self.est_bearing = msg.data

    def calculate_error(self):
        bearing_error  = self.gt_bearing - self.est_bearing
        distance_error = self.gt_distance - self.est_distance

        return bearing_error, distance_error

    def heartbeat(self):
        values = [self.tb3_gt_x, self.tb3_gt_y, self.tb3_gt_quat, self.est_distance, self.est_bearing]
        self.log.info('*'*50)

        if all(value is not None for value in values): 
            self.gt_distance = self.get_gt_distance()
            self.gt_bearing = self.get_gt_bearing()

            self.get_logger().info(f'Ground Truth Distance: {self.gt_distance}')
            self.get_logger().info(f'Ground Truth Bearing:  {self.gt_bearing}')
            self.get_logger().info(f'Estimated Distance:    {self.est_distance}')
            self.get_logger().info(f'Estimated Bearing:     {self.est_bearing}')

            bearing_error, distance_error = self.calculate_error()

            self.log.info('-'*50)
            self.get_logger().info(f'Error Distance:        {distance_error}')
            self.get_logger().info(f'Error Bearing:         {bearing_error}')

        self.log.info('*'*50)
        
    def spin(self):
        rclpy.spin(self)

def wrap_to_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def yaw_from_quat(w, x, y, z):
    # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main():
    rclpy.init()
    error_characterization_dist_bear = ErrorCharacterizationDistBear()
    error_characterization_dist_bear.spin()
    error_characterization_dist_bear.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
