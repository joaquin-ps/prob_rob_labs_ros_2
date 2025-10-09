import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import LinkStates
import math

class MonitorDoorAngle(Node):
    def __init__(self):
        super().__init__('door_angle_monitor')
        self.log = self.get_logger()
        
        # Subscribe to link states
        self.link_states_sub = self.create_subscription(
            LinkStates, '/gazebo/link_states', self.link_states_callback, 1)
        
        self.open_threshold = 2.5

    def link_states_callback(self, msg):
        # Door hinge:
        door_link_name = "hinged_glass_door::door"

        door_index = msg.name.index(door_link_name)
        orientation = msg.pose[door_index].orientation
        
        w, x, y, z = orientation.w, orientation.x, orientation.y, orientation.z
        door_angle = math.atan2(y, x)
        door_angle = abs(door_angle)
        
        if door_angle < self.open_threshold:
            status = "CLOSED"
        elif door_angle > self.open_threshold: 
            status = "OPEN"
        
        self.log.info(f"Door Angle: {door_angle} rad - {status}")
        self.log.info(f"Quaternion: w={w}, x={x}, y={y}, z={z}")
        
def main():
    rclpy.init()
    monitor = MonitorDoorAngle()
    
    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        monitor.log.info("Stopped")
    finally:
        monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
