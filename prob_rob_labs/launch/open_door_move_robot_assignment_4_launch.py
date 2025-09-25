import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='set to true for simulation'),
        DeclareLaunchArgument('velocity', default_value='50.0',
                              description='robot forward velocity'),
        Node(
            package='prob_rob_labs',
            executable='open_door_move_robot_assignment_4',
            name='open_door_move_robot_assignment_4',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}, 
                        {'velocity': LaunchConfiguration('velocity')}]
        )
    ])
