#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """
    Launch file to record bag files for shutter_kf package.
    """
    
    # Declare launch arguments
    # Get the package source directory (where launch files, scripts, etc. are located)
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(package_dir, 'data')
    
    path_arg = DeclareLaunchArgument(
        'path',
        default_value=data_dir,
        description='Location for bag file'
    )
    
    # Record bag process
    record_bag_process = ExecuteProcess(
        cmd=[
            'timeout', '300',
            'ros2', 'bag', 'record',
            '/filtered_target',
            '/target', 
            '/future_target',
            '/rosout',
            '-o', LaunchConfiguration('path') + '/track_target.bag'
        ],
        name='rosbag_record_data',
        output='screen'
    )
    
    return LaunchDescription([
        path_arg,
        record_bag_process
    ])
