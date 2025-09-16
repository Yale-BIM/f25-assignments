#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    # Declare launch arguments
    look_forward_arg = DeclareLaunchArgument(
        'look_forward',
        default_value='True',
        description='Make shutter look forward?'
    )
    
    target_x_plane_arg = DeclareLaunchArgument(
        'target_x_plane',
        default_value='1.5',
        description='X plane position for target'
    )
    
    target_radius_arg = DeclareLaunchArgument(
        'target_radius',
        default_value='0.05',
        description='Radius of the target'
    )
    
    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='30',
        description='Publishing rate for target'
    )

    # Include the shutter launch file
    shutter_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('shutter_bringup'),
                'launch',
                'shutter.launch.py'
            ])
        ]),
        launch_arguments={'simulation': 'true'}.items()
    )

    # Joint state publisher node
    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen'
    )

    # Generate target node
    generate_target_node = Node(
        package='shutter_lookat',
        executable='generate_target.py',
        name='generate_target',
        output='screen',
        parameters=[{
            'x_value': LaunchConfiguration('target_x_plane'),
            'radius': LaunchConfiguration('target_radius'),
            'publish_rate': LaunchConfiguration('publish_rate')
        }]
    )
    
    # RViz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('shutter_lookat'),
            'config',
            'rviz_target.cfg.rviz'
        ])]
    )

    # Look forward node (conditional)
    look_forward_node = Node(
        package='shutter_lookat',
        executable='look_forward.py',
        name='look_forward',
        condition=IfCondition(LaunchConfiguration('look_forward'))
    )

    return LaunchDescription([
        SetParameter(name='use_sim_time', value=True),
        look_forward_arg,
        target_x_plane_arg,
        target_radius_arg,
        publish_rate_arg,
        # joint_state_publisher_node,
        shutter_launch,
        generate_target_node,
        look_forward_node,
        rviz_node,
    ])
