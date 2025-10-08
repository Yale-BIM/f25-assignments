#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare launch arguments
    look_forward_arg = DeclareLaunchArgument(
        'look_forward',
        default_value='True',
        description='Make shutter look forward?'
    )
    
    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='0.25',
        description='Publish rate for the target generator'
    )
    
    save_state_actions_arg = DeclareLaunchArgument(
        'save_state_actions',
        default_value='False',
        description='Save state actions to file'
    )
    
    output_file_arg = DeclareLaunchArgument(
        'output_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('shutter_behavior_cloning'),
            'data',
            'state_action.txt'
        ]),
        description='Output file for state actions'
    )

    # Include shutter bringup launch file
    shutter_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('shutter_bringup'),
                'launch',
                'shutter.launch.py'
            ])
        ]),
        launch_arguments={
            'simulation': 'true'
        }.items()
    )

    # Generate random target node
    generate_random_target_node = Node(
        package='shutter_behavior_cloning',
        executable='generate_random_target.py',
        name='generate_random_target',
        output='screen',
        parameters=[{
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

    # Expert optimization node
    expert_opt_node = Node(
        package='shutter_behavior_cloning',
        executable='expert_opt.py',
        name='expert_opt',
        output='screen',
        parameters=[{
            'output_file': LaunchConfiguration('output_file'),
            'save_state_actions': LaunchConfiguration('save_state_actions')
        }]
    )

    return LaunchDescription([
        SetParameter(name='use_sim_time', value=True),
        look_forward_arg,
        publish_rate_arg,
        save_state_actions_arg,
        output_file_arg,
        shutter_bringup,
        generate_random_target_node,
        rviz_node,
        look_forward_node,
        expert_opt_node
    ])
