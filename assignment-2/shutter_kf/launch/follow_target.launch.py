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
        default_value='30',
        description='Publish rate for the target generator'
    )
    
    path_type_arg = DeclareLaunchArgument(
        'path_type',
        default_value='horizontal',
        description='Path type for target movement'
    )
    
    add_noise_arg = DeclareLaunchArgument(
        'add_noise',
        default_value='false',
        description='Add noise to target observations'
    )
    
    follow_target_topic_arg = DeclareLaunchArgument(
        'follow_target_topic',
        default_value='/filtered_target',
        description='Topic for filtered target'
    )
    
    headless_sim_arg = DeclareLaunchArgument(
        'headless_sim',
        default_value='false',
        description='Run simulation in headless mode'
    )
    
    fast_target_arg = DeclareLaunchArgument(
        'fast_target',
        default_value='false',
        description='Make target move faster'
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
            'simulation': 'true',
            'headless': LaunchConfiguration('headless_sim')
        }.items()
    )

    # Generate continuous target node
    generate_continuous_target_node = Node(
        package='shutter_kf',
        executable='generate_continuous_target.py',
        name='generate_continuous_target',
        output='screen',
        parameters=[{
            'publish_rate': LaunchConfiguration('publish_rate'),
            'path_type': LaunchConfiguration('path_type'),
            'add_noise': LaunchConfiguration('add_noise'),
            'fast': LaunchConfiguration('fast_target')
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

    # Kalman filter node
    kalman_filter_node = Node(
        package='shutter_kf',
        executable='kalman_filter.py',
        name='kalman_filter',
        output='screen'
    )

    # Expert optimization node
    expert_opt_node = Node(
        package='shutter_behavior_cloning',
        executable='expert_opt.py',
        name='expert_opt',
        output='screen',
        parameters=[{
            'save_state_actions': False
        }],
        remappings=[
            ('/target', LaunchConfiguration('follow_target_topic'))
        ]
    )

    return LaunchDescription([
        SetParameter(name='use_sim_time', value=True),
        look_forward_arg,
        publish_rate_arg,
        path_type_arg,
        add_noise_arg,
        follow_target_topic_arg,
        headless_sim_arg,
        fast_target_arg,
        shutter_bringup,
        generate_continuous_target_node,
        rviz_node,
        look_forward_node,
        kalman_filter_node,
        expert_opt_node
    ])
