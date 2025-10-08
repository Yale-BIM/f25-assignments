#!/usr/bin/env python3
# Public tests for CPSC459/559 Assignment 2 - Part III

PKG = "shutter_kf_public_tests"
NAME = 'test_filter_quality'

import sys
import unittest
import numpy as np
import time

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.duration import Duration
import ament_index_python
import message_filters
from geometry_msgs.msg import PoseStamped

import launch
import launch_ros
import launch_testing
import launch_testing.actions
from launch_testing.actions import ReadyToTest
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare

def generate_test_description():
    """
    Launch only the system under test. The test node itself runs via `colcon test`.
    This is equivalent to the ROS 1 launch file with <include> and no <test> tags.
    """
    
    # Declare launch arguments
    add_noise_arg = DeclareLaunchArgument(
        'add_noise',
        default_value='true',
        description='Add noise to target observations'
    )
    
    path_type_arg = DeclareLaunchArgument(
        'path_type',
        default_value='horizontal',
        description='Path type for target movement'
    )
    
    # Include shutter_kf follow_target launch file
    follow_target_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('shutter_kf'),
                'launch',
                'follow_target.launch.py'
            ])
        ]),
        launch_arguments={
            'add_noise': LaunchConfiguration('add_noise'),
            'path_type': LaunchConfiguration('path_type')
        }.items()
    )
    
    return (
        LaunchDescription([
            SetParameter(name='use_sim_time', value=True),
            add_noise_arg,
            path_type_arg,
            follow_target_launch,
            TimerAction(
                period=0.5,
                actions=[ReadyToTest()]
            ),
        ]), {}
    )

class TestFilterQuality(unittest.TestCase):
    """
    Public tests for kalman_filter.py
    """

    @classmethod
    def setUpClass(cls):
        """Initialize ROS for the test class."""
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        """Shutdown ROS after the test class."""
        rclpy.shutdown()

    def setUp(self):
        self.filtered_topic = "/filtered_target"    # filtered topic
        self.gt_topic = "/true_target"              # true topic
        self.msg_list = []

        self.node = rclpy.create_node(NAME, parameter_overrides=[Parameter('use_sim_time', value=True)])

    def tearDown(self):
        """Destroy the node after each test."""
        self.node.destroy_node()

    def _callback(self, filtered_msg, gt_msg):
        self.msg_list.append((filtered_msg, gt_msg))
    
    def test_quality(self):
        """
        Check that the filter quality meets the requirements
        """
        # Wait for system to initialize
        timeout_t = time.time() + 2.0
        while rclpy.ok() and time.time() < timeout_t:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        self.msg_list = []

        sub1 = message_filters.Subscriber(self.node, PoseStamped, self.filtered_topic)
        sub2 = message_filters.Subscriber(self.node, PoseStamped, self.gt_topic)
        subs = [sub1, sub2]
        ts = message_filters.ApproximateTimeSynchronizer(subs, 10, 0.25)
        ts.registerCallback(self._callback)
        
        timeout_t = time.time() + 30  # 30 seconds in the future

        # wait patiently for a message
        while rclpy.ok() and time.time() < timeout_t:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        # stop getting messages - subscriptions will be cleaned up when node is destroyed

        # did we succeed in getting messages?
        self.assertGreaterEqual(len(self.msg_list), 10, 
            f"Got less than 10 synchronized messages in 30 secs (num. messages={len(self.msg_list)}).")
        print(f"Got {len(self.msg_list)} synchronized messages.")
        
        # compute error
        err = []
        for i in range(len(self.msg_list)):
            m1 = self.msg_list[i][0]
            m2 = self.msg_list[i][1]
            diffx = m1.pose.position.x - m2.pose.position.x
            diffy = m1.pose.position.y - m2.pose.position.y
            diffz = m1.pose.position.z - m2.pose.position.z
            l2_err = np.sqrt(diffx*diffx + diffy*diffy + diffz*diffz)
            err.append(l2_err)

        avg_err = np.mean(err)
        std_err = np.std(err)
        self.assertLessEqual(avg_err, 0.04, 
            f"The average error {avg_err} (+- {std_err}) was greater than 0.04.")

        print(f"Avg err={avg_err}")
        print(f"Std err={std_err}")
