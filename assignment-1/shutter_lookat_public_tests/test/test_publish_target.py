#!/usr/bin/env python3
# Public tests for CPSC459/559 Assignment 2 - Part II

PKG = "shutter_lookat_public_tests"
NAME = 'test_publish_target'

import sys
import unittest
import tf2_ros

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
import pytest
import time

from utils import inspect_rostopic_info

import launch
import launch_ros
import launch_testing
import launch_testing.actions
from launch_testing.actions import ReadyToTest
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare

def generate_test_description():
    # Declare launch arguments

    simulation_arg = DeclareLaunchArgument(
        'simulation',
        default_value='true',
        description='Run in simulation mode'
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
        launch_arguments={'simulation': LaunchConfiguration('simulation')}.items()
    )

    # Look forward node
    look_forward_node = Node(
        package='shutter_lookat',
        executable='look_forward.py',
        name='look_forward'
    )

    # Generate target node
    generate_target_node = Node(
        package='shutter_lookat',
        executable='generate_target.py',
        name='generate_target',
        output='screen',
        parameters=[{
            'x_value': 1.5,
            'radius': 0.05,
            'publish_rate': 30
        }]
    )

    # Student's code node
    publish_target_node = Node(
        package='shutter_lookat',
        executable='publish_target_relative_to_realsense_camera.py',
        name='publish_target_relative_to_realsense_camera'
    )

    return (
        LaunchDescription(
            [
                SetParameter(name='use_sim_time', value=True),
                simulation_arg,
                shutter_launch,
                look_forward_node,
                generate_target_node,
                publish_target_node,
                TimerAction(
                    period=0.5,
                    actions=[ReadyToTest()]
                ),
            ]
        ), {},
    )


class TestPublishTarget(unittest.TestCase):
    """
    Public tests for publish_target_relative_to_realsense_camera.py
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
        """Create a node for each test."""
        self.node = rclpy.create_node(NAME, parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.node_name = "/publish_target_relative_to_realsense_camera"   # name of the node when launched for the tests
        self.target_topic = "/target"                                    # target topic
        self.target_frame = "target"                                     # target frame

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)

    def tearDown(self):
        """Destroy the node after each test."""
        self.node.destroy_node()
        
    def test_node_connections(self):
        """
        Check the node's connections
        """
        success = False
        timeout_t = time.time() + 5  # 5 seconds in the future
        
        while rclpy.ok() and not success and time.time() < timeout_t:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if inspect_rostopic_info(self.node_name):
                success = True
                break
            time.sleep(0.1)

        self.assertTrue(success, "Failed to verify that the node publish_target_relative_to_realsense_camera.py "
                                 "subscribes to the {} topic".format(self.target_topic))
        print("Verified that the publish_target_relative_to_realsense_camera.py node subscribes to the {} topic".
              format(self.target_topic))

    def test_frame_exists(self):
        """
        Check that the target frame exists in the tf tree
        """
        t = None        # transform
        err = None      # error
        timeout_t = time.time() + 15  # 15 seconds in the future
        exceeded_time = False

        # wait patiently for a transform
        while rclpy.ok() and t is None:
            rclpy.spin_once(self.node, timeout_sec=1.0)
            try:
                t = self.tf_buffer.lookup_transform("base_link",
                                                    self.target_frame,
                                                    rclpy.time.Time(),
                                                    timeout=rclpy.duration.Duration(seconds=1.0))  # wait for 1 second
            except tf2_ros.LookupException as e:
                err = e
                continue
            except tf2_ros.ConnectivityException as e:
                err = e
                continue
            except tf2_ros.ExtrapolationException as e:
                err = e
                continue

            if time.time() > timeout_t:
                exceeded_time = True
                break

        if err is not None:
            err_str = "Got error: {}".format(err)
        else:
            err_str = ""

        self.assertIsNotNone(t, f"Failed to find a transformation between base_link and {self.target_frame} (waited for {time.time()-timeout_t}" \
                                " secs / timeout: {exceeded_time}).{err_str}\nCheck how the node " \
                                "is publishing the transform for the target.")

        print("Success. Found the frame {} in the tf tree!".format(self.target_frame))
