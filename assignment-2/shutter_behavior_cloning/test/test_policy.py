#!/usr/bin/env python3
# Public tests to evaluate behavior cloning model

PKG = "shutter_behavior_cloning"
NAME = 'test_eval_policy'

import sys
import os
import numpy as np
import unittest
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.duration import Duration
import ament_index_python
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from threading import Lock
import tf2_ros
from visualization_msgs.msg import Marker
import uuid
import time

shutter_bc = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(shutter_bc, "src"))

from generate_random_target import dflt_x_min, dflt_x_max, dflt_y_min, dflt_y_max, dflt_z_min, dflt_z_max
from expert_opt import ExpertNode, transform_msg_to_T, make_joint_rotation, target_in_camera_frame

import launch
import launch_ros
import launch_testing
import launch_testing.actions
from launch_testing.actions import ReadyToTest
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction, LogInfo, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare

def generate_test_description():
    # Declare launch arguments
    run_expert_arg = DeclareLaunchArgument(
        'run_expert',
        default_value='False',
        description='Run expert instead of learned model'
    )
    
    run_rviz_arg = DeclareLaunchArgument(
        'run_rviz',
        default_value='True',
        description='Run RViz visualization'
    )
    
    model_arg = DeclareLaunchArgument(
        'model',
        default_value=os.environ.get('MODEL', ''),
        description='Path to model file'
    )
    
    normp_arg = DeclareLaunchArgument(
        'normp',
        default_value=os.environ.get('NORMP', ''),
        description='Path to normalization parameters file'
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

    # RViz node (conditional)
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('shutter_lookat'),
            'config',
            'rviz_target.cfg.rviz'
        ])],
        condition=IfCondition(LaunchConfiguration('run_rviz'))
    )

    # Look forward node
    look_forward_node = Node(
        package='shutter_behavior_cloning',
        executable='look_forward.py',
        name='look_forward'
    )

    # Expert optimization node (conditional)
    expert_opt_node = Node(
        package='shutter_behavior_cloning',
        executable='expert_opt.py',
        name='expert_opt',
        output='screen',
        parameters=[{
            'save_state_actions': False
        }],
        condition=IfCondition(LaunchConfiguration('run_expert'))
    )

    # Run policy node (conditional)
    run_policy_node = Node(
        package='shutter_behavior_cloning',
        executable='run_policy.py',
        name='run_policy',
        output='screen',
        parameters=[{
            'model': LaunchConfiguration('model'),
            'norm_params': LaunchConfiguration('normp')
        }],
        condition=UnlessCondition(LaunchConfiguration('run_expert'))
    )

    return (
        LaunchDescription([
            SetParameter(name='use_sim_time', value=True),
            run_expert_arg,
            run_rviz_arg,
            SetParameter(name='publish_marker', value=LaunchConfiguration('run_rviz')),
            model_arg,
            normp_arg,
            shutter_bringup,
            rviz_node,
            look_forward_node,
            expert_opt_node,
            run_policy_node,
            TimerAction(
                period=0.5,
                actions=[ReadyToTest()]
            ),
        ]), {}
    )

def make_pose(x, y, z, node, frame_id="base_footprint"):
    """
    Generate random target within a 3D space
    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    :param node: ROS 2 node for getting current time
    :param frame_id: frame id
    :return PoseStamped with (x,y,z) position
    """
    pose_msg = PoseStamped()
    pose_msg.header.stamp = node.get_clock().now().to_msg()
    pose_msg.header.frame_id = frame_id
    pose_msg.pose.position.x = x
    pose_msg.pose.position.y = y
    pose_msg.pose.position.z = z
    pose_msg.pose.orientation.w = 1.0
    return pose_msg


def move_angle_to_pi_range(a):
    """
    Move angle to [pi, -pi) range
    :param a: angle (in radians)
    :return: radians
    """
    result = a
    if a >= 2 * np.pi:
        result = a - np.floor(a / (2 * np.pi)) * 2 * np.pi
    elif a <= -2 * np.pi:
        result = a + np.floor(-a / (2 * np.pi)) * 2 * np.pi

    if result <= -np.pi:
        result = 2 * np.pi + result
    elif result > np.pi:
        result = -2 * np.pi + result

    return result


def min_angle_diff(a1, a2):
    """
    Compute minimum angle difference (a1 - a2)
    :param a1: angle (radians)
    :param a2: angle (radians)
    :return: a1 - a2
    """
    a1 = move_angle_to_pi_range(a1)
    a2 = move_angle_to_pi_range(a2)
    d = a1 - a2
    if d > np.pi:
        d = d - 2 * np.pi
    elif d < -np.pi:
        d = d + 2 * np.pi
    return d


def check_joint_moved(joints_list, threshold=1e-5):
    """
    Check if a joint changed position
    :param joints_list: list of joint angles
    :param threshold: threshold for comparison in radians
    """
    if len(joints_list) < 2:
        return False

    # compare each consecutive set of joints in the list
    j = joints_list[0]
    for i in range(1, len(joints_list)):
        current_joint = joints_list[i]
        diff = min_angle_diff(j, current_joint)
        if np.fabs(diff) > threshold:
            return True
        j = current_joint

    return False


class TestPolicy(unittest.TestCase):
    """
    Public tests for run_policy.py
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
        self.node = ExpertNode(NAME)
        self.node.set_parameters([Parameter('use_sim_time', value=True)])
        
        self.node.declare_parameter("max_wait", 3)  # Reduced wait time for faster testing
        self.node.max_wait = self.node.get_parameter("max_wait").value
        # Read the global publish_marker parameter set by the launch file

        self.node.output_file = os.path.join(os.path.expanduser("~/ros2_ws"), "test_policy_output.txt")

        self.node.publish_marker = True

        # prepare output file
        self.node.fid = open(self.node.output_file, 'w')

        # get robot model
        self.node.move = False

        # make positions repeatable
        np.random.seed(0)

        # generate samples
        num_samples = 25
        self.positions = np.column_stack((np.random.uniform(low=dflt_x_min, high=dflt_x_max, size=(num_samples, 1)),
                                          np.random.uniform(low=dflt_y_min, high=dflt_y_max, size=(num_samples, 1)),
                                          np.random.uniform(low=dflt_z_min, high=dflt_z_max, size=(num_samples, 1))))

        # buffer for joint positions
        self.record_joints = False
        self.joints_1 = []
        self.joints_3 = []
        self.joints_mutex = Lock()

        # error codes
        self.err = {"NO_EXPERT_SOLUTION": "Failed to compute the expert's solution",
                    "NO_JOINT_STATES": "Failed to get new joint states for the robot.",
                    "NO_MOTION": "The robot did not move."}

        # setup node connections
        self.node.create_subscription(JointState, '/joint_states', self._joints_callback, 5)

        timeout_t = time.time() + 3.0
        while rclpy.ok() and time.time() < timeout_t:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        # tf subscriber - use ROS 2 TF2
        self.node.tf_buffer = tf2_ros.Buffer()
        self.node.tf_listener = tf2_ros.TransformListener(self.node.tf_buffer, self.node)

        # wait for key joints
        timeout_t = time.time() + 60.0
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=1.0)
            try:
                self.node.tf_buffer.lookup_transform("base_link", "base_footprint", rclpy.time.Time(), timeout=Duration(seconds=1.0))
                self.node.tf_buffer.lookup_transform("camera_color_optical_frame", "base_link", rclpy.time.Time(), timeout=Duration(seconds=1.0)) # wait for 1 second
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                continue

            if time.time() > timeout_t:
                self.fail("Failed to get transforms in the robot body.")
                break

        self.node.target_pub = self.node.create_publisher(PoseStamped, '/target', 5)
        self.node.marker_pub = self.node.create_publisher(Marker, '/target_marker', 5)

        self.node.get_logger().info("Waiting for a few seconds so that all other nodes start...")
        timeout_t = time.time() + 3.0
        while rclpy.ok() and time.time() < timeout_t:
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def tearDown(self):
        """Destroy the node after each test."""
        self.node.cleanup()
        self.node.destroy_node()

    def _joints_callback(self, msg):
        """
        Joints callback
        :param msg: joint state
        """
        joint1_idx = -1
        joint2_idx = -1
        joint3_idx = -1
        joint4_idx = -1
        for i in range(len(msg.name)):
            if msg.name[i] == 'joint_1':
                joint1_idx = i
            elif msg.name[i] == 'joint_2':
                joint2_idx = i
            elif msg.name[i] == 'joint_3':
                joint3_idx = i
            elif msg.name[i] == 'joint_4':
                joint4_idx = i
        assert joint1_idx >= 0 and joint2_idx >= 0 and joint3_idx >= 0 and joint4_idx >= 0, \
            "Missing joints from joint state! joint1 = {}, joint2 = {}, joint3 = {}, joint4 = {}".\
                format(joint1_idx, joint2_idx, joint3_idx, joint4_idx)
                
        with self.joints_mutex:
            if not self.record_joints:  # empty the queue so that it's only size 1; otherwise, grow the queue
                self.joints_1 = []
                self.joints_3 = []

            # store joint
            self.joints_1.append(msg.position[joint1_idx])
            self.joints_3.append(msg.position[joint3_idx])

    def publish_marker_msg(self, pose_msg):
        # publish a marker to visualize the target in RViz
        marker_msg = Marker()
        marker_msg.header = pose_msg.header
        marker_msg.action = Marker.ADD
        marker_msg.color.a = 0.5
        marker_msg.color.b = 1.0
        marker_msg.lifetime.sec = 20
        marker_msg.lifetime.nanosec = 0
        marker_msg.id = 0
        marker_msg.ns = "target"
        marker_msg.type = Marker.SPHERE
        marker_msg.pose = pose_msg.pose
        marker_msg.scale.x = 0.1
        marker_msg.scale.y = 0.1
        marker_msg.scale.z = 0.1
        self.node.marker_pub.publish(marker_msg)

    def publish_sample(self, x, y, z):
        """
        Helper function to publish a sample an evaluate the motion of the robot
        :param x: x coordinate
        :param y: y coordinate
        :param z: z coordinate
        :return: abs diff j1, abs diff j2, str error code
        """
        # ensure we are not recording joints
        with self.joints_mutex:
            self.record_joints = False

        # make target
        pose_msg = make_pose(x, y, z, self.node)

        # get solution from expert
        self.joint1 = self.joints_1[-1]
        self.joint3 = self.joints_3[-1]
        solution = self.node.compute_joints_position(pose_msg, self.joint1, self.joint3)
        if solution is None:
            self.node.get_logger().error("Failed to compute expert's solution for target {}".format((x, y, z)))
            return None, None, self.err['NO_EXPERT_SOLUTION']
        j1, j3 = solution

        # publish new target
        self.node.target_pub.publish(pose_msg)
        if self.node.publish_marker:
            self.publish_marker_msg(pose_msg)

        # start saving joints
        with self.joints_mutex:
            self.record_joints = True

        # wait until for max duration for the robot to reach a new pose
        timeout_t = time.time() + self.node.max_wait
        while rclpy.ok() and time.time() < timeout_t:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        # analyze result... Did the robot reach the expert's solution?
        if len(self.joints_1) < 1 or len(self.joints_3) < 1:
            return None, None, self.err['NO_JOINT_STATES']

        # j1_moved = check_joint_moved(self.joints_1)
        # j3_moved = check_joint_moved(self.joints_3)

        diff1 = np.fabs(min_angle_diff(j1, self.joints_1[-1]))
        diff3 = np.fabs(min_angle_diff(j3, self.joints_3[-1]))

        return diff1, diff3, None

    def test_reaching_targets(self, max_ang_difference=0.0261799):
        """
        Test reaching the targets
        :param max_ang_difference: maximum angular difference to consider a trial successful (in radians, dflt: 0.5 deg)
        """
        headers = ["TRIAL", "X", "Y", "Z", "DIFFJ1", "DIFFJ3", "ACCEPTABLE"]
        self.node.fid.write("".join([x.ljust(12) for x in headers]) + "\n")

        for i in range(self.positions.shape[0]):
            p = self.positions[i]
            d1, d3, err = self.publish_sample(p[0], p[1], p[2])
            self.assertIsNone(err, "An error occurred while testing the model: {}".format(err))

            if d1 < max_ang_difference and d3 < max_ang_difference:
                acceptable = 1
            else:
                acceptable = 0

            # print info...
            table = ["%4d" % i,
                     "%.3f" % p[0],
                     "%.3f" % p[1],
                     "%.3f" % p[2],
                     "%.5f" % d1,
                     "%.5f" % d3,
                     "%1d" % acceptable]
            self.node.fid.write("".join([x.ljust(12) for x in table]) + "\n")
            self.node.fid.flush()
