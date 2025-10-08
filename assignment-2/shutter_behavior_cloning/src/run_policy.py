#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.parameter_client import AsyncParameterClient
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from urdf_parser_py.urdf import URDF
from std_msgs.msg import Float64MultiArray

import torch
import numpy as np
import sys
import os

class RunPolicyNode(Node):
    """
    Node that controls the robot to make it point towards a target.
    """

    def __init__(self):
        super().__init__('expert_opt')

        # params
        self.declare_parameter("model", "")
        self.declare_parameter("norm_params", "")

        self.model_file = self.get_parameter("model").value              # required path to model file
        self.normp_file = self.get_parameter("norm_params").value        # optional path to normalization parameters (empty str means no norm params)

        # TODO - complete the line below to load up your model and create any necessary class instance variables
        self.model = ...

        # joint values        
        self.current_pose = None #[0.0, 0.0, 0.0, 0.0]

        # get robot model
        self.parameter_client = AsyncParameterClient(self, "/robot_state_publisher")
        if not self.parameter_client.wait_for_services(timeout_sec=5.0):
            self.get_logger().error("Parameter client service not available")
            return
        future = self.parameter_client.get_parameters(['robot_description'])
        rclpy.spin_until_future_complete(self, future)
        
        urdf_xml = future.result().values[0].string_value
        self.robot = URDF.from_xml_string(urdf_xml)

        # joint publisher
        self.joint_pub = self.create_publisher(Float64MultiArray, "/unity_joint_group_controller/command", 5)

        # joint subscriber
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joints_callback, 5)
        self.target_sub = self.create_subscription(PoseStamped, '/target', self.target_callback, 5)

    def joints_callback(self, msg):
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
        self.current_pose = [msg.position[joint1_idx],
                             msg.position[joint2_idx],
                             msg.position[joint3_idx],
                             msg.position[joint4_idx]]
        
    def compute_joints_position(self, msg, joint1, joint3):
        """
        Helper function to compute the required motion to make the robot's camera look towards the target
        :param msg: target message that was received by the target callback
        :param joint1: current joint 1 position
        :param joint3: current joint 3 position
        :return: tuple with new joint positions for joint1 and joint3, or None if the computation failed
        """

        # TODO - remove None return statement and complete function with the logic that runs your model to compute new
        # joint positions 1 & 3 for the robot...
        return None

    def target_callback(self, msg):
        """
        Target callback
        :param msg: target message
        """
        if self.current_pose is None:
            self.get_logger().warn("Joint positions are unknown. Waiting to receive joint states.")
            return
        
        # check that the data is consistent with our model and that we have current joint information...
        if msg.header.frame_id != "base_footprint":
            self.get_logger().error("Expected the input target to be in the frame 'base_footprint' but got the {} frame instead. "
                         "Failed to command the robot".format(msg.header.frame_id))
            return

        # compute the required motion to make the robot look towards the target
        joint3 = self.current_pose[2]
        joint1 = self.current_pose[0]
        joint_positions = self.compute_joints_position(msg, joint1, joint3)
        if joint_positions is None:
            # we are done. we did not get a solution
            self.get_logger().warn("The compute_joints_position() function returned None. Failed to command the robot.")
            return
        else:
            # upack result
            new_j1, new_j3 = joint_positions

        # publish command
        msg = Float64MultiArray()
        msg.data = [float(new_j1), float(0.0), float(new_j3), float(0.0)]
        self.joint_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = None
    
    try:
        node = RunPolicyNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
