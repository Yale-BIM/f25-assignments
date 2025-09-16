#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

class LookForwardNode(Node):
    """Node that makes Shutter's wrist_1_link rotate to make the Zed camera look forward."""
    
    def __init__(self):
        super().__init__('look_forward')
        
        # publish rate
        publish_rate = 1  # Hz

        # parameters for the joint of interest
        self.joint_name = "joint_4"
        self.desired_joint_position = 0.0
        self.joint_reached_desired_position = False
        self.joint_command = [0.0, 0.0, 0.0, 0.0]

        # Publishers
        self.joint_pub = self.create_publisher(Float64MultiArray, "/unity_joint_group_controller/command", 5)

        # Subscribers
        self.joints_sub = self.create_subscription(JointState, "/joint_states", self.joints_callback, 5)

        # Create timer for publishing
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_joint_command)
        
        self.get_logger().info('Look forward node initialized')

    def publish_joint_command(self):
        """Publish joint command if not at desired position"""
        if not self.joint_reached_desired_position:
            msg = Float64MultiArray()
            self.joint_command[3] = self.desired_joint_position
            msg.data = self.joint_command
            self.joint_pub.publish(msg)
        else:
            self.get_logger().info('Joint reached desired position, stopping node')
            self.destroy_node()
            rclpy.shutdown()

    def joints_callback(self, msg):
        """Callback for joint state messages"""
        # current joint position
        self.joint_command = list(msg.position)
        joint_position = msg.position[3]
        self.get_logger().debug(f'joint position: {joint_position}')

        if np.fabs(joint_position - self.desired_joint_position) < 1e-2:
            self.joint_reached_desired_position = True
        else:
            self.joint_reached_desired_position = False
        self.get_logger().debug(f'reached? {self.joint_reached_desired_position}')


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = LookForwardNode()
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
