#!/usr/bin/env python3
import rclpy
from rclpy.time import Time
from rclpy.node import Node
from rclpy.logging import get_logger
import numpy as np
import copy

from shutter_lookat.msg import Target
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker


class SimulatedObject(object):
    """
    Simulated object that moves on a circular path in front of the robot.
    The path is contained in a plane parallel to the y-z plane (i.e., x is constant for all points in the path).
    """

    def __init__(self):
        """
        Constructor
        """
        self.x = 1.5                    # x coordinate for the center of the object
        self.center_y = 0.0             # y coordinate for the center of the object's path
        self.center_z = 0.50            # z coordinate for the center of the object's path
        self.angle = 0.0                # current angle for the object in its circular path (relative to the y axis)
        self.radius = 0.5               # radius of the object's circular path
        self.frame = "base_footprint"   # frame in which the coordinates of the object are computed
        self.path_type = "circular"     # type of motion: circular, horizontal, vertical 
        self.fast = False               # fast moving target?

    def step(self):
        """
        Update the position of the target based on the publishing rate of the node
        :param publish_rate: node's publish rate
        """
        if self.fast:
            denom = 150.0 # 1 full revolution in 5 secs at 30 Hz
        else:
            denom = 300.0
        self.angle += 2.0 * np.pi / denom  

    def get_obj_coord(self):
        if self.path_type == "circular":
            x = self.x
            y = self.center_y + np.sin(self.angle)*self.radius
            z = self.center_z + np.cos(self.angle)*self.radius
        elif self.path_type == 'horizontal':
            x = self.x
            y = self.center_y + np.sin(self.angle)*self.radius
            z = self.center_z 
        elif self.path_type == 'vertical':
            x = self.x
            y = self.center_y 
            z = self.center_z + np.cos(self.angle)*self.radius
        else:
            get_logger('generate_continuous_target').error(f"Unrecognized path_type for the moving target. Got {self.path_type} but expected 'circular', 'horizontal' or 'vertical'")
            return None
        
        return x,y,z


class GenerateContinuousTargetNode(Node):
    """ROS 2 node for generating continuous target motion"""

    def __init__(self):
        super().__init__('generate_target')

        # Get ROS parameters
        self.declare_parameter("x_value", 1.5)
        self.declare_parameter("radius", 0.1)
        self.declare_parameter("publish_rate", 30)
        self.declare_parameter("path_type", "horizontal")
        self.declare_parameter("add_noise", False)
        self.declare_parameter("fast", False)

        x_value = self.get_parameter("x_value").value
        self.radius = self.get_parameter("radius").value
        publish_rate = self.get_parameter("publish_rate").value
        path_type = self.get_parameter("path_type").value
        self.add_noise = self.get_parameter("add_noise").value
        fast = self.get_parameter("fast").value

        # Create the simulated object
        self.object = SimulatedObject()
        self.object.path_type = path_type
        self.object.fast = fast
        self.object.x = x_value

        # Define publishers
        self.vector_pub = self.create_publisher(Target, '/target', 5)           # observation
        self.vector_gt_pub = self.create_publisher(PoseStamped, '/true_target', 5)   # true target position
        self.marker_pub = self.create_publisher(Marker, '/target_marker', 5)    # visualization for the observation

        # Create timer for publishing at constant rate
        self.timer = self.create_timer(1.0/publish_rate, self.publish_target)
        
        self.timestamp_buffer = None

    def publish_target(self):
        """Publish the target at a constant rate"""
        
        # publish the location of the target as a PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.object.frame
        x, y, z = self.object.get_obj_coord()
        
        if self.add_noise:
           obs_x = x + np.random.normal(loc=0.0, scale=0.05)
           obs_y = y + np.random.normal(loc=0.0, scale=0.05)
           obs_z = z + np.random.normal(loc=0.0, scale=0.03)
        else:
           obs_x = x
           obs_y = y
           obs_z = z
           
        pose_msg.pose.position.x = obs_x
        pose_msg.pose.position.y = obs_y
        pose_msg.pose.position.z = obs_z
        pose_msg.pose.orientation.w = 1.0

        # Check if current Time exceeds clock speed
        if self.timestamp_buffer is not None and self.timestamp_buffer >= Time.from_msg(pose_msg.header.stamp):
            self.get_logger().warn('Publish rate exceeds clock speed; check your clock publish rate')
            return

        target_msg = Target()
        target_msg.pose = pose_msg
        target_msg.radius = self.radius
        self.vector_pub.publish(target_msg)
        self.timestamp_buffer = Time.from_msg(pose_msg.header.stamp)

        # publish a marker to visualize the target in RViz
        marker_msg = Marker()
        marker_msg.header = pose_msg.header
        marker_msg.action = Marker.ADD
        marker_msg.color.a = 0.5
        marker_msg.color.r = 1.0
        marker_msg.lifetime.sec = 1
        marker_msg.lifetime.nanosec = 0
        marker_msg.id = 0
        marker_msg.ns = "target"
        marker_msg.type = Marker.SPHERE
        marker_msg.pose = pose_msg.pose
        marker_msg.scale.x = 2.0*self.radius
        marker_msg.scale.y = 2.0*self.radius
        marker_msg.scale.z = 2.0*self.radius
        self.marker_pub.publish(marker_msg)

        # publish true obj position
        gt_pose_msg = copy.deepcopy(pose_msg)
        gt_pose_msg.pose.position.x = x
        gt_pose_msg.pose.position.y = y
        gt_pose_msg.pose.position.z = z
        self.vector_gt_pub.publish(gt_pose_msg)

        # update the simulated object state
        self.object.step()


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = GenerateContinuousTargetNode()
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
