#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time
import numpy as np

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
        self.radius = 0.1               # radius of the object's circular path
        self.frame = "base_footprint"   # frame in which the coordinates of the object are computed

    def step(self):
        """
        Update the position of the target based on the publishing rate of the node
        :param publish_rate: node's publish rate
        """
        self.angle += 2.0 * np.pi / 300  # 1 full revolution in 10 secs at 30 Hz


class GenerateTargetNode(Node):
    """
    ROS 2 node to generate a simulated moving target
    """
    
    def __init__(self):
        super().__init__('generate_target')
        
        # Create the simulated object
        self.object = SimulatedObject()

        # Get ROS parameters
        self.declare_parameter('x_value', 1.5)
        self.declare_parameter('radius', 0.1)
        self.declare_parameter('publish_rate', 30)
        
        x_value = self.get_parameter('x_value').get_parameter_value().double_value
        radius = self.get_parameter('radius').get_parameter_value().double_value
        publish_rate = self.get_parameter('publish_rate').get_parameter_value().integer_value
        
        self.object.x = x_value
        self.object.radius = radius
        self.timestamp_buffer = None
        
        # Define publishers
        self.vector_pub = self.create_publisher(Target, '/target', 5)
        self.marker_pub = self.create_publisher(Marker, '/target_marker', 5)

        # Create timer for publishing
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_target)
        
        self.get_logger().info('Generate target node initialized')

    def publish_target(self):
        """
        Publish the target at a constant rate
        """
        # publish the location of the target as a Vector3Stamped
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.object.frame
        pose_msg.pose.position.x = self.object.x
        pose_msg.pose.position.y = self.object.center_y + np.sin(self.object.angle) * self.object.radius
        pose_msg.pose.position.z = self.object.center_z + np.cos(self.object.angle) * self.object.radius
        pose_msg.pose.orientation.w = 1.0

        # Check if current Time exceeds clock speed
        if self.timestamp_buffer is not None and Time.from_msg(self.timestamp_buffer).nanoseconds >= Time.from_msg(pose_msg.header.stamp).nanoseconds:
            self.get_logger().warn('Publish rate exceeds clock speed; check your clock publish rate')
            return

        target_msg = Target()
        target_msg.pose = pose_msg
        target_msg.radius = self.object.radius
        self.vector_pub.publish(target_msg)
        self.timestamp_buffer = pose_msg.header.stamp

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
        marker_msg.scale.x = 2.0 * self.object.radius
        marker_msg.scale.y = 2.0 * self.object.radius
        marker_msg.scale.z = 2.0 * self.object.radius
        self.marker_pub.publish(marker_msg)

        # update the simulated object state
        self.object.step()


def main(args=None):
    rclpy.init(args=args)
    node = None
    
    try:
        node = GenerateTargetNode()
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
