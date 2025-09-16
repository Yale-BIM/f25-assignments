#!/usr/bin/env python3
# Simple script to save color and depth images captured by a realsense camera

import sys
import rclpy
from rclpy.node import Node
import cv_bridge
import message_filters
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
import cv2

class ImageCaptureNode(Node):
    """
    ROS 2 node to capture and save RealSense camera images
    """
    
    def __init__(self):
        super().__init__('capture_images')
        
        # Create subscriptions
        self.image_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        self.info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/color/camera_info')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        
        # Synchronize the subscriptions
        self.ts = message_filters.TimeSynchronizer([self.image_sub, self.info_sub, self.depth_sub], 10)
        self.ts.registerCallback(self.callback)
        
        self.get_logger().info('Image capture node initialized')

    def callback(self, image, camera_info, depth):
        """
        Process data and quit.
        :param image: image message
        :param camera_info: camera info message
        :param depth: depth image message
        """
        self.get_logger().info(f"Got data at {self.get_clock().now()}")

        bridge = cv_bridge.CvBridge()
        cv_image = bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
        cv_depth = bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")

        # convert depth from mm to meters
        cv_depth = cv_depth.astype(float) / 1000.0 

        # convert image to grayscale
        cv_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # save images
        np.savez('images.npz',
                 gray=cv_gray,
                 depth=cv_depth,
                 width=camera_info.width,
                 height=camera_info.height,
                 K=camera_info.k)
        
        self.get_logger().info("Images saved successfully!")
        # once an image is saved, quit!
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    
    node = ImageCaptureNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
