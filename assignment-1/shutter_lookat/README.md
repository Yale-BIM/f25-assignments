# Shutter Look At Package

This ROS package has starting code for the [Assignment 1](../README_ros2.md).

## Python Nodes (within the scripts directory)

- **look_forward_ros2.py:** Node that makes shutter orient its zed camera forward. It exists when the camera
is looking forward.

- **generate_target_ros2.py:** Node that publishes the location of a simulated target in front of the robot.

## Launch files

- **generate_target_ros2.launch.py:** Launch file that brings up the robot, generates a simulated target, 
opens up rviz2, and makes the robot look forward.
