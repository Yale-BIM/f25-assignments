#!/usr/bin/env python3

import os
from datetime import datetime
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
import tf2_ros
import transformations as tft
import tf2_geometry_msgs
import numpy as np
from urdf_parser_py.urdf import URDF
from rclpy.parameter_client import AsyncParameterClient
from scipy.optimize import least_squares
from std_msgs.msg import Bool
from std_msgs.msg import Float64MultiArray


def transform_msg_to_T(trans):
    """
    Convert TransformStamped message to 4x4 transformation matrix
    :param trans: TransformStamped message
    :return:
    """
    # extract relevant information from transform
    q = [trans.transform.rotation.x,
         trans.transform.rotation.y,
         trans.transform.rotation.z,
         trans.transform.rotation.w]
    t = [trans.transform.translation.x,
         trans.transform.translation.y,
         trans.transform.translation.z]
    # convert to matrices
    Rq = tft.quaternion_matrix(q)
    Tt = tft.translation_matrix(t)
    return np.dot(Tt, Rq)


def make_joint_rotation(angle, rotation_axis='x'):
    """
    Make rotation matrix for joint (assumes that joint angle is zero)
    :param angle: joint angle
    :param rotation_axis: rotation axis as string or vector
    :return: rotation matrix
    """
    # set axis vector if input is string
    if not isinstance(rotation_axis, list):
        assert rotation_axis in ['x', 'y', 'z'], "Invalid rotation axis '{}'".format(rotation_axis)
        if rotation_axis == 'x':
            axis = (1.0, 0.0, 0.0)
        elif rotation_axis == 'y':
            axis = (0.0, 1.0, 0.0)
        else:
            axis = (0.0, 0.0, 1.0)
    else:
        axis = rotation_axis
    # make rotation matrix
    R = tft.rotation_matrix(angle, axis)
    return R


def target_in_camera_frame(angles, target_pose, rotation_axis1, rotation_axis2, T1, T2):
    """
    Transform target to camera frame
    :param angles: joint angles
    :param target_pose: target pose
    :param rotation_axis1: str representation for the rotation axis of joint1
    :param rotation_axis2: str representation for the rotation axis of joint3
    :param T1: transform - base_link to biceps
    :param T2: transform - biceps to camera_link
    :return: target in camera_link, target in base_link
    """

    # make transform for joint 1
    R1 = make_joint_rotation(angles[0], rotation_axis=rotation_axis1)

    # make transform for joint 3
    R2 = make_joint_rotation(angles[1], rotation_axis=rotation_axis2)

    # transform target to camera_link...
    p = np.array([[target_pose[0], target_pose[1], target_pose[2], 1.0]]).transpose()

    # target in base_link
    p1 = np.dot(np.dot(T1, R1), p)

    # target in camera_link
    result = np.dot(np.dot(T2, R2), p1)

    return result[0:2].flatten()


class ExpertNode(Node):
    """
    Node that simulates an expert controller using optimization. It controls two joints of the robot to make it
    point towards a target.
    """

    def __init__(self, name='expert_opt'):
        super().__init__(name)

        # params
        self.declare_parameter("base_link", "base_link")
        self.declare_parameter("biceps_link", "biceps_link")
        self.declare_parameter("camera_link", "camera_color_optical_frame")
        self.declare_parameter("save_state_actions", True)
        self.declare_parameter("output_file", "")

        self.base_link = self.get_parameter("base_link").value
        self.biceps_link = self.get_parameter("biceps_link").value
        self.camera_link = self.get_parameter("camera_link").value
        self.save_state_actions = self.get_parameter("save_state_actions").value

        if self.save_state_actions:
            # Get package path using ament
            import ament_index_python
            try:
                bc_dir = ament_index_python.get_package_share_directory("shutter_behavior_cloning")
                default_output_file = os.path.join(bc_dir, "data", "state_actions.txt")
            except:
                # Fallback to current directory if package not found
                default_output_file = os.path.join(os.getcwd(), "data", "state_actions.txt")
            
            output_file_param = self.get_parameter("output_file").value
            self.output_file = output_file_param if output_file_param else default_output_file

            base_dir = os.path.dirname(self.output_file)
            if not os.path.isdir(base_dir):
                os.makedirs(base_dir)

            self.fid = open(self.output_file, 'a')  # open output buffer to record state-action pairs...
            date = datetime.now()
            self.fid.write("# data from {}\n".format(date.strftime("%d/%m/%Y %H:%M:%S")))

        else:
            self.fid = None

        # joint values
        self.move = True
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

        # joint publishers
        self.joint_pub = self.create_publisher(Float64MultiArray, "/unity_joint_group_controller/command", 5)

        # tf subscriber
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # joint subscriber
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joints_callback, 5)
        self.target_sub = self.create_subscription(PoseStamped, '/target', self.target_callback, 5)
        self.move_sub = self.create_subscription(Bool, '/move_towards_target', self.move_callback, 5)

    def cleanup(self):
        """
        Be good with the environment.
        """
        if self.fid is not None:
            self.fid.close()

    def move_callback(self, msg):
        """
        Move on/off callback
        """
        self.move = msg.data
        self.get_logger().warn("Motion is: {}".format(self.move))

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

    def get_p_T1_T2(self, msg):
        """
        Helper function for compute_joints_position()
        :param msg: target message
        :return: target in baselink, transform from base_link to biceps, transform from biceps to camera
        """

        # transform the target to baselink if it's not in that frame already
        if msg.header.frame_id != self.base_link:
            try:
                transform = self.tf_buffer.lookup_transform(self.base_link,
                                                            msg.header.frame_id,  # source frame
                                                            rclpy.time.Time(),
                                                            timeout=rclpy.duration.Duration(seconds=1.0))
                pose_transformed = tf2_geometry_msgs.do_transform_pose_stamped(msg, transform)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                self.get_logger().warn("Failed to compute new position for the robot because {}".format(e))
                return None, None, None
        else:
            pose_transformed = msg

        p = [pose_transformed.pose.position.x,
             pose_transformed.pose.position.y,
             pose_transformed.pose.position.z]

        # get transform from base link to camera link (base_link -> biceps_link and biceps_link -> camera_link)
        try:
            transform = self.tf_buffer.lookup_transform(self.biceps_link,
                                                        self.base_link,  # source frame
                                                        rclpy.time.Time(),
                                                        timeout=rclpy.duration.Duration(seconds=1.0))
            T1 = transform_msg_to_T(transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn("Failed to compute new position for the robot because {}".format(e))
            T1 = None

        try:
            transform = self.tf_buffer.lookup_transform(self.camera_link,
                                                        self.biceps_link,  # source frame
                                                        rclpy.time.Time(),
                                                        timeout=rclpy.duration.Duration(seconds=1.0))
            T2 = transform_msg_to_T(transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(str(e))
            T2 = None

        return p, T1, T2

    def compute_joints_position(self, msg, joint1, joint3):
        """
        Helper function to compute the required motion to make the robot's camera look towards the target
        :param msg: target message
        :param joint1: current joint 1 position
        :param joint3: current joint 3 position
        :return: new joint positions for joint1 and joint3; or None if something went wrong
        """
        p, T1, T2 = self.get_p_T1_T2(msg)
        if p is None or T1 is None or T2 is None:
            return None

        # compute the required motion for the robot using black-box optimization
        x0 = [-np.arctan2(p[1], p[0]), 0.0]
        res = least_squares(target_in_camera_frame, x0,
                            bounds=([-np.pi, -np.pi * 0.5], [np.pi, np.pi * 0.5]),
                            args=(p, self.robot.joints[1].axis, self.robot.joints[3].axis, T1, T2))
        # print("result: {}, cost: {}".format(res.x, res.cost))

        offset_1 = -res.x[0]
        offset_3 = -res.x[1]

        # cap offset for joint3 based on joint limits
        if joint3 + offset_3 > self.robot.joints[3].limit.upper:
            new_offset_3 = offset_3 + self.robot.joints[3].limit.upper - (joint3 + offset_3)
            self.get_logger().info("Computed offset of {} but this led to exceeding the joint limit ({}), "
                          "so the offset was adjusted to {}".format(offset_3, self.robot.joints[3].limit.upper,
                                                                    new_offset_3))
        elif joint3 + offset_3 < self.robot.joints[3].limit.lower:
            new_offset_3 = offset_3 + self.robot.joints[3].limit.lower - (joint3 + offset_3)
            self.get_logger().info("Computed offset of {} but this led to exceeding the joint limit ({}), "
                          "so the offset was adjusted to {}".format(offset_3, self.robot.joints[3].limit.lower,
                                                                    new_offset_3))
        else:
            new_offset_3 = offset_3

        new_j1 = joint1 + offset_1
        new_j3 = joint3 + new_offset_3

        return new_j1, new_j3

    def target_callback(self, msg):
        """
        Target callback
        :param msg: target message
        """
        if self.current_pose is None:
            self.get_logger().warn("Joint positions are unknown. Waiting to receive joint states.")
            return

        if not self.move:
            return

        # compute the required motion to make the robot look towards the target
        joint3 = self.current_pose[2]
        joint1 = self.current_pose[0]
        joint_angles = self.compute_joints_position(msg, joint1, joint3)
        if joint_angles is None:
            # we are done. the node might not be ready yet...
            return
        else:
            # upack results
            new_j1, new_j3 = joint_angles

        # write state and action (offset motion) to disk
        if self.fid is not None:
            self.fid.write("%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" %
                           (msg.header.frame_id, msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                            joint1, joint3, new_j1, new_j3))
            self.fid.flush()

        # publish command
        msg = Float64MultiArray()
        msg.data = [float(new_j1), float(0.0), float(new_j3), float(0.0)]
        self.joint_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = None
    
    try:
        node = ExpertNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.cleanup()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
