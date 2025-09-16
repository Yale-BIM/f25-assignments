# Motion Control for Shutter

Shutter's motion can be controlled in four different ways:
1. Publish goal positions directly to `/joint_group_controller/command`
2. Compute joint goals with the `shutter_servo` package
3. Teleoperation with the `shutter_teleop` package
4. Compute trajectories to known goals with MoveIt

This document describes basic pros and cons for each method, and points to additional useful documentation.
The table below summarizes example use cases for each method:

| Method | Key Advantage | Good use case |
|--|--|--|
| [Publishing Goal Positions](#method-1-publishing-goal-positions) | Simple | Prototyping joint goal positions |
| [Shutter Servo](#method-2-shutter-servo-package) | Enforces motion constraints | Reactive motion |
| [Shutter Teleop](#method-3-shutter-teleop-package) | User control | Collecting expert demonstrations |
| [MoveIt](#method-4-moveit-planning-library) | Planning | Large motions to known goal poses |

## Method 1: Publishing Goal Positions

With this method, joint goal positions are published directly to the hardware interface.
The physical robot expects the goal to be published to the topic `/joint_group_controller/command`, whereas the simulated robot expects the goal to be published to the topic `/unity_joint_group_controller/command`.
In both cases, the goal is a four-element array specifying the desired position (in radians) for each joint.

Note that goals represent *absolute position*, rather than a displacement.
Hence, specifying an incremental motion for a single joint depends on the current position.

**Advantages:** Simple. Does not require any extra configuration or starting additional ROS nodes.

**Disadvantages:** No self-collision checking, kinematic singularity avoidance or joint limits enforcement. No parameterization by timing; sequential positions must be published manually.


## Method 2: Shutter Servo Package

With this method, joint goal positions are processed by the [MoveIt Servo package](https://github.com/yale-img/moveit/tree/master/moveit_ros/moveit_servo) before they are forwarded to the hardware interface.
This pipeline adds kinematic singularity and self-collision checking, to minimize the chances of breaking the robot.
The main `shutter_servo` node expects the goal to be published to the topic `/shutter_servo/servo/joint_command`.
As with [Method 1](#method-1-publishing-goal-positions), goals represent *absolute position*, rather than a displacement.

An example of using the `shutter_servo` package is available in the `shutter_opt_control` package.
In particular, the launch file `shutter_servo.launch` should be included in your project's main launch file:

```
<!-- Shutter Servo -->
<include file="$(find shutter_servo)/launch/shutter_servo.launch">
  <arg name="simulation" value="$(arg simulation)"/>
</include>
```

Topics for the commands may also need to be remapped, depending on whether the robot is physical or simulated:

```
<remap from="/joint_group_controller/command" to="/unity_joint_group_controller/command" if="$(arg simulation)"/>
<remap from="/optimize_joints_towards_target/command" to="/shutter_servo/servo/joint_command"/>
```

Shutter Servo is part of the [shutter-ros repository](https://gitlab.com/interactive-machines/shutter/shutter-ros/-/tree/master/shutter_servo).
Documentation is sparse as the package is still being actively developed.

**Advantages:** Enforces self-collision avoidance and kinematic singularity detection. Soft real-time responsiveness.

**Disadvantages:** Requires additional setup of ROS nodes and parameters. No parameterization by timing; sequential positions must be published manually.


## Method 3: Shutter Teleop Package

Shutter can also be teleoperated with a joystick, such as the Dualshock 3 controller for a PlayStation.
Teleoperation could be used to obtain expert demonstrations for motion in an imitation learning setting.
Teleoperation is implemented in the [shutter_teleop package](https://gitlab.com/interactive-machines/shutter/shutter-ros/-/tree/master/shutter_teleop).
More detail is available in the [shutter-ros online documentation](https://shutter-ros.readthedocs.io/en/noetic_moveit/packages/shutter_teleop.html).

The `shutter_teleop` package is also notable as it implements an alternative interface to MoveIt Servo.
Instead of sending joint goals to MoveIt Servo, the joystick commands modify a Cartesian pose.
This pose is converted into joint velocity commands that move the robot's end effector (defined here as the `camera_link` frame) to match the pose goal.
It is possible (though not yet implemented) to port this pose tracking interface to the `shutter_servo` package.

**Advantages:** Can obtain demonstrations from a human expert. Cartesian pose could be more intuitive to reason about than joint values.

**Disadvantages:** Requires additional setup of ROS nodes and parameters. No parameterization by timing; pose goals must be published sequentially (manually or by a timer).


## Method 4: MoveIt Planning Library

With this method, trajectories are planned and executed with the MoveIt library.
Trajectories have the advantage of specifying sequential robot motion.
For instance, a large motion to a position or pose goal can be generated, which will avoid self-collisions without requiring each intermediate joint position command to be manually specified.

For more details, check out the [Move Group Python Interface tutorial](https://ros-planning.github.io/moveit_tutorials/doc/move_group_python_interface/move_group_python_interface_tutorial.html).
The [shutter_moveit_config package](https://shutter-ros.readthedocs.io/en/latest/packages/shutter_moveit_config.html) contains an example using MoveIt that is heavily adapted from the Move Group Python Interface tutorial.
The example can be executed with the following command:

```bash
$ roslaunch shutter_moveit_config python_interface_tutorial.launch moveit_controller_manager:=[fake | simple | ros_control]
```

The argument ``moveit_controller_manager`` specifies the robot platform to execute.
``fake`` is the default and simulates the robot entirely within RViz.
``simple`` expects the standalone Unity-built simulation for Shutter.
``ros_control`` expects the real robot.

The example uses MoveIt to plan and execute trajectories in three different ways: joint goals, pose goals and Cartesian goals.
Joint position goals are generally the easiest style of specifying goal states for Shutter.
Cartesian path planning is particularly difficult, owing to the low degrees of freedom available to Shutter.

By default, the example uses Shutter's [IKFast inverse kinematics solver](https://gitlab.com/interactive-machines/shutter/shutter-ros/-/tree/master/shutter_ikfast_plugin) and the OMPL implementation of [BFMT\* motion planner](https://ompl.kavrakilab.org/classompl_1_1geometric_1_1BFMT.html).
MoveIt is a modular library, so different IK solvers and planners can be specified.
Not all combinations of planners and solvers will be equally successful or applicable to a given scenario.
For example, CHOMP only supports joint goals, and TRAC-IK often fails to find continuous pose sequences for Cartesian path planning.

Note that Shutter's behavior tree interface to MoveIt uses a C++ interface instead of the Python interface, so there are fewer examples in the Interactive Machines Group codebase.
However, MoveIt is a widely used motion planning library in both academia and industry, so there is a broader ecosystem of documentation and tutorials available.

**Advantages:** Motion planning considers self-collisions, kinematic singularity avoidance and time. Trajectories can be more easily synchronised with a high-level control architecture, such as behavior trees or finite-state machines. More external documentation and support available.

**Disadvantages:** Requires additional setup of ROS nodes and parameters. Fewer examples in Interactive Machines Group codebases.


## Additional resources

+ MoveIt tutorials: https://ros-planning.github.io/moveit_tutorials/doc/getting_started/getting_started.html
+ Nice [blog post](https://picknik.ai/visual%20servoing/moveit%20servo/2021/01/21/fast-visual-servoing-with-moveit.html) about MoveIt planning and MoveIt Servo integration
+ [shutter-ros online documentation](https://shutter-ros.readthedocs.io/en/latest/)

If any issues during setup or integration arise, please contact course staff via Slack.
Including a screenshot of warnings and errors from ROS is recommended to facilitate troubleshooting.
