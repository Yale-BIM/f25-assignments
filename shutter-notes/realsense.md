# Intel RealSense Usage Notes

RealSense is a range of cameras manufactured by Intel that provide depth perception.
Shutter has a [D435i RealSense camera](https://www.intelrealsense.com/depth-camera-d435/) mounted above the face display.
The RealSense gives Shutter egocentric perception along the *eye in hand* paradigm.


## Installation

The [RealSense ROS wrapper](https://github.com/IntelRealSense/realsense-ros) is installed via `apt-get` on the course laptops.
The following packages are necessary:
+ `ros-noetic-realsense2-camera`
+ `ros-noetic-realsense2-description`

You can verify that the packages are installed with the command and expected output:
```
$ apt search ros-noetic-realsense2-camera
Sorting... Done
Full Text Search... Done
ros-noetic-realsense2-camera/focal,now 2.3.2-1focal.20221003.141840 amd64 [installed]
  RealSense Camera package allowing access to Intel T265 Tracking module and SR300 and D400 3D cameras
```
Note that the library version 2.3.2 is **required**; older versions of the library are not validated to work with ROS1 and the current RealSense firmware versions.


## Usage

The primary roslaunch file for the RealSense camera is `rs_camera.launch` in the `realsense2_camera` package.
You can start the RealSense camera by itself with the following command:
```
$ roslaunch realsense2_camera rs_camera.launch
```
You can verify that the camera is publishing images by viewing the topic `/camera/color/image_raw` in RViz.


### Common Launch File Arguments

`rs_camera.launch` has many optional arguments to control the RealSense camera's perception capabilities.
The arguments can be specified on the command-line, or passed from another launch file with the `<include>` tag.
The following table enumerates some useful arguments:
| Name  | Description                      | Use Case                                          |
| -------------  | -----------                      | --------                                          |
| initial_reset  | perform hardware reset at start  | reset can improve stability                       |
| enable_depth   | toggle depth camera              | disabling depth can improve stability             |
| color_width    | set image width in pixels        |                                                   |
| color_height   | set image height in pixels       |                                                   |
| color_fps      | set image FPS                    | FPS value may require specific width and height   |
| enable_sync    | synchronize sensor frames        | sync can improve stability                        |

A more thorough enumeration of available launch file arguments is available in the [RealSense ROS Wrapper repository](https://github.com/IntelRealSense/realsense-ros#launch-parameters).
The [RealSense D400 series dataset](https://www.intelrealsense.com/wp-content/uploads/2022/05/Intel-RealSense-D400-Series-Datasheet-April-2022.pdf) enumerates available camera resolution and FPS values (see Table 4-2 starting on page 70).

## Troubleshooting Steps

The RealSense camera requires a USB3.0 connection.
It is recommended to connect the RealSense directly to your machine, rather than through a USB hub.
Checking the camera-side USB connection can also help to resolve errors when starting the RealSense.

Launch file parameters can also improve reliability and stability.
In particular, setting `initial_reset:=true` is recommended.
Disabling sensors that are not required (e.g., the depth camera) is often helpful as well.