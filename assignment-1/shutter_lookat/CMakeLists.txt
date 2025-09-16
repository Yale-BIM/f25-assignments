cmake_minimum_required(VERSION 3.8)
project(shutter_lookat)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rosidl_default_runtime REQUIRED)

find_package(geometry_msgs REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Target.msg"
  DEPENDENCIES geometry_msgs std_msgs
)

# Install Python scripts
install(PROGRAMS
  scripts/generate_target.py
  scripts/look_forward.py
  scripts/publish_target_relative_to_realsense_camera.py
  scripts/virtual_camera.py
  scripts/fancy_virtual_camera.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.launch.py"
)

# Install config files
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Install message files
install(DIRECTORY msg/
  DESTINATION share/${PROJECT_NAME}/msg
  FILES_MATCHING PATTERN "*.msg"
)

# Export dependencies
ament_export_dependencies(rosidl_default_runtime)
ament_export_dependencies(geometry_msgs)
ament_export_dependencies(rclpy)
ament_export_dependencies(std_msgs)
ament_export_dependencies(tf2_ros)
ament_export_dependencies(tf2_geometry_msgs)
ament_export_dependencies(visualization_msgs)
ament_export_dependencies(sensor_msgs)
ament_export_dependencies(cv_bridge)
ament_export_dependencies(message_filters)

# Test dependencies
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
