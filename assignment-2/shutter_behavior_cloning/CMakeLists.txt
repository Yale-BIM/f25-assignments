cmake_minimum_required(VERSION 3.8)
project(shutter_behavior_cloning)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_ros REQUIRED)
find_package(ament_cmake_pytest REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(launch_testing_ament_cmake REQUIRED)
find_package(ament_lint_auto REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(shutter_lookat REQUIRED)
find_package(shutter_bringup REQUIRED)

set(MODEL_PATH "?????????")
set(NORM_PARAMS_PATH "")

# Install Python modules
# ament_python_install_package(${PROJECT_NAME})

# Install Python executables
file(GLOB PYTHON_SRC_FILES "src/*.py")
file(GLOB PYTHON_SCRIPT_FILES "scripts/*.py")
install(PROGRAMS
  ${PYTHON_SRC_FILES}
  ${PYTHON_SCRIPT_FILES}
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install data directory
install(DIRECTORY
  data
  DESTINATION share/${PROJECT_NAME}/
)

# --- Testing ---
if(BUILD_TESTING)
  # Integration tests

  function(add_ros_isolated_launch_test path)
    # --- This part is the same as before ---
    set(RUNNER "${ament_cmake_ros_DIR}/run_test_isolated.py")
    add_launch_test("${path}" RUNNER "${RUNNER}" ${ARGN})

    # --- START OF NEW CODE TO ADD PYTHON PATH ---
    # 1. Create the CTest name by replacing '/' with '_' in the path.
    #    This creates the name that CTest actually uses (e.g., 'test_test_publish_target.py').
    string(REPLACE "/" "_" ctest_name ${path})

    # 2. Set the PYTHONPATH environment variable specifically for this test.
    #    This tells Python to look for modules in your 'test' directory and virtual environment.
    set_tests_properties(${ctest_name} PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/test:$ENV{PYTHONPATH}:$ENV{VIRTUAL_ENV}/lib/python3.12/site-packages"
      TIMEOUT 200
    )
    # --- END OF NEW CODE ---
  endfunction()
  
  add_ros_isolated_launch_test(test/test_policy.py)
endif()

ament_package()
