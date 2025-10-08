cmake_minimum_required(VERSION 3.8)
project(shutter_kf_public_tests)

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
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(shutter_lookat REQUIRED)
find_package(shutter_bringup REQUIRED)
find_package(shutter_kf REQUIRED)

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
    #    This tells Python to look for modules in your 'test' directory.
    set_tests_properties(${ctest_name} PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/test:$ENV{PYTHONPATH}"
    )
    # --- END OF NEW CODE ---
  endfunction()
  
  add_ros_isolated_launch_test(test/test_kalman_filter.py)
  add_ros_isolated_launch_test(test/test_filter_quality.py)
endif()

ament_package()
