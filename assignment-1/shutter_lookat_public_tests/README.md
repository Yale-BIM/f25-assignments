# Shutter Lookat Public Tests (ROS 2)

Run the ROS 2 integration tests as described below:

## Testing Approach

In ROS 2, testing is done in two steps:
1. **Start the system** using launch files
2. **Run tests** using colcon test framework

## Running Tests

### For Part II of Assignment 1

First, start the system:
```bash
$ ros2 launch shutter_lookat_public_tests test_publish_target_ros2.launch.py
```

Then, in a separate terminal, run the tests:
```bash
$ cd <path-to-your-workspace>
$ colcon test --packages-select shutter_lookat_public_tests
$ colcon test-result --all --verbose
```

### For Part III of Assignment 1

First, start the system:
```bash
$ ros2 launch shutter_lookat_public_tests test_virtual_camera_ros2.launch.py
```

Then, in a separate terminal, run the tests:
```bash
$ cd <path-to-your-workspace>
$ colcon test --packages-select shutter_lookat_public_tests
$ colcon test-result --all --verbose
```

### For Part IV of Assignment 1

First, start the system:
```bash
$ ros2 launch shutter_lookat_public_tests test_fancy_virtual_camera_ros2.launch.py
```

Then, in a separate terminal, run the tests:
```bash
$ cd <path-to-your-workspace>
$ colcon test --packages-select shutter_lookat_public_tests
$ colcon test-result --all --verbose
```

## Alternative: Direct pytest execution

You can also run tests directly with pytest (after starting the system):

```bash
# For Part II
$ python3 -m pytest src/f25-assignments-tmp/assignment-1/shutter_lookat_public_tests/test/test_publish_target_ros2.py -v

# For Part III  
$ python3 -m pytest src/f25-assignments-tmp/assignment-1/shutter_lookat_public_tests/test/test_virtual_camera_ros2.py -v

# For Part IV
$ python3 -m pytest src/f25-assignments-tmp/assignment-1/shutter_lookat_public_tests/test/test_fancy_virtual_camera_ros2.py -v
```

## Notes

- Make sure to build your workspace before running tests: `colcon build`
- The system must be running (via launch files) before executing tests
- Tests use pytest framework instead of rostest (ROS 1 approach)
- Launch files only start the system components, they don't run tests automatically
