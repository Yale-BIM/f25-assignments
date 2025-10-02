# Shutter Lookat Public Tests (ROS 2)

Run the ROS 2 integration tests as described below:

## Testing Approach

In ROS 2, testing is done in two steps:
1. **Start the system** using launch files
2. **Run tests** using colcon test framework

## Running Tests

### For Part II of Assignment 1

```bash
$ colcon test --packages-select shutter_lookat_public_tests --event-handlers console_direct+ --ctest-args -R test_publish_target
```

### For Part III of Assignment 1

```bash
$ colcon test --packages-select shutter_lookat_public_tests --event-handlers console_direct+ --ctest-args -R test_virtual_camera
```

### For Part IV of Assignment 1

```bash
$ colcon test --packages-select shutter_lookat_public_tests --event-handlers console_direct+ --ctest-args -R test_fancy_virtual_camera
```
