import os
import subprocess
import rclpy

def inspect_rostopic_info(node_name):
    """
    Helper function to check a node's connections
    :return: True if the node subscribes to the target_topic
    """
    out = subprocess.Popen(['ros2', 'node', 'info', node_name],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()

    if stderr is not None:
        print("Failed to run ros2 node info. Error:\n{}".format(stderr.decode('utf-8')))
        return False

    stdout = stdout.decode('utf-8')
    headers = ["Publishers:", "Subscribers:", "Service Servers:"]

    in_sub = False
    for line in stdout.split('\n'):
        line = line.strip()
        # rclpy.logwarn(line)  # print output of ros2 node info
        if line in headers:
            in_sub = False
            if line == "Subscribers:":
                in_sub = True

        if in_sub and "/target: shutter_lookat/msg/Target" in line:
            return True

    return False

def compute_import_path(*args):
    test_dir = os.path.dirname(os.path.abspath(__file__))
    import_dir = os.path.abspath(os.path.join(test_dir, '..', '..'))
    for path in args:
        import_dir = os.path.abspath(os.path.join(import_dir, path))
    return import_dir
