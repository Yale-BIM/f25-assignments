# Getting Started with vcstool

## Introduction
[vcstool](http://wiki.ros.org/vcstool) is a set of command-line tools for managing projects from multiple source control systems.
It streamlines the maintainence of workspaces that might have many version-controlled components, including the workspaces used for Shutter's codebase.

This document is intended to provide a quickstart to setting up a new workspace with vcstool, examples of common usage scenarios, and a draft of best practices.

## Key Features of vcstool
There are three key features that make vcstool useful for Shutter.

1. Importing a workspace configuration.
2. Exporting a workspace configuration. This capability is useful for specifying a configuration to share across different installations, as well as for saving the current state of the workspace.
3. Checking the status of various repositories.

## Starting from scratch
This section assumes that the dependencies enumerated in [shutter-ros documentation](https://shutter-ros.readthedocs.io/en/latest/) have been installed, but that other steps have not been completed.
This section explains in greater detail what the installation documentation actually does.

First, create a workspace and clone [shutter-ros repository](https://gitlab.com/interactive-machines/shutter/shutter-ros):

```
mkdir -pv ~/catkin_ws/src/ros-planning
cd ~/catkin_ws/src
git clone https://gitlab.com/interactive-machines/shutter/shutter-ros.git

cd shutter-ros
git submodule update --init --recursive
```

If we inspect the files in `shutter-ros`, one of the files is `noetic_moveit.repos`.
This file contais the specifications for the repositores that we want to clone with vcstool.
One of the entries in `noetic_moveit.repos` is replicated below:

```
moveit:
  type: git
  url: https://github.com/yale-img/moveit.git
  version: devel
```

We can use vcstool to import this set of repository configurations into another directory.

```
cd ~/catkin_ws/src
vcs import --input shutter-ros/noetic_moveit.repos --recursive ros-planning/
```

This command reads the workspace configuration in `noetic_moveit.repos`, checks out any submodules recursively, and clones into the base path `ros-planning`.
Note that the last argument could be replaced by any valid path, including the current directory (i.e., `.`), which is the default path used by vcstool if no other path is supplied.

After cloning this new workspace configuration, complete the installation steps:

```
# update rosdep
$ cd ~/catkin_ws
$ rosdep update

# install dependencies for Shutter
$ rosdep install -y -r --ignore-src --rosdistro=noetic --from-paths src

# build workspace
$ catkin_make -DCMAKE_BUILD_TYPE=Release

# install shutter_face Python dependencies
$ source ~/catkin_ws/devel/setup.bash
$ roscd shutter_face_ros
$ pip install -r requirements.txt --user
```


## Common usage examples

Note that these examples assume that the command is desired to run on the entire workspace.
Navigating to a specific repository within the workspace, or passing the flag `--repos {repository path}` will run the command on that repository only.

### Saving the current state of the workspace WITH commit hashes

```
cd ~/catkin_ws/src/
vcs export --nested --exact > working_state.repos
less working_state.repos
```

### Checking if any repositories have untracked files or uncommitted changes

```
cd ~/catkin_ws/src
vcs status --nested --skip-empty
```

### Print most recent commit message

```
cd ~/catkin_ws/src
vcs log --nested -l 1
```

### Switching to a different workspace configuration

Note that this command overwrites directories by default, so be sure to commit and push any changes before importing.

```
cd ~/catkin_ws/src
vcs import --input {name_of_config.repos} --recursive
```

### Saving a new workspace configuration

```
cd ~/catkin_ws/src
vcs export --nested > {name_of_config.repos}
```

## Best Practice Guidelines

+ Check if any repositories have uncommitted changes with `vcs status` before importing a new repository.
+ Export configurations to descriptive filenames.
+ Maintain configurations that include exact hash commits. These configurations will help to coordinate code versions.
+ However, try to avoid importing often from configurations with exact hash commits, as these repositories may be checked out in detached HEAD states.

If distributing repository configurations to collaborators, you can create a new git repository called `<my project name>_ws` with a `repository_config` directory.
Make sure to add the `build`, `devel`, and `src` directories to the repository `.gitignore`, since those directories will contain all of the files that ROS builds for your workspace.

## Resources

+ ROS Wiki: [http://wiki.ros.org/vcstool](http://wiki.ros.org/vcstool)
+ Source on github: [https://github.com/dirk-thomas/vcstool](https://github.com/dirk-thomas/vcstool)
+ rosinstall file format (not used currently): [https://docs.ros.org/en/independent/api/rosinstall/html/rosinstall_file_format.html](https://docs.ros.org/en/independent/api/rosinstall/html/rosinstall_file_format.html)
+ wstool (similar utility): [http://wiki.ros.org/wstool](http://wiki.ros.org/wstool)
