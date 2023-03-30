# Turtlebot_Sensor_Fusion
The Turtlebot4 Estimator is an object-oriented software package that estimates the linear velocity 
and traveled distance of a Turtlebot4 robot using data from an IMU, optical mouse sensor, 
and Marvelmind coordinate system. The package is implemented in Python using the ROS2 framework.

# Environments
TurtleBot 4 is the next-generation of the world’s most popular open source robotics platform for education and research, offering better computing power, better sensors and a world class user experience at an affordable price point. Currently the ros2 version has been updated to humble, Ubuntu22.04, and the turtlebot4 has added new namespace features. Our turtlebot4's namespace is yosemite.

# Features
The Turtlebot4 Estimator provides the following features:
1. Object-oriented design: The estimator is designed using object-oriented principles, providing a modular and extensible architecture.
2. Multi-sensor fusion: The estimator fuses data from multiple sensors, including an IMU, optical mouse sensor, and Marvelmind 
   coordinate system, to provide accurate estimates of the robot's motion.
3. Linear velocity and distance estimation: The estimator estimates the robot's linear velocity and traveled distance, 
   providing key metrics for robot navigation and control.
4. ROS2 integration: The estimator is integrated with the ROS2 framework, allowing for easy communication with other ROS2 nodes and modules.

# Installation
1. Install ROS2 on your system, following the instructions on the ROS2 website.
2. Clone the Turtlebot4 Estimator repository into your ROS2 workspace:
```
$ cd ~/ros2_ws/src
$ git clone https://github.com/tianmuw/Turtlebot_Sensor_Fusion.git
```
3. Build the package using colcon
```
$ cd /path/to/your/ros2/workspace
$ colcon build
```

4. Source the ROS2 installation and setup files
```
$ source /etc/opt/ros2/install/setup.bash
$ source ~/ros2_ws/install/setup.bash
```
