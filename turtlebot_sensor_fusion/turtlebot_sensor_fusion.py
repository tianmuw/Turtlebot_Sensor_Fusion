"""
author: tianmuw
Description: Use imu and hedge on turtlebot4, to estimate the linear velocity and traveled distance.
"""

import numpy as np
import sys
import yaml
import rclpy
from rclpy.node import Node
from marvelmind_ros2_msgs_upstream.msg import HedgePositionAddressed
from sensor_msgs.msg import IMU
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data, qos_profile_services_default
from irobot_create_msgs.msg import Mouse
from rclpy.executor import ExternalShutdownException
from message_filters import TimeSynchronizer, Subscriber, ApproximateTimeSynchronizer


class Turtlebot4KF(Node):

    def __init__(self, name):
        super().__init__(name)

        """
        Turtlebot basic parameters
        """
        with open('/turtlebot_sensor_fusion/yaml/param.yaml', 'r') as f:
            params = yaml.safe_load(f)

        robot_namespace = params[namespace]
        init_pose = np.array(params[parameters][init_pose])
        init_vel = np.array(params[parameters][init_vel])

        # Create a queue
        self.queue = list()
        # Create subscription
        """
        IMU: provides angular velocity, linear acceleration and orientation.
        In this program, use the linear acceleration as the input data.
        """

        self.sub_imu = Subscriber(self, IMU, f'{robot_namespace}/imu', qos_profile=qos_profile_sensor_data)
        """
        cmd_vel: provides the linear velocity.
        In this program, use the linear velocity as the measurement data.
        """

        self.sub_mouse = Subscriber(self, Mouse, f'{robot_namespace}/mouse', qos_profile=qos_profile_sensor_data)
        """
        marvelmind coordinates: provides the distance.
        In this program, use the distance as the measurement data. 
        """

        self.sub_marvel = Subscriber(self, HedgePositionAddressed, f'{robot_namespace}/hedge_position_addressed',
                                     qos_profile=qos_profile_sensor_data)

        queue_size = 30

        self.ts = ApproximateTimeSynchronizer(
            [self.sub_imu, self.sub_mouse, self.sub_marvel],
            queue_size,
            0.01,  # defines the delay (in seconds) with which messages can be synchronized
        )

        self.ts.registerCallback(self.all_callback)

        # Create publisher
        """
        Odometry: Contains position and velocity data.
        """
        self.pub_odom = self.create_publisher(
            Odometry,
            f'{robot_namespace}/filtered',
            qos_profile_services_default
        )

        # Set up the time
        # self.last_time = self.get_clock().now().to_msg()
        self.last_time = self.get_clock().now().seconds

        # Define the state [x,y,vx,vy]
        self.state = np.array([0., 0., 0., 0.])
        self.state[:2] = init_pose
        self.state[:4] = init_vel

        # Define the measurement [x,y,vx,vy]

        # Define the input [ax,ay]
        self.input = np.array([0., 0.])

        # Define the noise and covariance
        # self.H = np.eye(2)
        self.P = np.random.rand(4, 4)
        self.Q = np.random.rand(4, 4)
        self.R = np.random.rand(2, 2)
        self.w = np.random.normal(loc=0, scale=0.6, size=4)

        # Create time loop
        self.timer = self.create_timer(1. / 10, self.timer_callback)

    def timer_callback(self):
        measurement = self.queue.pop()
        dt = measurement[0]
        mea_vector = np.array(measurement[1:])

        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        B = np.array([[0.5 * (dt ** 2), 0],
                      [0, 0.5 * (dt ** 2)],
                      [dt, 0],
                      [0, dt]])

        H = np.eye(4)

        self.state, self.P = self.kalman_filter(A, B, self.state, self.input, H, mea_vector,
                                                self.P, self.Q, self.R, self.w)
        self.pub_odom_msg()

    def kalman_filter(self, A, B, state, input, H, measurement, P, Q, R, w):
        """

        :param state: [x,y] or [vx, vy]
        :param measurement: [x, y] or [vx, vy
        :param P: np.array((2, 2))
        :return: next_state [x,y] or [vx, vy], P_next
        """
        state_pred = np.dot(A, state) + np.dot(B, input) + w
        P_pred = np.dot(A, np.dot(P, A.T)) + Q
        y = measurement - np.dot(H, state_pred)
        s = np.dot(H, np.dot(P_pred, H.T)) + R
        K = np.dot(P_pred, np.dot(H.T, np.linalg.inv(s)))
        state_update = state_pred + np.dot(K, y)
        P_update = np.dot((np.eye(4) - np.dot(K, H)), P_pred)
        return state_update, P_update

    def all_callback(self, imu_msg, mouse_msg, marvel_msg):
        self.get_logger().info(f'Get the current measured positions and linear velocities.')
        current_time = imu_msg.header.stamp.to_sec()
        dt = current_time - self.last_time

        lin_acc = np.array(imu_msg.linear_acceleration[:2])
        self.input = lin_acc

        dx = mouse_msg.integrated_x
        dy = mouse_msg.integrated_y
        lin_vel_x = dx / dt
        lin_vel_y = dy / dt

        x = marvel_msg.x_m
        y = marvel_msg.y_m

        measurement = (dt, x, y, lin_vel_x, lin_vel_y)
        self.queue.append(measurement)
        self.last_time = current_time

    def pub_odom_msg(self):
        odom_msg = Odometry()
        current_time = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.header.stamp = current_time

        odom_msg.pose.pose.position.x = self.state[0]
        odom_msg.pose.pose.position.y = self.state[1]

        odom_msg.child_frame_id = 'base_link'
        odom_msg.twist.twist.linear.x = self.state[2]
        odom_msg.twist.twist.linear.y = self.state[3]

        self.pub_odom.publish(odom_msg)
        self.get_logger().info(f'Publishing: odom_msgs')


def main(args=None):
    rclpy.init(args=args)
    filter = Turtlebot4KF('Filter')
    try:
        rclpy.spin(filter)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        sys.exit(1)
    finally:
        rclpy.try_shutdown()
        filter.destroy_node()


if __name__ == '__main__':
    main()
