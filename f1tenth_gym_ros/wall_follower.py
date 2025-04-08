#!/usr/bin/env python
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

import time
import math


class WallFollow(Node):
    """
    Implement Wall Following on the car
    """

    def __init__(self):
        super().__init__('wall_follow_node')

        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # TODO: create subscribers and publishers
        self.subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        # TODO: set PID gains
        self.kp = 1  # Proportional gain
        self.kd = 0  # Derivative gain
        self.ki = 0  # Integral gain

        # TODO: store history
        self.integral = 0
        self.prev_error = 0
        self.error = 0

        # TODO: store any necessary values you think you'll need
        speed = 3  # Default speed
        self.last_time = time.time()  # Track last update time
        self.counter = 0.0

    def get_range(self, range_data, angle):
        # TODO: implement
        # Calculate the perpendicular distance to the wall
        alpha = np.arctan((range_data[0] * np.cos(angle) - range_data[1]) / (range_data[0] * np.sin(angle)))
        Dt = range_data[1] * np.cos(angle)  # Distance at current time
        L = 0.5  # Constant offset length
        Dt_1 = Dt + L * np.sin(alpha)  # Future distance calculation
        return Dt_1

    #def get_error(self, range_data, dist):
    """
    Calculates the error to the wall. Follow the wall to the left (going counterclockwise in the Levine loop).
    You potentially will need to use get_range()

    Args:
    range_data: single range array from the LiDAR (D_t+1)
    dist: desired distance to the wall (0.9)

    Returns:
    error: calculated error (desired - range_data)
    """

    # TODO: implement
    #return dist - range_data

    def pid_control(self, error, velocity, dead_end, avg_L, avg_R):
        current_time = time.time()
        delta_time = current_time - self.last_time  # Time difference since last update

        self.last_time = current_time
        self.integral += self.prev_error * delta_time
        angle = self.kp * error + self.kd * (error - self.prev_error) / delta_time + self.ki * self.integral

        # TODO: Use kp, ki & kd to implement a PID controller
        # VELOCITY adjustment

        angle_0 = np.deg2rad(0)
        angle_10 = np.deg2rad(10)
        angle_20 = np.deg2rad(20)

        # Adjust velocity based on steering angle
        if angle_0 <= angle < angle_10:
            velocity = 1.5  # Higher speed for small corrections
        elif angle_10 <= angle < angle_20:
            velocity = 1.0  # Moderate speed for medium corrections
        else:
            velocity = 0.5  # Slow down for sharp turns

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = velocity
        drive_msg.drive.steering_angle = angle

        if dead_end:
            drive_msg.drive.steering_angle = -1.5  # Reverse steering to escape dead-end
            drive_msg.drive.speed = 1.0  # Slow speed for maneuvering

        self.publisher.publish(drive_msg)  # Publish drive command
        return 0.0

    def scan_callback(self, msg):
        """
        Callback function for LaserScan messages. Calculate the error and publish the drive message in this function.

        Args:
        msg: Incoming LaserScan message

        Returns:
        None
        """
        theta = 30  # Angle for distance calculation
        a = msg.ranges[780]  # Range at angle a
        b = msg.ranges[900]  # Range at angle b

        avg_r = np.average(msg.ranges[400])  # Average range on right side
        avg_l = np.average(msg.ranges[700])  # Average range on left side
        print("l:", avg_l)
        print("r:", avg_r)

        future_dist = self.get_range([a, b], np.deg2rad(theta))  # Predict future distance

        error = future_dist - 0.9  # TODO: replace with error calculated by get_error()

        velocity = 0.0  # TODO: calculate desired car velocity based on error

        # Check if the vehicle is in a dead-end
        dead_end = False
        if (abs(avg_l - avg_r) > 1 and avg_l < 2):
            dead_end = True

        # call pid_control to pass PID to ackermann/drive topic
        self.pid_control(error, velocity, dead_end, avg_l, avg_r)


def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    wall_follow_node = WallFollow()
    rclpy.spin(wall_follow_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    wall_follow_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
