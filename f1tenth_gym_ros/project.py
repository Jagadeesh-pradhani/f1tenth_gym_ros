#!/usr/bin/env python
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

import numpy as np
import time
import math


class AutonomousDriver(Node):
    """
    This node implements a fused autonomous driving controller.
    It uses a wall–following (PID–based) control law when the front
    region is clear and switches to reactive gap–following when an obstacle
    is detected ahead.
    """

    def __init__(self):
        super().__init__('autonomous_driver')
        # Subscribe to the LIDAR scan and publish drive commands
        self.subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        # ------ Parameters for Wall Following (PID) ------
        self.kp = 1.0  # Proportional gain
        self.ki = 0.0  # Integral gain (tune if needed)
        self.kd = 0.0  # Derivative gain (tune if needed)
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = time.time()
        self.desired_distance = 0.9  # desired distance from the wall

        # ------ General Parameters ------
        self.front_threshold = 1.5  # if an obstacle in the front is closer than this, use gap following

    def get_range(self, range_data, angle):
        """
        Predicts a future (projected) distance to the wall based on two LIDAR
        measurements and the angle between them.
        Args:
          range_data: list with two range measurements [a, b]
          angle: the angle (in radians) between the two measurements
        Returns:
          Dt_1: the predicted distance after a short lookahead.
        """
        # Protect against division by zero if a reading is zero.
        if range_data[0] == 0 or range_data[1] == 0:
            return 0.0
        # Calculate the angle alpha between the readings.
        alpha = math.atan((range_data[0] * math.cos(angle) - range_data[1])
                          / (range_data[0] * math.sin(angle) + 1e-6))
        Dt = range_data[1] * math.cos(angle)
        L = 0.5  # lookahead distance (can be tuned)
        Dt_1 = Dt + L * math.sin(alpha)
        return Dt_1

    def pid_control(self, error):
        """
        Computes a PID control action (steering angle) using the error value.
        """
        current_time = time.time()
        dt = current_time - self.last_time if (current_time - self.last_time) > 0 else 1e-6

        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        steering = self.kp * error + self.ki * self.integral + self.kd * derivative

        self.prev_error = error
        self.last_time = current_time
        return steering

    def wall_follow_control(self, msg):
        """
        Uses a pair of LIDAR measurements taken from the left side to predict
        the future distance from the wall and then computes an error.
        A PID controller computes the steering angle required to maintain a safe distance.
        """
        # The following indices are chosen for a left wall. If you wish to follow the right
        # side or modify the angle, update the indices as needed.
        try:
            a = msg.ranges[780]  # reading from a point on the left
            b = msg.ranges[900]  # reading further to the left
        except IndexError:
            a = msg.ranges[len(msg.ranges) // 2]
            b = msg.ranges[-1]

        theta = math.radians(30)  # angle between the two readings (30 degrees)
        future_distance = self.get_range([a, b], theta)
        # Error is the difference between the desired and predicted distances.
        error = self.desired_distance - future_distance

        steering = self.pid_control(error)
        # Adjust forward velocity based on the magnitude of steering.
        abs_steer = abs(steering)
        if abs_steer < math.radians(10):
            velocity = 1.5  # small correction → higher speed
        elif abs_steer < math.radians(20):
            velocity = 1.0
        else:
            velocity = 0.5  # sharp turn → slow down

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering
        drive_msg.drive.speed = velocity

        self.publisher.publish(drive_msg)
        self.get_logger().info(f"WallFollowing mode: steering={steering:.2f}, speed={velocity:.2f}")

    def gap_follow_control(self, msg, ranges):
        """
        Implements a gap following strategy to steer away from obstacles.
        The steps are:
          1. Preprocess the range data (exclude readings near the LIDAR boundaries).
          2. Find the closest obstacle by aggregating over a small window.
          3. Zero out (create a "bubble") around the closest obstacle.
          4. Identify the largest contiguous gap in the LIDAR data.
          5. Select the best point (with maximum distance) within the gap and compute the corresponding angle.
        """
        num_readings = len(ranges)
        # Exclude the extreme boundaries which might have poor readings.
        min_indx = int(20 * (num_readings / 270))
        max_indx = int(num_readings - 20 * (num_readings / 270))
        proc_ranges = np.array(ranges[min_indx:max_indx])
        # Replace nan or infinite values with 0.
        proc_ranges = np.where(np.isnan(proc_ranges) | np.isinf(proc_ranges), 0.0, proc_ranges)

        # Find the closest obstacle by summing over a sliding window.
        closest_distance = float('inf')
        closest_indx = -1
        for i in range(2, len(proc_ranges) - 2):
            window_sum = np.sum(proc_ranges[i-2:i+3])
            if window_sum < closest_distance:
                closest_distance = window_sum
                closest_indx = i

        # Create a "bubble" around the closest obstacle by zeroing out nearby readings.
        bubble_radius = 150  # number of indices to clear (tunable)
        start_bubble = max(0, closest_indx - bubble_radius)
        end_bubble = min(len(proc_ranges)-1, closest_indx + bubble_radius)
        proc_ranges[start_bubble:end_bubble+1] = 0.0

        # Identify the largest gap (consecutive non-zero values).
        max_gap = 0
        gap_start = 0
        gap_end = 0
        current_start = None
        current_length = 0
        for i, distance in enumerate(proc_ranges):
            if distance > 0:
                if current_start is None:
                    current_start = i
                    current_length = 1
                else:
                    current_length += 1
            else:
                if current_start is not None:
                    if current_length > max_gap:
                        max_gap = current_length
                        gap_start = current_start
                        gap_end = i - 1
                    current_start = None
                    current_length = 0
        # Check if a gap extends until the end.
        if current_start is not None and current_length > max_gap:
            max_gap = current_length
            gap_start = current_start
            gap_end = len(proc_ranges) - 1

        # If no gap is found, choose straight ahead.
        if max_gap == 0:
            chosen_angle = 0.0
        else:
            # Within the largest gap, select the index with the maximum distance.
            best_distance = 0.0
            best_index = gap_start
            for i in range(gap_start, gap_end + 1):
                if proc_ranges[i] > best_distance:
                    best_distance = proc_ranges[i]
                    best_index = i
            # Convert best_index (within proc_ranges) back to the full LIDAR index and then compute the corresponding angle.
            chosen_angle = msg.angle_min + (best_index + min_indx) * msg.angle_increment

        # Set the speed based on the steering magnitude.
        abs_angle = abs(chosen_angle)
        if abs_angle < math.radians(10):
            velocity = 1.3
        elif abs_angle < math.radians(20):
            velocity = 1.0
        else:
            velocity = 0.5

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = chosen_angle
        drive_msg.drive.speed = velocity

        self.publisher.publish(drive_msg)
        self.get_logger().info(f"GapFollowing mode: steering={chosen_angle:.2f}, speed={velocity:.2f}")

    def scan_callback(self, msg):
        """
        Callback to process incoming LIDAR data.
        The routine first determines whether an obstacle is present in front.
        If yes, it runs the gap following (obstacle avoidance) routine.
        Otherwise, it follows the wall using a PID approach.
        """
        # Convert the incoming ranges to a numpy array.
        ranges = np.array(msg.ranges)
        # Replace zeros (which can be spurious) with a high value for obstacle detection.
        proc_ranges = np.where(ranges == 0, 10.0, ranges)

        # Define a front sector (here 60 readings centered at the middle of the array).
        center_index = len(proc_ranges) // 2
        front_sector = proc_ranges[center_index - 30:center_index + 30]
        min_front = np.min(front_sector)

        if min_front < self.front_threshold:
            # If any object is too close ahead, switch to gap following.
            self.get_logger().info("Obstacle detected ahead - switching to gap following")
            self.gap_follow_control(msg, ranges)
        else:
            # Otherwise, use wall following control.
            self.get_logger().info("No immediate obstacle - using wall following")
            self.wall_follow_control(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AutonomousDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
