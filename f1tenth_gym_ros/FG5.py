#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from scipy import ndimage
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

# Vehicle dimensions
VEHICLE_WIDTH = 0.2032
WHEEL_BASE = 0.0381
MAX_STEERING_ANGLE = 0.36

class GapFollowerNode(Node):
    """
    Implements the Follow-the-Gap algorithm for Ackermann steering.
    """

    def __init__(self):
        super().__init__('gap_follower')

        # Declare parameters
        self.declare_parameter('window_size', 5)
        self.declare_parameter('max_horizon', 2.5)
        self.declare_parameter('disparity_threshold', 0.15)
        self.declare_parameter('bubble_radius', 0.9 * (VEHICLE_WIDTH + 2 * WHEEL_BASE))
        self.declare_parameter('kp', 1.0)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.1)
        self.declare_parameter('max_control', MAX_STEERING_ANGLE)
        self.declare_parameter('low_speed', 0.5)
        self.declare_parameter('high_speed', 3.0)
        self.declare_parameter('speed_attenuation', 1.5)
        self.declare_parameter('steering_smooth', 0.3)

        # Initialize control state
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.prev_steering = 0.0

        # Subscriber & Publisher
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)

    def preprocess_scan(self, scan_ranges, min_range, max_range):
        """
        Clip invalid values, limit range, and smooth using moving average.
        """
        clipped = np.clip(scan_ranges, min_range, max_range)
        horizon = self.get_parameter('max_horizon').value
        clipped = np.clip(clipped, 0.0, horizon)
        window = np.ones(self.get_parameter('window_size').value) / self.get_parameter('window_size').value
        return ndimage.convolve1d(clipped, window, mode='nearest')

    def detect_disparities(self, ranges):
        """
        Identify indices where the difference between consecutive measurements
        exceeds the disparity threshold.
        """
        thresh = self.get_parameter('disparity_threshold').value
        diff = np.hstack((0.0, np.diff(ranges)))
        left = np.where(diff > thresh)[0] - 1
        right = np.where(diff < -thresh)[0]
        return np.sort(np.hstack((left, right)))

    def compute_bubble_indices(self, ranges):
        """
        Combine closest point and disparity indices to form obstacles.
        """
        closest = np.argmin(ranges)
        disparity_pts = self.detect_disparities(ranges)
        return np.sort(np.hstack((closest, disparity_pts)))

    @staticmethod
    def find_largest_gap(free_ranges):
        """
        Locate the largest contiguous sequence of non-zero measurements.
        """
        padded = np.hstack((0.0, free_ranges, 0.0))
        zeros = np.where(padded < 1e-4)[0]
        gaps = np.diff(zeros) - 1
        valid = np.where(gaps > 0)[0]
        bounds = np.vstack((zeros[valid], zeros[valid + 1] - 2)).T
        # Select gap with maximum area
        areas = []
        for start, end in bounds:
            seg = free_ranges[start:end+1]
            areas.append(np.sum(seg**2))
        idx = int(np.argmax(areas))
        return bounds[idx]

    @staticmethod
    def select_best_point(start, end, ranges, angles):
        """
        Choose the point in the gap with maximum range, closest to center angle.
        """
        segment = ranges[start:end+1]
        segment_angles = angles[start:end+1]
        max_dist = np.max(segment)
        candidates = np.where(segment == max_dist)[0]
        best = candidates[np.argmin(np.abs(segment_angles[candidates]))]
        return start + best

    def compute_steering(self, error):
        """
        PID controller for steering.
        """
        kp = self.get_parameter('kp').value
        ki = self.get_parameter('ki').value
        kd = self.get_parameter('kd').value
        max_ctrl = self.get_parameter('max_control').value
        smooth = self.get_parameter('steering_smooth').value

        derivative = error - self.prev_error
        self.prev_error = error
        self.integral_error += error

        raw = kp * error + ki * self.integral_error + kd * derivative
        clipped = np.clip(raw, -max_ctrl, max_ctrl)
        steer = smooth * clipped + (1 - smooth) * self.prev_steering
        self.prev_steering = steer
        return steer

    def compute_speed(self, error):
        """
        Exponential decay speed profile based on steering error.
        """
        low = self.get_parameter('low_speed').value
        high = self.get_parameter('high_speed').value
        atten = self.get_parameter('speed_attenuation').value
        return (high - low) * np.exp(-abs(error) * atten) + low

    def scan_callback(self, scan: LaserScan):
        # Convert to numpy
        ranges = np.array(scan.ranges)
        angles = np.arange(len(ranges)) * scan.angle_increment + scan.angle_min

        # Preprocess and extract front sector (-60° to +60°)
        proc = self.preprocess_scan(ranges, scan.range_min, scan.range_max)
        mask = (angles > -np.pi/3) & (angles < np.pi/3)
        front = proc[mask]
        front_angles = angles[mask]

        # Zero out bubble (obstacle) regions
        bubbles = self.compute_bubble_indices(front)
        zero_mask = np.zeros_like(front, dtype=bool)
        radius = self.get_parameter('bubble_radius').value
        for idx in bubbles:
            dist = front[idx]
            theta = np.arcsin(min(radius / dist, 1.0)) if dist > radius else np.pi/2
            span = int(theta / scan.angle_increment)
            low = max(idx - span, 0)
            high = min(idx + span, len(front) - 1)
            zero_mask[low:high] = True
        front[zero_mask] = 0.0

        # Find gap and best point
        start, end = self.find_largest_gap(front)
        best = self.select_best_point(start, end, front, front_angles)

        # Compute control commands
        error = front_angles[best]
        steering = self.compute_steering(error)
        speed = self.compute_speed(error)

        # Publish drive message
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = steering
        msg.drive.speed = speed
        self.drive_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GapFollowerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
