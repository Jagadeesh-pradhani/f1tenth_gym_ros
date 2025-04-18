#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from scipy.ndimage import convolve1d
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

MAX_STEER = 0.36  # Maximum steering angle (rad)

class FollowGapController(Node):
    """
    Implements the Follow-Gap algorithm for Ackermann steering with safety bubble handling.
    Adds ability to bypass steering smoothing on sudden required changes.
    """

    def __init__(self):
        super().__init__('follow_gap_controller')

        # Algorithm parameters
        self.window_size      = self.declare_parameter('window_size', 5).value
        self.max_horizon      = self.declare_parameter('max_horizon', 2.5).value
        self.disparity_thresh = self.declare_parameter('disparity_thresh', 0.15).value
        self.bubble_radius    = self.declare_parameter(
            'bubble_radius', 0.9*(0.2032 + 2*0.0381)
        ).value

        # Steering smoothing and sudden-turn threshold
        self.steer_alpha       = self.declare_parameter('steer_alpha', 0.9).value
        self.turning_thresh    = self.declare_parameter(
            'turning_thresh', 90.0 * np.pi / 180.0
        ).value

        # PID gains and limits
        self.kp          = self.declare_parameter('kp', 1.0).value
        self.ki          = self.declare_parameter('ki', 0.0).value
        self.kd          = self.declare_parameter('kd', 0.1).value
        self.max_control = self.declare_parameter('max_control', MAX_STEER).value

        # Velocity profile
        self.low_vel         = self.declare_parameter('low_vel', 0.5).value
        self.high_vel        = self.declare_parameter('high_vel', 3.0).value
        self.vel_attenuation = self.declare_parameter('velocity_attenuation', 1.5).value

        # Internal PID state
        self.prev_error = 0.0
        self.integral   = 0.0
        self.prev_steer = 0.0

        # ROS2 interfaces
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

    def preprocess(self, ranges, rmin, rmax):
        """
        Clip sensor readings, limit horizon, and smooth with a moving average.
        """
        clipped = np.clip(ranges, rmin, rmax)
        clipped = np.clip(clipped, 0.0, self.max_horizon)
        kernel = np.ones(self.window_size) / self.window_size
        return convolve1d(clipped, kernel, mode='nearest')

    def find_bubbles(self, ranges):
        """
        Identify indices for closest obstacles and sharp disparities.
        """
        disp = np.hstack((0.0, np.diff(ranges)))
        left_disp  = np.where(disp >  self.disparity_thresh)[0] - 1
        right_disp = np.where(disp < -self.disparity_thresh)[0]
        closest_idx = np.argmin(ranges)
        return np.sort(np.hstack((closest_idx, left_disp, right_disp)))

    @staticmethod
    def find_max_gap(ranges):
        """
        Find the largest continuous zero-valued region (free space).
        Returns (start, end) indices in the original array.
        """
        padded = np.concatenate(([0.0], ranges, [0.0]))
        zeros = np.where(padded < 1e-4)[0]
        lengths = np.diff(zeros) - 1
        if lengths.size == 0:
            return 0, -1
        best = np.argmax(lengths)
        start = zeros[best] + 1 - 1
        end   = zeros[best + 1] - 1 - 1
        return start, end

    @staticmethod
    def find_best_point(start, end, ranges, angles):
        """
        Within the gap [start, end], choose the angle of maximum distance.
        If gap is empty, default to pointing straight ahead.
        """
        if end < start or start < 0 or end >= ranges.size:
            return ranges.size // 2
        sub = ranges[start:end+1]
        if sub.size == 0:
            return ranges.size // 2
        idx = np.argmax(sub)
        return start + int(idx)

    def compute_steer(self, error):
        """
        PID-based steering with optional smoothing bypass on large required change.
        """
        d_error = error - self.prev_error
        self.prev_error = error
        self.integral += error

        raw = self.kp*error + self.ki*self.integral + self.kd*d_error
        clipped = np.clip(raw, -self.max_control, self.max_control)

        # Bypass smoothing if required turn is large
        if abs(clipped) > self.turning_thresh:
            smooth = clipped
        else:
            smooth = self.steer_alpha*clipped + (1 - self.steer_alpha)*self.prev_steer

        self.prev_steer = smooth
        return smooth

    def compute_speed(self, error):
        """
        Adaptive velocity: slows down for larger steering angles.
        """
        return (self.high_vel - self.low_vel) * np.exp(-abs(error)*self.vel_attenuation) + self.low_vel

    def lidar_callback(self, msg: LaserScan):
        """
        Process LiDAR scan, apply Follow-Gap + safety bubble, then publish drive commands.
        """
        raw = np.array(msg.ranges)
        angles = np.arange(raw.size) * msg.angle_increment + msg.angle_min

        proc = self.preprocess(raw, msg.range_min, msg.range_max)
        mask = (angles > -np.pi/3) & (angles < np.pi/3)
        front_ranges = proc[mask]
        front_angles = angles[mask]

        zeros = np.zeros_like(front_ranges, dtype=bool)
        for idx in self.find_bubbles(front_ranges):
            d = front_ranges[idx]
            theta = (abs(np.arcsin(self.bubble_radius / d)) if d > self.bubble_radius else np.pi/2)
            span = int(theta / msg.angle_increment)
            lo, hi = max(idx - span, 0), min(idx + span, front_ranges.size - 1)
            zeros[lo:hi+1] = True
        front_ranges[zeros] = 0.0

        start, end = self.find_max_gap(front_ranges)
        best_idx = self.find_best_point(start, end, front_ranges, front_angles)
        error = front_angles[best_idx]

        steer = self.compute_steer(error)
        speed = self.compute_speed(error)

        drive = AckermannDriveStamped()
        drive.drive.steering_angle = steer
        drive.drive.speed = speed
        self.drive_pub.publish(drive)


def main(args=None):
    rclpy.init(args=args)
    node = FollowGapController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()