#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np
import math

class GapFollower(Node):
    def __init__(self):
        super().__init__('gap_follower')
        self.scan_subscriber = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        # Parameters for speed control
        self.high_speed = 1.3
        self.medium_speed = 1.0
        self.low_speed = 0.5

    def lidar_callback(self, msg):
        """
        Process LIDAR data to select a steering angle based on the largest gap.
        """
        # Work with a portion of the LIDAR scan that represents the forward view.
        ranges = np.array(msg.ranges)
        num_readings = len(ranges)

        # Define working indices to avoid problematic boundary readings.
        min_index = int(20 * (num_readings / 270))
        max_index = int(num_readings - 20 * (num_readings / 270))
        proc_ranges = ranges[min_index:max_index].copy()

        # Replace nan or inf with a large number (assuming clear space)
        proc_ranges = np.where(np.isnan(proc_ranges) | np.isinf(proc_ranges), 10.0, proc_ranges)

        # Step 1: Find the closest obstacle using a sliding window
        closest_sum = float('inf')
        closest_idx = -1
        window_size = 5  # sliding window size
        for i in range(window_size, len(proc_ranges) - window_size):
            window_sum = np.sum(proc_ranges[i - window_size//2 : i + window_size//2 + 1])
            if window_sum < closest_sum:
                closest_sum = window_sum
                closest_idx = i

        # Step 2: Create a "bubble" around the closest obstacle.
        bubble_radius = 150  # number of indices to clear (tune based on sensor resolution)
        start_bubble = max(0, closest_idx - bubble_radius)
        end_bubble = min(len(proc_ranges) - 1, closest_idx + bubble_radius)
        proc_ranges[start_bubble:end_bubble+1] = 0.0

        # Step 3: Find the largest gap (contiguous segment with nonzero values).
        max_gap_size = 0
        gap_start = 0
        gap_end = 0
        current_start = None
        for i, distance in enumerate(proc_ranges):
            if distance > 0.0:
                if current_start is None:
                    current_start = i
            else:
                if current_start is not None:
                    gap_size = i - current_start
                    if gap_size > max_gap_size:
                        max_gap_size = gap_size
                        gap_start, gap_end = current_start, i - 1
                    current_start = None
        # Edge case: gap extends till the end
        if current_start is not None:
            gap_size = len(proc_ranges) - current_start
            if gap_size > max_gap_size:
                max_gap_size = gap_size
                gap_start, gap_end = current_start, len(proc_ranges) - 1

        # Step 4: Within the largest gap, choose the index with maximum distance.
        best_distance = 0.0
        best_index = gap_start
        for i in range(gap_start, gap_end + 1):
            if proc_ranges[i] > best_distance:
                best_distance = proc_ranges[i]
                best_index = i

        # Convert best index back to full LIDAR index and compute the corresponding angle.
        chosen_angle = msg.angle_min + (best_index + min_index) * msg.angle_increment

        # Step 5: Set speed depending on the steering magnitude.
        abs_angle = abs(chosen_angle)
        if abs_angle < math.radians(10):
            velocity = self.high_speed
        elif abs_angle < math.radians(20):
            velocity = self.medium_speed
        else:
            velocity = self.low_speed

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = chosen_angle
        drive_msg.drive.speed = velocity
        self.drive_publisher.publish(drive_msg)

        self.get_logger().info(f"GapFollow: steering={chosen_angle:.2f}, speed={velocity:.2f}, best_distance={best_distance:.2f}")

def main(args=None):
    rclpy.init(args=args)
    gap_follower = GapFollower()
    rclpy.spin(gap_follower)
    gap_follower.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
