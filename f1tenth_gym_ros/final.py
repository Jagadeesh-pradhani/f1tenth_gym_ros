import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np
import math

class ReactiveGapFollower(Node):
    def __init__(self):
        super().__init__('reactive_gap_follower')
        self.publisher_ = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.lidar_subscription = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.odom_subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.angle = 0.0
        self.ranges = []
        self.vx = 0.0  # Forward speed from odometry
        # Safety parameter: minimum acceptable iTTC (can be tuned further)
        self.min_ittc_threshold = 0.5  

    def odom_callback(self, msg):
        # Extract the vehicle's forward velocity (assuming along the x-axis)
        self.vx = msg.twist.twist.linear.x

    def lidar_callback(self, lidar_info):
        # Step 1: Preprocess LIDAR data from front 270 degrees excluding 20 degree margins on each side.
        min_indx = int(0 + 20 * (1080 / 270))
        max_indx = int(1080 - 20 * (1080 / 270))
        self.ranges = np.array(lidar_info.ranges[min_indx:max_indx])
        self.ranges = np.where(np.isnan(self.ranges) | np.isinf(self.ranges), 0.0, self.ranges)

        # Step 2: Identify the closest obstacle (using a small smoothing window)
        closest_distance = float('inf')
        closest_indx = -1
        for i in range(len(self.ranges)):
            window = self.ranges[max(0, i - 2):min(len(self.ranges), i + 3)]
            window_sum = np.sum(window)
            if window_sum < closest_distance and window_sum > 0.0:
                closest_distance = window_sum
                closest_indx = i

        # Step 3: Create a safety bubble by zeroing out a region around the closest obstacle
        bubble_radius = 150  # indices for safety bubble (adjustable)
        start_bubble = max(0, closest_indx - bubble_radius)
        end_bubble = min(len(self.ranges), closest_indx + bubble_radius + 1)
        self.ranges[start_bubble:end_bubble] = 0.0

        # Step 4: Identify the largest gap
        best_start, best_end = -1, -1
        longest_gap = 0
        current_start = -1
        for i in range(len(self.ranges)):
            if current_start == -1 and self.ranges[i] > 0.0:
                current_start = i  # potential start of a gap
            elif current_start != -1 and self.ranges[i] <= 0.0:
                gap_length = i - current_start
                if gap_length > longest_gap:
                    longest_gap = gap_length
                    best_start, best_end = current_start, i - 1
                current_start = -1
        # Check if a gap is open until the end of the array.
        if current_start != -1:
            gap_length = len(self.ranges) - current_start
            if gap_length > longest_gap:
                best_start, best_end = current_start, len(self.ranges) - 1

        if best_start == -1 or best_end == -1:
            self.angle = 0.0
            self.reactive_control(safe_stop=True)
            return

        # Step 5: Evaluate candidate beams in the largest gap using a score that favors both safety (iTTC) and forward direction.
        best_index = best_start
        best_score = -1.0

        for i in range(best_start, best_end + 1):
            current_range = self.ranges[i]
            beam_angle = lidar_info.angle_min + (i + min_indx) * lidar_info.angle_increment
            projected_velocity = self.vx * math.cos(beam_angle)
            if projected_velocity < 0.01:
                continue

            # Compute the instantaneous time-to-collision
            ittc = current_range / projected_velocity

            # Only consider beams meeting the safety margin
            if ittc >= self.min_ittc_threshold:
                # Weight the iTTC with cosine to prefer beams closer to straight ahead
                score = ittc * math.cos(beam_angle)
                if score > best_score:
                    best_score = score
                    best_index = i

        # Fallback: if none of the beams reach the iTTC threshold, pick the beam with the maximum range in the gap.
        if best_score < 0:
            current_max = 0.0
            for i in range(best_start, best_end + 1):
                if self.ranges[i] > current_max:
                    current_max = self.ranges[i]
                    best_index = i
            self.get_logger().warn("No beam met the iTTC threshold; defaulting to maximum range beam.")

        # Compute the steering angle using the chosen beam index.
        self.angle = lidar_info.angle_min + (best_index + min_indx) * lidar_info.angle_increment
        self.reactive_control()

    def reactive_control(self, safe_stop=False):
        ackermann_drive_result = AckermannDriveStamped()
        if safe_stop:
            ackermann_drive_result.drive.steering_angle = 0.0
            ackermann_drive_result.drive.speed = 0.0
            self.get_logger().info("No valid gap found. Stopping the vehicle.")
        else:
            ackermann_drive_result.drive.steering_angle = self.angle
            # Speed control based on steering angle magnitude.
            abs_angle = abs(self.angle)
            if abs_angle <= 0.17:   # ~10 degrees
                ackermann_drive_result.drive.speed = 1.3
            elif abs_angle <= 0.35:  # ~20 degrees
                ackermann_drive_result.drive.speed = 1.0
            else:
                ackermann_drive_result.drive.speed = 0.5
            self.get_logger().info(f"Steering Angle: {self.angle:.3f}, Speed: {ackermann_drive_result.drive.speed:.2f}")
        self.publisher_.publish(ackermann_drive_result)

def main(args=None):
    rclpy.init(args=args)
    node = ReactiveGapFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
