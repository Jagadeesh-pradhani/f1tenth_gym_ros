import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np

class ReactiveGapFollower(Node):
    def __init__(self):
        super().__init__('reactive_gap_follower')
        self.publisher_ = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.subscription = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.subscription  # Prevent unused variable warning
        self.angle = 0.0
        self.ranges = []

    def lidar_callback(self, lidar_info):
        # Step 1: Preprocess LIDAR information. --> take reading only from angle shown in slide 9.
        min_indx = int(0 + 20 * (1080 / 270))
        max_indx = int(1080 - 20 * (1080 / 270))
        self.ranges = np.array(lidar_info.ranges[min_indx:max_indx])  # the array containing the distances to obstacles

        # Account for nan and infs
        self.ranges = np.where(np.isnan(self.ranges) | np.isinf(self.ranges), 0.0, self.ranges)

        # Step 2: Get the closest obstacle
        closest_distance = float('inf')  # Initialize to a large number
        closest_indx = -1  # Initialize to an invalid index
        for i in range(len(self.ranges)):
            distance = np.sum(self.ranges[i-2:i+3])  # do whatever that works for you
            if distance < closest_distance:
                closest_distance = distance  # what?
                closest_indx = i  # what?

        # Step 3: Create the bubble of zeros around the closest obstacle.
        radius = 150  # This is 150 indices.
        for i in range(max(0, closest_indx - radius), min(len(self.ranges), closest_indx + radius + 1)):
            self.ranges[i] = 0.0

        # Step 4: Finding the largest gap, in the new range the maximum consecutive non-zeros.
        start, end, current_start = -1, -1, -1  # what?
        longest_duration, duration = 0.0, 0.0  # what?
        for i in range(len(self.ranges)):
            if current_start == -1 and self.ranges[i] > 0.0:  # Potential gap is discovered
                current_start = i  # Index of the first in this potential gap
            elif current_start != -1 and self.ranges[i] <= 0.0:  # A block of obstacles
                duration = i - current_start  # fill in
                if duration > longest_duration:
                    longest_duration = duration  # fill in
                    start, end = current_start, i-1  # fill in
                current_start = -1  # fill in (reset it)

        # Check if the last gap is calculated correctly.
        if current_start != -1:
            duration = len(self.ranges) - current_start  # calculate it
            if duration > longest_duration:
                longest_duration = duration  # fill in
                start, end = current_start, len(self.ranges)-1  # fill in

        # Step 5: From the largest gap, find an optimal ray to follow (naively the maximum)
        current_max = 0.0
        for i in range(start, end + 1):
            if self.ranges[i] > current_max:
                current_max = self.ranges[i]
                self.angle = lidar_info.angle_min + (i + min_indx) * lidar_info.angle_increment

        self.reactive_control()

    def reactive_control(self):
        ackermann_drive_result = AckermannDriveStamped()
        ackermann_drive_result.drive.steering_angle = self.angle

        # Do the same speed control we did last lab for speed according to angle.
        if 0 <= abs(self.angle) <= 0.17:  # ~10 degrees
            ackermann_drive_result.drive.speed = 1.3
        elif 0.17 < abs(self.angle) <= 0.35:  # ~20 degrees
            ackermann_drive_result.drive.speed = 1.0
        else:
            ackermann_drive_result.drive.speed = 0.5

        self.publisher_.publish(ackermann_drive_result)
        self.get_logger().info(f'Steering Angle: {self.angle}, Speed: {ackermann_drive_result.drive.speed}')


def main(args=None):
    rclpy.init(args=args)
    node = ReactiveGapFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
