#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import math
import time

class WallFollower(Node):
    def __init__(self):
        super().__init__('wall_follower')
        # Subscriber to LIDAR scan and publisher for driving commands.
        self.scan_subscriber = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        # PID Parameters (tweak as needed)
        self.kp = 1.0
        self.ki = 0.0
        self.kd = 0.0

        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = time.time()

        # Desired distance from the wall in meters
        self.desired_distance = 0.9

    def get_range(self, range_data, angle):
        """
        Use two LIDAR measurements to predict future perpendicular distance.
        Args:
            range_data: list of two range values [a, b]
            angle: angle (radians) between the two measurements
        Returns:
            Predicted distance Dt_1 (float)
        """
        a, b = range_data
        # Avoid division by zero or invalid data
        if a <= 0 or b <= 0:
            self.get_logger().warn("Invalid LIDAR readings for wall prediction, using default distance.")
            return self.desired_distance
        alpha = math.atan((a * math.cos(angle) - b) / (a * math.sin(angle) + 1e-6))
        Dt = b * math.cos(angle)
        L = 0.5  # Lookahead distance (tunable)
        Dt_1 = Dt + L * math.sin(alpha)
        return Dt_1

    def pid_control(self, error):
        """
        Compute steering using a PID approach.
        """
        current_time = time.time()
        dt = current_time - self.last_time if current_time - self.last_time > 0 else 1e-6

        self.integral += error * dt
        derivative = (error - self.prev_error) / dt

        steering = self.kp * error + self.ki * self.integral + self.kd * derivative

        self.prev_error = error
        self.last_time = current_time

        return steering

    def scan_callback(self, msg):
        """
        Process the LIDAR scan, compute error from the wall, and publish a drive command.
        """
        # Select two points from the left side for wall detection.
        # These indices assume a 1080-point scan. Adjust if necessary.
        try:
            a = msg.ranges[780]  # nearer left measurement
            b = msg.ranges[900]  # farther left measurement
        except IndexError:
            # Fallback to center values if scan length is different.
            mid = len(msg.ranges) // 2
            a = msg.ranges[mid-50]
            b = msg.ranges[mid]

        # Angle between the two LIDAR beams (adjust if needed)
        theta = math.radians(30)
        future_distance = self.get_range([a, b], theta)
        # Error: positive if too far, negative if too close.
        error = self.desired_distance - future_distance

        steering = self.pid_control(error)

        # Adjust the forward speed based on the magnitude of the steering action.
        abs_steer_deg = abs(math.degrees(steering))
        if abs_steer_deg < 10:
            velocity = 1.5
        elif abs_steer_deg < 20:
            velocity = 1.0
        else:
            velocity = 0.5

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering
        drive_msg.drive.speed = velocity
        self.drive_publisher.publish(drive_msg)

        self.get_logger().info(f"WallFollow: steering={steering:.2f}, speed={velocity:.2f}, error={error:.2f}")

def main(args=None):
    rclpy.init(args=args)
    wall_follower = WallFollower()
    rclpy.spin(wall_follower)
    wall_follower.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
