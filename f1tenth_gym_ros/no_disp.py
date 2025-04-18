#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
import heapq


class ReactiveFollowGap(Node):
    """
    Implement Wall Following on the car
    This is just a template, you are free to implement your own node!
    """

    def __init__(self):
        super().__init__('reactive_node')
        # Topics & Subs, Pubs
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # TODO: Subscribe to LIDAR
        self.subscriber = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
        # TODO: Publish to drive
        self.publisher = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.rb = 20  # cm
        self.window_size = 10
        self.safe_thres = 1.2
        self.lower_idx = 180
        self.upper_idx = 900

    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        proc_ranges = []
        for i in range(0, len(ranges), self.window_size):
            mean = round(sum(ranges[i:i + self.window_size]) / self.window_size, 5)
            for j in range(self.window_size):
                proc_ranges.append(mean)
        return np.array(proc_ranges)

    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
        """
        queue = []
        i = 0
        while i < len(free_space_ranges):
            if free_space_ranges[i] >= self.safe_thres:
                safe_start_i = i
                i += 1
                while i < len(free_space_ranges) and free_space_ranges[i] >= self.safe_thres:
                    i += 1
                safe_end_i = i - 1
                if safe_end_i != safe_start_i:
                    safe_range = -(np.max(free_space_ranges[safe_start_i:safe_end_i]))
                    gap_idx = (safe_start_i, safe_end_i)
                    queue.append((safe_range, gap_idx))
            else:
                i += 1
        heapq.heapify(queue)
        return queue

    def find_best_point(self, safe_ranges, ranges):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
	    Naive: Choose the furthest point within ranges and go there
        """
        target = 0
        if len(safe_ranges) == 0:
            return np.argmax(ranges)
        else:
            while not len(safe_ranges) == 0:
                safe_start, safe_end = heapq.heappop(safe_ranges)[1]
                target = (safe_start + safe_end) // 2
                if self.lower_idx <= target <= self.upper_idx:
                    return target
            return target

    def process_bubbles(self, ranges):
        i = 0
        while i < len(ranges):
            if ranges[i] <= 0.8:
                ranges[max(0, i - self.rb):i + self.rb] = 0
                i += self.rb
            else:
                i += 1
        return ranges

    def lidar_callback(self, data):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        ranges = data.ranges
        angle_min = data.angle_min
        angle_increment = data.angle_increment
        proc_ranges = self.preprocess_lidar(ranges)

        # TODO:
        # Find closest point to LiDAR
        # closest_point = np.argmin(proc_ranges)

        # Eliminate all points inside 'bubble' (set them to zero)
        proc_ranges = self.process_bubbles(proc_ranges)

        # Find max length gap
        safe_ranges = self.find_max_gap(proc_ranges)

        # Find the best point in the gap
        i = self.find_best_point(safe_ranges, proc_ranges)
        steering_angle = angle_min + angle_increment * i
        # print(f'turn_angle: {steering_angle}')
        speed = 3.0
        if abs(steering_angle) >= np.deg2rad(150):
            speed = 0.5
        # Publish Drive message
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = steering_angle
        msg.drive.speed = speed
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()