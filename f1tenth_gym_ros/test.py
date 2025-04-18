"""
Change the map in f1tenth_gym_ros/config/sim.yaml from levine to levine_blocked or levine_obs. Also change it back after implementation

"""

import rclpy
from rclpy.node import Node

import math

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive # type: ignore

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

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

        self.sub_lidar = self.create_subscription(LaserScan,lidarscan_topic,self.lidar_callback,10)
        self.pub_drive =  self.create_publisher(AckermannDriveStamped,drive_topic,10)


    # def preprocess_lidar(self, ranges,angle_increment):
    #     """ Preprocess the LiDAR scan array. Expert implementation includes:
    #         1.Setting each value to the mean over some window
    #         2.Rejecting high values (eg. > 3m)
    #     """
    #     proc_ranges = list(ranges)
    #     bubble_radius = 0.22 # since car length is 0.3m

    #     # Replace 'inf' with left/right neighbor if available
    #     for i in range(len(proc_ranges)):
    #         if math.isinf(proc_ranges[i]):
    #             left = next((proc_ranges[j] for j in range(i - 1, -1, -1) if not math.isinf(proc_ranges[j])), None)
    #             right = next((proc_ranges[j] for j in range(i + 1, len(proc_ranges)) if not math.isinf(proc_ranges[j])), None)

    #             if left is not None and right is not None:
    #                 proc_ranges[i] = (left + right) / 2.0
    #             elif left is not None:
    #                 proc_ranges[i] = left
    #             elif right is not None:
    #                 proc_ranges[i] = right
    #             else:
    #                 proc_ranges[i] = 3.0  # default max if nothing nearby

    #     for i,r in enumerate(proc_ranges):
    #         if 0.01 < r <=3.0: # Valid detection range (non-zero, non-infinite, below 3m)
    #             try:
    #                 angle_spread = math.atan(bubble_radius / r) #This computes the angle (in radians) that a 17cm radius makes at distance r using simple trigonometry:
    #                 index_spread = int(angle_spread / angle_increment) #This computes the spread of index to inflate

    #                 # Inflate bubble around index i
    #                 for j in range(i - index_spread, i + index_spread + 1):  
    #                      if 0 <= j < len(proc_ranges): #avoid checking indexes outside the array like end points
    #                          proc_ranges[j] = 0.0  # mark as occupied
    #             except ZeroDivisionError:
    #                  pass  # skip r = 0.0 cases

    #         elif r > 3.0: # if distance is more than 3m make it inf
    #             proc_ranges[i] = float('inf')

    #     return proc_ranges

    def preprocess_lidar(self, ranges, angle_increment):
        """Preprocess the LiDAR scan:
        1. Replace inf with nearest neighbor values.
        2. Find closest obstacle.
        3. Create safety bubble (set values to 0) only around that point.
        4. Reject anything above max range (e.g., 3.0).
        """

        max_range = 3.5
        bubble_radius = 0.20  # based on car width or half-diagonal
        

        proc_ranges = list(ranges)

        # Step 1: Replace 'inf' with nearest valid neighbor
        for i in range(len(proc_ranges)):
            if math.isinf(proc_ranges[i]):
                left = next((proc_ranges[j] for j in range(i - 1, -1, -1) if not math.isinf(proc_ranges[j])), None)
                right = next((proc_ranges[j] for j in range(i + 1, len(proc_ranges)) if not math.isinf(proc_ranges[j])), None)

                if left is not None and right is not None:
                    proc_ranges[i] = (left + right) / 2.0
                elif left is not None:
                    proc_ranges[i] = left
                elif right is not None:
                    proc_ranges[i] = right
                else:
                    proc_ranges[i] = max_range  # fallback

        # Step 2: Clamp long-range returns
        for i in range(len(proc_ranges)):
            if proc_ranges[i] > max_range:
                proc_ranges[i] = max_range

        # Step 3: Find the closest point and apply the bubble
        closest_idx = np.argmin(proc_ranges)
        r = proc_ranges[closest_idx]

        if r > 0.01:  # only if the point is valid
            angle_spread = math.atan(bubble_radius / r)
            index_spread = int(angle_spread / angle_increment)

            for j in range(closest_idx - index_spread, closest_idx + index_spread + 1):
                if 0 <= j < len(proc_ranges):
                    proc_ranges[j] = 0.0  # Clear the bubble

        return proc_ranges


    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
        """
        max_start = 0
        max_end = 0
        current_start = None
        max_len = 0
        for i, r in enumerate(free_space_ranges):
            if r > 0.0:  # Free space
              if current_start is None:
                current_start = i  # Start of a new gap
            else:  # Obstacle detected (r == 0.0)                                                                
                if current_start is not None:
                    current_len = i - current_start
                    if current_len > max_len:
                        max_len = current_len
                        max_start = current_start
                        max_end = i - 1
                    current_start = None  # End the current gap

        # Check if the gap continued till the end of the array (if in the end of arrary there was no 0.0, example =[0.0,1.1,2.0,2.9,0.0,2.5,2.8])
        if current_start is not None:
            current_len = len(free_space_ranges) - current_start 
            if current_len > max_len:
                max_start = current_start
                max_end = len(free_space_ranges) - 1

        return self.find_best_point(max_start,max_end,free_space_ranges)
    
    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
	    
        """
        #including edge buffer of 10percent and choosing the furthest point
        gap_width = end_i - start_i
        buffer = gap_width // 10
        max_dist = 0.0
        best_idx = start_i + buffer

        for i in range(start_i + buffer, end_i - buffer + 1):
            if math.isfinite(ranges[i]) and ranges[i] > max_dist:
                max_dist = ranges[i]
                best_idx = i

        return best_idx

    def lidar_callback(self, data):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        ranges = data.ranges

        start_idx = int((math.radians(-100) - data.angle_min)/ data.angle_increment)
        end_idx = int((math.radians(100) - data.angle_min)/ data.angle_increment)

        ranges = data.ranges[start_idx:end_idx+1]  #this are only the ranegs between -90degree and +90 degree

        free_space_ranges = self.preprocess_lidar(ranges,data.angle_increment)

        best_idx = self.find_max_gap(free_space_ranges) #it finds max gap and then finds best point

        slice_angle_min = math.radians(-90)
        target_angle = slice_angle_min + best_idx * data.angle_increment

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = target_angle
        drive_msg.drive.speed = max(0.5, 1.5 - abs(target_angle) * 3.0)
        # drive_msg.drive.speed = 1.0

        self.pub_drive.publish(drive_msg)

        self.get_logger().info(f"Target index: {best_idx}, angle: {math.degrees(target_angle):.2f}°")

        
        
        # TODO:
        #Find closest point to LiDAR

        #Eliminate all points inside 'bubble' (set them to zero) 

        #Find max length gap 

        #Find the best point in the gap 

        #Publish Drive message


def main(args=None):
    rclpy.init(args=args)
    print("GapFollow Initialized")
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()