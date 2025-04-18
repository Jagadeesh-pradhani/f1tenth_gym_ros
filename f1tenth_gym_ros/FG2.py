import math
from typing import List, Any
import rclpy
from rclpy.node import Node
from enum import Enum
from copy import deepcopy

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from std_msgs.msg import String
from rcl_interfaces.msg import SetParametersResult
from threading import Lock

import gap_follow.lidarutils as lu
import gap_follow.gap_follow_utils as gf

class FollowGapState(Enum):
    INIT = 1
    MOVING_STRAIGHT = 2
    DISPARITY_CONTROL = 3
    # STUCK=4

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
        processed_scan_topic = '/processed_scan'

        # Set up separate parameter dictionary and then declare the parameters.
        self.__parameters_mutex = Lock()
        self.__parameters = {
            "gap_scan_angle_range_deg": 90,
            "side_safety_dist_minimum_m": 0.3,
            "moving_straight_safety_timeout": 5,
            "gap_depth_threshold_m": 1.4, # Lower speed, lower gap threshold. TODO Should calculate dynamically with current speed.
            "range_upper_bound_m": 3,
            "disparity_threshold_m": 0.3,
            "lidar_angle_min_rad": -2.3499999046325684,
            "lidar_angle_max_rad": 2.3499999046325684,
            "lidar_angle_increment_rad": 0.004351851996034384,
            "lidar_num_ranges": 1080
        }
        self.declare_parameters(namespace="", 
                                parameters=[(str(key), self.__parameters[key]) for key in self.__parameters])

        # Register a parameter callback.
        self.add_on_set_parameters_callback(callback=self.__parameter_callback)

        # Create a scan subscriber.
        self.__scan_subscriber = self.create_subscription(msg_type=LaserScan,
                                                          topic=lidarscan_topic, 
                                                          callback=self.__lidar_callback, 
                                                          qos_profile=10)
        # Create a drive publisher.
        self.__drive_publisher = self.create_publisher(msg_type=AckermannDriveStamped,
                                                       topic=drive_topic,
                                                       qos_profile=10)
        # Create a new LiDAR publisher that will publish to a topic for the
        # processed LiDAR readings (preprocessed+disparity-extended).
        self.__processed_scan_publisher = self.create_publisher(msg_type=LaserScan,
                                                                topic=processed_scan_topic,
                                                                qos_profile=10)
        
        # Variable for robot width.
        self.__car_width_m = 0.2032
        # TODO:
        # Added this "fudge factor" for it to be able to pad disparities a tiny
        # bit less and get through tight gaps--but more risky! On a safe
        # car--we've stick with the width strictly, but willing to take some
        # risk here. Might be able to use a smaller fudge factor, too--I just
        # jumped up to subtracting half its width!

        # Store last drive message.
        # TODO: add some sort of interface functions needed to get/set this!
        # Right now, you can modify/access it from anywhere in the node and I'm
        # already worried that it's going to get modified without me realizing
        # it!
        self.__last_drive_message = AckermannDriveStamped()

        # TODO: Call functions or add code here to initialize values that can be
        # computed up front / at initialization. Writing a function could be
        # good for the sake of being able to recompute those values at runtime
        # as a part of the parameter callback (if one of these values depends on
        # one or more parameters).
        
        # Grab relevant LiDAR values to compute these ranges with.
        angle_increment = self.__get_local_parameter("lidar_angle_increment_rad")
        angle_min = self.__get_local_parameter("lidar_angle_min_rad")
        angle_max = self.__get_local_parameter("lidar_angle_max_rad")
        num_ranges = self.__get_local_parameter("lidar_num_ranges")
        gap_scan_angle_range_rad = math.radians(self.__get_local_parameter("gap_scan_angle_range_deg"))
        # Get lower left side index range.
        self.__left_start_angle_rad = gap_scan_angle_range_rad
        self.__left_end_angle_rad = self.__get_local_parameter("lidar_angle_max_rad")
        self.__left_side_index_range = lu.get_index_range_from_angles(start_angle_rad=self.__left_start_angle_rad,
                                                                      end_angle_rad=self.__left_end_angle_rad,
                                                                      angle_min_rad=angle_min,
                                                                      angle_max_rad=angle_max,
                                                                      angle_increment_rad=angle_increment,
                                                                      num_ranges=num_ranges)
        # Get lower right side index range.
        self.__right_start_angle_rad = self.__get_local_parameter("lidar_angle_min_rad")
        self.__right_end_angle_rad = -gap_scan_angle_range_rad
        self.__right_side_index_range = lu.get_index_range_from_angles(start_angle_rad=self.__right_start_angle_rad,
                                                                      end_angle_rad=self.__right_end_angle_rad,
                                                                      angle_min_rad=angle_min,
                                                                      angle_max_rad=angle_max,
                                                                      angle_increment_rad=angle_increment,
                                                                      num_ranges=num_ranges)
        # Get the (main) middle index range.
        self.__middle_start_angle_rad = -gap_scan_angle_range_rad
        self.__middle_end_angle_rad = gap_scan_angle_range_rad
        self.__middle_index_range = lu.get_index_range_from_angles(start_angle_rad=self.__middle_start_angle_rad,
                                                                   end_angle_rad=self.__middle_end_angle_rad,
                                                                   angle_min_rad=angle_min,
                                                                   angle_max_rad=angle_max,
                                                                   angle_increment_rad=angle_increment,
                                                                   num_ranges=num_ranges)
        # NOTE: If I don't want to have to deal with any rounding issues here,
        # could really just set the middle index range to +1 from the last of
        # the left, and -1 from the first of the right. I.e., like:
        # self.__middle_index_range = lu.IndexRange(starting_index=self.__right_side_index_range.ending_index,
        #                                        ending_index=self.__left_side_index_range.starting_index)

        # State Variable.
        self.__current_state = FollowGapState.INIT

        # TODO: Figure out a better way to implement this.
        self.__moving_straight_timer = 0

    def __get_local_parameter(self, param_name: str) -> Any:
        """Function to get the value of a parameter from the parameter
        dictionary maintained locally by the node.

        Args:
            param_name (str): The key for the parameter in the self.__parameters
            dictionary.

        Raises:
            exc: KeyError exception if the provided param_name isn't a key in
            the self.__parameters dictionary.

        Returns:
            Any: The value of the parameter if it exists in the parameters
            dictionary.
        """
        with self.__parameters_mutex:
            try:
                value = self.__parameters[param_name]
            except KeyError as exc:
                self.get_logger().error(f"Failed to get parameter with name {param_name}. Key not found in parameter dictionary!")
                raise exc
            else:
                return value
        
    def __parameter_callback(self, params: List[rclpy.Parameter]) -> SetParametersResult:
        """Function called whenever any of the node's parameters are updated and
        updates a local copy of the parameter used during publishing.
        """
        with self.__parameters_mutex:
            for param in params:
                if param.name in list(self.__parameters):
                    self.__parameters[param.name] = param.value
                    self.get_logger().info(f"Received updated version of parameter {str(param.name)}, new value {param.value}")
            return SetParametersResult(successful=True)
        
    def publish_control(self, 
                        new_steering_angle: float, 
                        new_speed: float) -> None:
        """Helper function that handles publishing the provided steering angle
        and speed to the drive topic. Constructs an AckermannDriveStamped
        message and publishes it to the drive topic.

        :param new_steering_angle: The steering angle (in radians) to be
            commanded. 
        :type steering_angle: float
        :param new_speed: The new speed (in m/s) to be commanded.
        :type new_speed: float
        """
        # Construct new Ackermann drive message, set speed and steering
        # angle.
        drive_message = AckermannDriveStamped()
        drive_message.drive.steering_angle = new_steering_angle
        drive_message.drive.speed = new_speed
        # Store this latest drive message.
        self.__last_drive_message = drive_message
        # Publish the populated message to the drive topic.
        self.__drive_publisher.publish(drive_message)
        return
    
    def __publish_processed_ranges(self, ranges: List[float], original_scan_message: LaserScan) -> None:
        """Takes a ranges array (originally from a LaserScan message) that has
        been modified after preprocessing steps and publishes it back out as a
        new LaserScan message.

        Args:
            ranges (List[float]): Updated ranges array.
            original_scan_message (LaserScan): Original scan message. Other,
            non-modified values are extracted from here.
        """
        new_scan_message = deepcopy(original_scan_message)
        new_scan_message.ranges = ranges
        self.__processed_scan_publisher.publish(new_scan_message)
    
    def velocity_from_steering_angle(self, steering_angle: float) -> float:
        """Piecewise function that returns a suitable longitudinal velocity
        given a steering angle. Recommended configuration from lab handout:

        If the steering angle is between 0 degrees and 10 degrees, the car
        should drive at 1.5 meters per second.
        If the steering angle is between 10 degrees and 20 degrees, the speed
        should be 1.0 meters per second.
        Otherwise, the speed should be 0.5 meters per second.

        :param steering_angle: Steering angle (in radians) as measured from the
            x-axis of the base link.
        :type steering_angle: float
        :return: The "recommended" longitudinal velocity (in m/s). 
        :rtype: float
        """
        # return 0.6
        steering_angle_deg = math.degrees(steering_angle)
        longitudinal_velocity = 0
        if steering_angle_deg <= 10:
            longitudinal_velocity = 0.95
        elif steering_angle_deg <= 20:
            longitudinal_velocity = 0.75
        else:
            longitudinal_velocity = 0.5
        return longitudinal_velocity

    # TODO: Move this to gap_follow_utils.
    # def speed_from_selected_gap_max_depth(self, gap_max_depth) -> float:
    #     # Right now, only really doing proportional control.
    #     return gap_max_depth


    def get_gap_depth_threshold_from_speed(self, current_speed: float) -> float:
        # Implementing as piecewise function for now, as speed is also
        # piecewise.
        # if current_speed < 0.5:
        #     return 1.0
        # if current_speed < 0.6:
        #     return 1.3
        # if current_speed < 1.0:
        #     return 1.5
        # return 2.0

        # if current_speed < 0.5:
        #     return current_speed*3
        # if current_speed < 0.75:
        #     return current_speed*2
        # if current_speed < 1.0:
        #     return current_speed*2
        # return 1.5*math.sqrt(current_speed)
        return float(np.clip(math.log10(100*current_speed + 1)/1.2, a_min=0.2, a_max=4.0))
    
    # TODO: Move this function (and the above function) into gap_follow_utils.
    # Doesn't really need to be part of the node class itself.
    def speed_from_gap_max_depth(self, 
                                 gap_max_depth: float,
                                 max_speed: float,
                                 min_speed: float) -> float:
        return float(np.clip(math.log10(10*gap_max_depth+1)/0.5, min_speed, max_speed))

    def __sides_too_close(self, ranges: List[float]) -> bool:
        """Returns whether or not the car is too close to an obstacle on either
        one of its sides. Uses the precomputed side 

        Args:
            ranges (List[float]): List of range values from the LaserScan to be
            used to check for car's proximity to nearby obstacles on its sides.

        Returns:
            bool: True if the car is within the side_safety_dist_minimum_m
            threshold on either side, False if not.
        """
        # First, grab the minimum distance threshold that we'll use for both
        # sides.
        minimum_distance_threshold = self.__get_local_parameter("side_safety_dist_minimum_m")
        # First, check to see if it's too close to anything on its right side.
        right_indices = self.__right_side_index_range
        if gf.ranges_under_threshold(ranges=ranges,
                                  range_indices=right_indices,
                                  minimum_distance_m=minimum_distance_threshold):
            return True
        # If the right is fine, check to left to make sure everything is okay.
        left_indices = self.__left_side_index_range
        if gf.ranges_under_threshold(ranges=ranges,
                                  range_indices=left_indices,
                                  minimum_distance_m=minimum_distance_threshold):
            return True
        # Otherwise, it's not too close on either side, return False.
        return False

    def __going_to_hit_wall(self, ranges: List[float], last_steering_angle_rad: float) -> bool:
        """Returns whether or not the car is currently bound to hit the wall
        based on its current distance to obstacles on either of its sides and
        its current steering angle.

        Args:
            ranges (List[float]): List of range values from LaserScan.
            last_steering_angle_rad (float): Last steering angle (in radians)
            published to /drive.

        Returns:
            bool: True if the car is bound to collide with an obstacle on one of
            its sides, False if not.
        """
        return self.__sides_too_close(ranges=ranges) and last_steering_angle_rad != 0

    def __moving_straight_state(self) -> None:
        """Wrapper function for all the operations ("side effects") to be
        carried out in the MOVING_FORWARD state. Sets steering angle to 0, speed
        equal to its previous value, and publishes the Ackermann message to
        /drive.
        """
        self.get_logger().warning(f"Going to hit wall apparently! Steering straight!")
        new_steering_angle = 0.0
        # TODO: In case the speed is too high in these scenarios, add an
        # instance variable that stores the last followed gap depth, and
        # compute what the new velocity should be based on that previous
        # depth, but with the distance travelled since that was sent
        # (calculated using current velocity and time between cycles).
        new_speed = self.__last_drive_message.drive.speed
        # Then, use the drive publisher to publish the ackermann steering
        # message with these values.
        # NOTE: Problem is that we set the steering angle zero, but then on the
        # next callback invocation, it doesn't stay in safety mode. Need some
        # sort of counter or something to stay in this state for a certain
        # period of time.

        self.publish_control(new_steering_angle=new_steering_angle, new_speed=new_speed)
        return
    
    def __disparity_control_state(self, ranges: List[float]) -> None:
        """Wrapper function for all the operations/actions to be carried out
        while the car is in the DISPARITY_CONTROL state.
        """

        # Before doing anything else, grab the working range of indices. I.e.,
        # all range values except for those on the side.
        self.get_logger().info("In state disparity control")
        range_indices = self.__middle_index_range

        # 1. (TODO) Clamp all range values to maximum depth value. Choosing not
        #    to implement this first, just to see what performance is like
        #    without it.

        # 2. Next, find all the index pairs in the ranges array where there is a
        #    disparity that exceeds the disparity threshold.
        disparity_threshold_m = self.__get_local_parameter("disparity_threshold_m")
        disparities = gf.find_disparities(ranges=ranges,
                                          range_indices=range_indices,
                                          disparity_threshold_m=disparity_threshold_m)

        # TODO: Call a function here that takes the found disparities and
        # removes/filters out those whose minimum range is ABOVE the gap_depth
        # threshold--as we don't need to worry about padding/extending those
        # disparities just yet. In fact, they really just act like an obstacle
        # we don't even know if we'll need to avoid yet.

        # 3. Then, extend each disparity according to the width of the car and
        #    the depth (range) at which that disparity occurs.
        angle_increment_rad = self.__get_local_parameter("lidar_angle_increment_rad")
        gf.pad_disparities(ranges=ranges, 
                           range_indices=range_indices,
                           disparities=disparities,
                           car_width_m=self.__car_width_m,
                           angle_increment_rad=angle_increment_rad)
        # NOTE: steps 2 and 3 could probably be combined into a single function.
        # However, probably more testable and easier to understand if they're
        # separate for now.
        

        # 4. Once the ranges array has been processed (padding locations where
        #    there is sufficient disparity, in the above case), we can now use a
        #    function to go into the updated ranges and find all the gaps.
        # gap_depth_threshold_m =
        # self.__get_local_parameter("gap_depth_threshold_m")
        # TODO: Temporarily testing dynamic gap depth threshold based on speed.
        gap_depth_threshold_m = self.get_gap_depth_threshold_from_speed(self.__last_drive_message.drive.speed)
        gaps = gf.find_gaps(ranges=ranges,
                            range_indices=range_indices,
                            gap_depth_threshold_m=gap_depth_threshold_m)

        # NOTE: MAKE SURE TO CHECK FOR WHEN FIND_GAPS DOESN'T RETURN ANY
        # GAPS--NEED SOME SORT OF OTHER STATE TO DROP INTO OR SOME RECOURSE
        # PLAN! MAYBE MAKE A BLOCKED STATE OR SOMETHING LIKE THAT?
        # NOTE: Past this point, this condition should really be used as a guard
        # condition which controls whether the state machine goes into one of
        # two states: 1.) disparity_control_with_gaps and 2.)
        # disparity_control_no_gaps. For now, this is fine--but know there's a
        # better way of doing it.
        if len(gaps) == 0:
            self.get_logger().warning(f"Car couldn't find any gaps--stopping for now!")
            # self.get_logger().warning(f"Range subset with no gaps:\n{ranges[range_indices[0]:range_indices[-1]]}")
            self.get_logger().warning(f"Largest gap depth in available range: {max(ranges[range_indices[0]:range_indices[-1]]):.6f} meters.")
            self.publish_control(new_steering_angle=0.0, new_speed=0.0)
            return
        
        # 5. If there is at least one gap returned, call a function to get the
        #    gap with the greatest depth.
        gap, depth = gf.get_max_depth_gap(gaps=gaps, ranges=ranges)

        # 6. Then, call a function that gets the middle of the selected gap.
        #    I.e., returns the index of the middle of that gap.
        gap_middle_index = gf.get_gap_middle_index(gap_left_index=gap.left_index,
                                                   gap_right_index=gap.right_index)

        # 7. Finally, to get the steering angle that corresponds to the middle
        #    of the gape, use the lidar utils function to get the angle that
        #    corresponds to the index in the middle of the gap.
        angle_min_rad = self.__get_local_parameter("lidar_angle_min_rad")
        gap_middle_angle = lu.get_angle_from_index(gap_middle_index, 
                                                   angle_increment_rad=angle_increment_rad,
                                                   angle_min_rad=angle_min_rad)
        # Use the angle to this gap's middle as our new steering angle.
        new_steering_angle = gap_middle_angle

        # 6. Finally, call a separate function that computes velocity as a
        #    function of the depth of the selected gap. Could also use a
        #    simplified version of this function for now that just uses constant
        #    speed, or could also base it on steering angle.
        # new_speed =
        # self.velocity_from_steering_angle(steering_angle=new_steering_angle)
        # new_speed = self.speed_from_gap_max_depth(gap_max_depth=depth,
        # max_speed=2.5, min_speed=0.3)
        new_speed = self.speed_from_gap_max_depth(gap_max_depth=depth, max_speed=4.0, min_speed=0.3)

        # 7. Finally, call function to publish newly computed steering angle and
        #    speed.
        self.publish_control(new_steering_angle=new_steering_angle, new_speed=new_speed)
        return

    def __lidar_callback(self, laser_scan: LaserScan):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an
        AckermannDriveStamped Message.
        """
        ranges: List[float] = laser_scan.ranges

        if self.__current_state == FollowGapState.INIT:
            # Unconditional guard used to transition into DISPARITY_CONTROL
            # state.
            if True:
                self.__current_state = FollowGapState.DISPARITY_CONTROL

        elif self.__current_state == FollowGapState.MOVING_STRAIGHT:
            
            # Apply side effects of being in MOVING_STRAIGHT state. I.e., set
            # steering angle to 0 and make speed unchanged.
            self.__moving_straight_state()

            if self.__moving_straight_timer == 0:
                self.__moving_straight_timer = self.__get_local_parameter("moving_straight_safety_timeout")
            else:
                self.__moving_straight_timer -= 1

            # Guard condition here.
            # If side is still too close, continue moving straight.
            # TODO: just NOTE that I MAY HAVE TO also add a second guard
            # condition here to check whether our steering angle is nonzero.
            # This comes directly from the slides on disparity based control.
            # NOTE: AlSO--if this side collision condition starts causing
            # problems, give it a try without it.
            if self.__moving_straight_timer > 0:
                self.__current_state = FollowGapState.MOVING_STRAIGHT
            elif self.__going_to_hit_wall(ranges=ranges, last_steering_angle_rad=self.__last_drive_message.drive.steering_angle):
                self.__current_state = FollowGapState.MOVING_STRAIGHT
            # Otherwise, switch back to disparity control state.
            else:
                self.__current_state = FollowGapState.DISPARITY_CONTROL

        elif self.__current_state == FollowGapState.DISPARITY_CONTROL:

            # Take the ranges and 
            self.__disparity_control_state(ranges=ranges)
            # TEMPORARY: Publish the modified range array back for visualization.
            self.__publish_processed_ranges(ranges=ranges, original_scan_message=laser_scan)

            # Guard Condition here.
            # If side is still too close, continue moving straight.
            if self.__going_to_hit_wall(ranges=ranges, last_steering_angle_rad=self.__last_drive_message.drive.steering_angle):
                self.__current_state = FollowGapState.MOVING_STRAIGHT
            # Otherwise, remain in disparity control state.
            else:
                self.__current_state = FollowGapState.DISPARITY_CONTROL


def main(args=None):
    rclpy.init(args=args)
    print("GapFollow Initialized")
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()