import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np
import math
import threading
import time
from functools import lru_cache


class PIDController:
    def __init__(self, kp, ki, kd, node):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.node = node
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = self.node.get_clock().now()
        self._lock = threading.Lock()

    def update(self, error):
        with self._lock:
            current_time = self.node.get_clock().now()
            dt = (current_time - self.last_time).nanoseconds / 1e9
            if dt <= 0.0:
                dt = 1e-3
            self.integral += error * dt
            # Anti-windup
            self.integral = np.clip(self.integral, -10.0, 10.0)
            derivative = (error - self.prev_error) / dt
            output = self.kp * error + self.ki * self.integral + self.kd * derivative
            self.prev_error = error
            self.last_time = current_time
            return output


class ReactiveGapFollower(Node):
    def __init__(self):
        super().__init__('reactive_gap_follower')
        self.last_time_ = time.time()
        self.integral_ = 0.0
        self.prev_error_ = 0.0
        
        # Create separate callback groups for concurrency
        self.lidar_callback_group = MutuallyExclusiveCallbackGroup()
        self.odom_callback_group = MutuallyExclusiveCallbackGroup()
        self.control_callback_group = ReentrantCallbackGroup()
        
        # Publishers with QoS profiles for better performance
        self.publisher_ = self.create_publisher(
            AckermannDriveStamped, 
            '/drive', 
            10
        )
        
        # Subscriptions with callback groups for parallel processing
        self.lidar_subscription = self.create_subscription(
            LaserScan, 
            '/scan', 
            self.lidar_callback, 
            10,
            callback_group=self.lidar_callback_group
        )
        
        self.odom_subscription = self.create_subscription(
            Odometry, 
            '/odom', 
            self.odom_callback, 
            10,
            callback_group=self.odom_callback_group
        )
        
        # Thread-safe variables
        self._lock = threading.Lock()
        self.angle = 0.0
        self.ranges = np.array([])
        self.vx = 0.0  # Forward velocity
        self.last_scan_time = self.get_clock().now()
        self.last_control_time = self.get_clock().now()
        
        # Precompute indices for the front 270° (exclude 20° margins)
        self.total_beams = 1080
        self.min_indx = int(20 * (self.total_beams / 270))
        self.max_indx = int(self.total_beams - 20 * (self.total_beams / 270))
        
        # Safety and tuning parameters (tunable)
        self.min_ittc_threshold = 0.5    # Min acceptable instantaneous time-to-collision
        self.min_gap_threshold = 0.5     # Minimum clearance (meters)
        
        # Buffer zone parameters for preventing wall sticking
        self.buffer_zone_radius = 1.2    # Radius of buffer zone around robot (meters)
        self.buffer_width_front = 0.8    # Width of buffer zone in front (meters)
        self.buffer_width_sides = 0.6    # Width of buffer zone on sides (meters)
        self.turn_anticipation_factor = 1.5  # Higher values make robot anticipate turns earlier
        self.side_clearance_weight = 1.2     # Weighting for side clearance (higher values prefer wider side gaps)
        
        # Wall detection
        self.wall_detection_threshold = 1.0  # Distance threshold for detecting walls (meters)
        self.wall_angle_threshold = 30       # Angle threshold for wall detection (degrees)
        self.consecutive_wall_readings = 0   # Counter for consecutive wall detections
        self.max_wall_readings = 3           # Number of readings to confirm wall
        self.wall_detected = False           # Wall detection flag

        # Speed tuning parameters (faster speeds)
        self.speed_straight = 2.0   # Target speed for near-zero steering angle
        self.speed_moderate = 1.7   # Target speed for moderate steering angles
        self.speed_sharp = 1.0      # Target speed for large steering angles
        self.recovery_speed = 1.0   # Recovery maneuver speed
        self.wall_speed = 1.0       # Speed when near walls

        # PID Controller for smoothing speed commands
        pid_kp = 0.5
        pid_ki = 0.1
        pid_kd = 0.05
        self.speed_pid = PIDController(pid_kp, pid_ki, pid_kd, self)
        
        # Create a timer for periodic control updates (separate from sensor callbacks)
        self.control_timer = self.create_timer(
            0.05,  # 20Hz control loop
            self.control_loop,
            callback_group=self.control_callback_group
        )
        
        # Shared state for communication between threads
        self.latest_ranges = None
        self.latest_angle_min = 0.0
        self.latest_angle_increment = 0.0
        self.need_recovery = False
        self.safe_stop = False
        
        # Cache for beam angle calculations (huge performance boost)
        self.beam_angles = None
        
        # State tracking for dynamic behavior
        self.wall_follow_mode = False
        self.wall_follow_direction = 0  # 0=none, -1=left, 1=right
        self.wall_follow_timer = 0
        self.turn_in_progress = False
        self.turn_direction = 0  # 0=none, -1=left, 1=right
        
        # Buffer for smoothing steering commands
        self.steering_history = []
        self.max_steering_history = 3

    @lru_cache(maxsize=1024)
    def _calculate_beam_angle(self, index, angle_min, angle_increment):
        """Calculate beam angle with caching for performance"""
        return angle_min + index * angle_increment

    def odom_callback(self, msg):
        with self._lock:
            self.vx = msg.twist.twist.linear.x

    def compute_beam_angles(self, angle_min, angle_increment):
        """Precompute all beam angles for the scan - major performance improvement"""
        if self.beam_angles is None:
            self.beam_angles = np.array([angle_min + i * angle_increment for i in range(self.total_beams)])
        return self.beam_angles

    def detect_walls(self, scan_ranges, beam_angles):
        """Detect walls and sharp turns using slope analysis"""
        wall_detected = False
        wall_direction = 0  # 0=none, -1=left, 1=right
        
        # Create segments for left, center, and right sections
        segments = {
            'left': (0, len(scan_ranges) // 3),
            'center': (len(scan_ranges) // 3, 2 * len(scan_ranges) // 3),
            'right': (2 * len(scan_ranges) // 3, len(scan_ranges))
        }
        
        segment_distances = {}
        for segment_name, (start, end) in segments.items():
            valid_ranges = scan_ranges[start:end]
            valid_ranges = valid_ranges[valid_ranges > 0]
            segment_distances[segment_name] = np.mean(valid_ranges) if len(valid_ranges) > 0 else float('inf')
        
        # Check for front obstacles (potential walls)
        center_dist = segment_distances['center']
        if center_dist < self.wall_detection_threshold:
            wall_detected = True
            
            # Determine which side has more space
            if segment_distances['left'] > segment_distances['right']:
                wall_direction = -1  # Turn left
            else:
                wall_direction = 1   # Turn right
                
            self.get_logger().info(f"Wall detected! Left: {segment_distances['left']:.2f}m, "
                                  f"Center: {center_dist:.2f}m, Right: {segment_distances['right']:.2f}m, "
                                  f"Direction: {'Left' if wall_direction == -1 else 'Right'}")
        
        return wall_detected, wall_direction

    def create_buffer_zone(self, scan_ranges, beam_angles):
        """Create a buffer zone around robot to prevent getting too close to walls"""
        # Apply distance-based buffer - farther is better with non-linear weighting
        # We'll modify the ranges to penalize getting too close to obstacles
        
        # Calculate buffer radius that varies by angle
        # Wider at the front, narrower at the sides
        theta = np.abs(beam_angles)
        buffer_radius = np.ones_like(beam_angles)
        
        # Front sector (narrower angles) gets larger buffer
        front_sector = theta < np.radians(30)
        buffer_radius[front_sector] = self.buffer_width_front
        
        # Side sectors get medium buffer
        side_sector = (theta >= np.radians(30)) & (theta < np.radians(90))
        buffer_radius[side_sector] = self.buffer_width_sides
        
        # Rear sectors get smaller buffer
        buffer_radius[~(front_sector | side_sector)] = self.buffer_width_sides * 0.5
        
        # Apply buffer by reducing range readings
        buffered_ranges = scan_ranges - buffer_radius
        
        # Make sure we don't have negative ranges
        buffered_ranges = np.maximum(buffered_ranges, 0.0)
        
        return buffered_ranges

    def process_lidar_data(self, ranges, angle_min, angle_increment):
        """Process lidar data in a separate thread"""
        # Process front 270° (exclude 20° margins)
        scan_ranges = np.array(ranges[self.min_indx:self.max_indx])
        # Efficient numpy operations instead of slow loop
        scan_ranges = np.where(np.isnan(scan_ranges) | np.isinf(scan_ranges), 0.0, scan_ranges)
        
        # Check if we have enough clear space
        processed_full = np.where(np.isnan(ranges) | np.isinf(ranges), 100.0, ranges)
        is_safe = np.min(processed_full) >= self.min_gap_threshold
        
        if not is_safe:
            self.get_logger().warn("Minimum gap threshold breached! Initiating recovery behavior.")
            with self._lock:
                self.need_recovery = True
                self.full_ranges = processed_full
            return
        
        # Compute beam angles for the scan subset
        beam_angles = self.beam_angles[self.min_indx:self.max_indx]
        
        # Detect walls and sharp turns
        wall_detected, wall_direction = self.detect_walls(scan_ranges, beam_angles)
        
        # Update wall detection state with hysteresis
        if wall_detected:
            self.consecutive_wall_readings += 1
        else:
            self.consecutive_wall_readings = max(0, self.consecutive_wall_readings - 1)
            
        if self.consecutive_wall_readings >= self.max_wall_readings:
            self.wall_detected = True
            self.wall_follow_direction = wall_direction
            self.wall_follow_timer = 10  # Number of cycles to maintain wall following
        elif self.wall_follow_timer > 0:
            self.wall_follow_timer -= 1
        else:
            self.wall_detected = False
            self.wall_follow_direction = 0
        
        # Apply buffer zone to scan ranges
        buffered_ranges = self.create_buffer_zone(scan_ranges, beam_angles)
        
        # Vectorized operations for finding closest obstacle with smoothing window
        # Use numpy's rolling window approach for speed
        window_size = 5
        # Use strided array for fast rolling window (much faster than loop)
        from numpy.lib.stride_tricks import sliding_window_view
        if len(buffered_ranges) >= window_size:
            windows = sliding_window_view(buffered_ranges, window_size)
            window_sums = np.sum(windows, axis=1)
            valid_windows = window_sums > 0.0
            if np.any(valid_windows):
                closest_window_idx = np.argmin(window_sums[valid_windows])
                closest_idx = np.where(valid_windows)[0][closest_window_idx] + window_size // 2
            else:
                closest_idx = len(buffered_ranges) // 2
        else:
            closest_idx = len(buffered_ranges) // 2

        # Create a safety bubble around the closest obstacle
        bubble_radius = 150
        start_bubble = max(0, closest_idx - bubble_radius)
        end_bubble = min(len(buffered_ranges), closest_idx + bubble_radius + 1)
        
        # Create a copy to avoid modifying original data
        modified_ranges = buffered_ranges.copy()
        modified_ranges[start_bubble:end_bubble] = 0.0
        
        # If in wall follow mode, artificially boost the ranges in the preferred direction
        if self.wall_detected:
            if self.wall_follow_direction == -1:  # Turn left
                # Boost left ranges
                left_idx = 0
                mid_idx = len(modified_ranges) // 2
                boost_factor = 1.5  # Adjustable boost factor
                modified_ranges[left_idx:mid_idx] = np.minimum(
                    modified_ranges[left_idx:mid_idx] * boost_factor,
                    10.0  # Cap at reasonable maximum
                )
            elif self.wall_follow_direction == 1:  # Turn right
                # Boost right ranges
                mid_idx = len(modified_ranges) // 2
                right_idx = len(modified_ranges)
                boost_factor = 1.5  # Adjustable boost factor
                modified_ranges[mid_idx:right_idx] = np.minimum(
                    modified_ranges[mid_idx:right_idx] * boost_factor,
                    10.0  # Cap at reasonable maximum
                )

        # Identify gaps using vectorized operations
        # First create a binary array where 1 = gap, 0 = obstacle
        gap_binary = modified_ranges > 0.0
        
        # Find transitions (0->1 or 1->0) using numpy diff
        transitions = np.diff(np.concatenate(([0], gap_binary, [0])))
        gap_starts = np.where(transitions == 1)[0]
        gap_ends = np.where(transitions == -1)[0] - 1
        
        if len(gap_starts) == 0 or len(gap_ends) == 0:
            with self._lock:
                self.safe_stop = True
            return
            
        # Calculate gap lengths
        gap_lengths = gap_ends - gap_starts + 1
        
        # Calculate gap quality scores - combine length with clearance and position
        gap_scores = np.zeros_like(gap_lengths, dtype=float)
        for i in range(len(gap_lengths)):
            # Get the range values in this gap
            gap_ranges = modified_ranges[gap_starts[i]:gap_ends[i]+1]
            # Average clearance in gap
            avg_clearance = np.mean(gap_ranges)
            # Gap length score
            length_score = gap_lengths[i] / max(1, np.max(gap_lengths))
            # Gap position score - prefer gaps in front of robot
            gap_center_idx = (gap_starts[i] + gap_ends[i]) / 2
            gap_center_angle = beam_angles[int(gap_center_idx)]
            position_score = math.cos(gap_center_angle)  # 1.0 for straight ahead, lower for sides
            
            # Special weighting if we're in wall follow mode
            if self.wall_detected:
                # If following right wall, prefer gaps to the left and vice versa
                if self.wall_follow_direction == 1 and gap_center_angle < 0:
                    position_score *= 1.5  # Boost left gaps when right wall detected
                elif self.wall_follow_direction == -1 and gap_center_angle > 0:
                    position_score *= 1.5  # Boost right gaps when left wall detected
            
            # Combine scores - weightings can be tuned
            gap_scores[i] = 0.4 * length_score + 0.4 * position_score + 0.2 * min(1.0, avg_clearance)
        
        # Find the best gap based on our scoring
        best_gap_idx = np.argmax(gap_scores)
        best_start = gap_starts[best_gap_idx]
        best_end = gap_ends[best_gap_idx]
        
        # Identify the best path through the gap
        gap_ranges = modified_ranges[best_start:best_end+1]
        gap_angles = beam_angles[best_start:best_end+1]
        
        # Find the widest point in the gap
        if len(gap_ranges) > 0:
            # Weight by both range and centrality in the gap
            centrality = 1.0 - 2.0 * np.abs(np.arange(len(gap_ranges)) - len(gap_ranges)/2) / len(gap_ranges)
            range_weight = gap_ranges / np.max(gap_ranges) if np.max(gap_ranges) > 0 else np.ones_like(gap_ranges)
            combined_weight = centrality * range_weight
            
            # If in wall follow mode, adjust weighting
            if self.wall_detected:
                # Bias towards one side based on wall direction
                side_bias = np.linspace(-1, 1, len(gap_ranges))
                if self.wall_follow_direction == -1:  # Left wall, bias right
                    side_weight = 0.5 + 0.5 * side_bias
                elif self.wall_follow_direction == 1:  # Right wall, bias left
                    side_weight = 0.5 - 0.5 * side_bias
                else:
                    side_weight = np.ones_like(side_bias)
                combined_weight *= side_weight
                
            best_in_gap_index = np.argmax(combined_weight)
        else:
            best_in_gap_index = 0
            
        # Calculate the best angle
        best_index = best_start + best_in_gap_index
        best_global_index = best_index + self.min_indx
        best_angle = angle_min + best_global_index * angle_increment
        
        # Smooth steering by averaging with previous commands
        self.steering_history.append(best_angle)
        if len(self.steering_history) > self.max_steering_history:
            self.steering_history.pop(0)
        
        smoothed_angle = np.mean(self.steering_history)
        
        # Update shared state with thread safety
        with self._lock:
            self.angle = smoothed_angle
            self.safe_stop = False
            self.need_recovery = False

    def lidar_callback(self, lidar_info):
        # Update scan time to track freshness
        self.last_scan_time = self.get_clock().now()
        
        # Store scan parameters
        with self._lock:
            self.latest_angle_min = lidar_info.angle_min
            self.latest_angle_increment = lidar_info.angle_increment
            # Make a copy of the ranges to avoid data races
            self.latest_ranges = np.array(lidar_info.ranges)
        
        # Precompute beam angles for the entire scan
        self.compute_beam_angles(lidar_info.angle_min, lidar_info.angle_increment)
        
        # Process the lidar data in a separate thread for performance
        processing_thread = threading.Thread(
            target=self.process_lidar_data,
            args=(self.latest_ranges, lidar_info.angle_min, lidar_info.angle_increment)
        )
        processing_thread.daemon = True
        processing_thread.start()

    def recover_gap(self):
        with self._lock:
            processed = self.full_ranges
        
        num_points = len(processed)
        mid_index = num_points // 2
        left_gap = np.mean(processed[:mid_index])
        right_gap = np.mean(processed[mid_index:])
        self.get_logger().info(f"Left gap: {left_gap:.2f}, Right gap: {right_gap:.2f}")
        
        # Choose direction with more clearance, but with stronger bias
        # This helps commit to one direction rather than oscillating
        steering_angle = -0.5 if left_gap > right_gap else 0.5  # More aggressive steering in recovery
        target_speed = self.recovery_speed * 0.8  # Slower in recovery for safety
        speed_command = self._compute_speed_command(target_speed)
        steering_angle = self.compute_pid_steering(steering_angle)
        
        ackermann_drive_result = AckermannDriveStamped()
        ackermann_drive_result.drive.steering_angle = steering_angle
        ackermann_drive_result.drive.speed = speed_command
        self.publisher_.publish(ackermann_drive_result)
        self.get_logger().info(f"Recovery mode: steering {steering_angle:.2f} with speed {speed_command:.2f}")

    def compute_pid_steering(self, target_angle):
        """PID controller for steering smoothing."""
        current_time = time.time()
        dt = current_time - self.last_time_
        if dt == 0:
            dt = 1e-6

        error = target_angle  # we treat 0 as desired heading

        self.integral_ += error * dt
        derivative = (error - self.prev_error_) / dt

        control = 0.5 * error + 0.0 * self.integral_ + 0.1 * derivative

        self.prev_error_ = error
        self.last_time_ = current_time

        return control

    def _compute_speed_command(self, target_speed):
        with self._lock:
            error = target_speed - self.vx
        pid_adjustment = self.speed_pid.update(error)
        speed_command = target_speed + pid_adjustment
        return max(0.0, speed_command)

    def control_loop(self):
        """Main control loop running at fixed frequency"""
        # Check if we have recent data (timeout detection)
        current_time = self.get_clock().now()
        scan_age = (current_time - self.last_scan_time).nanoseconds / 1e9
        
        if scan_age > 0.5:  # If data is older than 0.5s, emergency stop
            ackermann_drive_result = AckermannDriveStamped()
            ackermann_drive_result.drive.steering_angle = 0.0
            ackermann_drive_result.drive.speed = 0.0
            self.publisher_.publish(ackermann_drive_result)
            self.get_logger().warn(f"Emergency stop: stale sensor data ({scan_age:.2f}s old)")
            return
        
        # Access shared state safely
        with self._lock:
            need_recovery = self.need_recovery
            safe_stop = self.safe_stop
            current_angle = self.angle
            wall_detected = self.wall_detected
        
        if need_recovery:
            self.recover_gap()
            return
        
        ackermann_drive_result = AckermannDriveStamped()
        
        if safe_stop:
            ackermann_drive_result.drive.steering_angle = 0.0
            ackermann_drive_result.drive.speed = 0.0
            self.get_logger().info("Safety stop: insufficient clear gap detected.")
        else:
            ackermann_drive_result.drive.steering_angle = current_angle
            abs_angle = abs(current_angle)
            
            # Determine target speed based on steering angle and wall detection
            if wall_detected:
                # Slower when wall detected
                target_speed = self.wall_speed
            elif abs_angle <= 0.17:
                target_speed = self.speed_straight
            elif abs_angle <= 0.35:
                target_speed = self.speed_moderate
            else:
                target_speed = self.speed_sharp

            speed_command = self._compute_speed_command(target_speed)
            ackermann_drive_result.drive.speed = speed_command
            
            # Log less frequently to reduce overhead
            if (current_time - self.last_control_time).nanoseconds / 1e9 > 0.5:
                wall_status = "Wall detected!" if wall_detected else "No wall"
                self.get_logger().info(f"{wall_status} Steering: {current_angle:.3f}, Speed: {speed_command:.2f}")
                self.last_control_time = current_time
                
        self.publisher_.publish(ackermann_drive_result)


def main(args=None):
    rclpy.init(args=args)
    node = ReactiveGapFollower()
    
    # Use a multithreaded executor for parallel processing
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()