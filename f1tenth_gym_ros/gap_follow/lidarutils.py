"""Python utility module containing common lidar utility functions.
"""
import math
from typing import List
import numpy as np

def angle_in_range(angle_rad: float, angle_min_rad: float, angle_max_rad: float) -> bool:
    """Returns whether or not the provided angle is within the minimum and
    maximum angle of the the given laser_scan message.

    Args:
        angle_rad (float): The angle to check in radians.
        angle_min_rad (float): The minimum (smallest) angle measured by the
        LiDAR from the scan at hand.
        angle_max_rad (float): The maximum (most positive) angle measured by the
        LiDAR from the scan at hand.

    Returns:
        bool: True if angle_rad is within the possible angles, False if not. 
    """
    if angle_rad > angle_max_rad or angle_rad < angle_min_rad:
        return False
    return True

def get_index_from_angle(angle_rad: float, 
                         angle_increment_rad: float, 
                         angle_min_rad: float,
                         angle_max_rad: float,
                         num_ranges: int) -> int:
    """Returns index in ranges array that corresponds to an angle measured from
    the x-axis of the LiDAR frame.

    Args:
        angle_rad (float): Angle (in radians) as measured from the x-axis of the
        LiDAR frame.
        angle_increment_rad (float): The angle between each LiDAR range in a
        scan.
        angle_min_rad (float): The minimum (most negative, lowest) angle
        a range measurement is taken by the LiDAR.
        angle_max_rad (float): The maximum (most positive, highest) angle
        a range measurement is taken by the LiDAR.
        num_ranges (int): The number of entries in a scan message's ranges
        array.

    Raises:
        Exception: Throws an exception if the provided angle is outside of range
        of angles measured by the LiDAR.

    Returns:
        int: The approximate index in the LaserScan ranges array that
        corresponds to the provided angle.
    """
    if not angle_in_range(angle_rad=angle_rad, angle_min_rad=angle_min_rad, angle_max_rad=angle_max_rad):
        raise Exception(f"Provided angle {angle_rad:.4f} ({math.degrees(angle_rad):.4f}) is outside of the range within the provided laserscan message: [{angle_min_rad:.4f} : {angle_max_rad:.4f}]")
    adjusted_angle_rad = angle_rad + -angle_min_rad
    unclipped_index = math.floor(adjusted_angle_rad/angle_increment_rad)
    min_ranges_index = 0
    max_ranges_index = num_ranges - 1
    clipped_index = int(np.clip(unclipped_index, a_min=min_ranges_index, a_max=max_ranges_index))
    return clipped_index

def get_index_range_from_angles(start_angle_rad: float,
                                end_angle_rad: float,
                                angle_min_rad: float, 
                                angle_max_rad: float,
                                angle_increment_rad: float, 
                                num_ranges: int) -> List[int]:
    """Returns a list of indices from ranges array that span the range of the
    angles provided. NOTE that it would make the most sense for the provided
    start_angle to the be the lesser or more-negative angle, where the end_angle
    is always greater than the start_angle. This way, the indices returned are
    can be traversed in order in the corresponding array.

    Args:
        start_angle_rad (float): The first angle (in radians) you want the index
        of.
        end_angle_rad (float): The second angle (in radians) you want the index
        of. Should be greater than the start angle.
        angle_min_rad (float): The minimum (smallest) angle measured by the
        LiDAR from the scan at hand.
        angle_max_rad (float): The maximum (most positive) angle measured by the
        LiDAR from the scan at hand.
        angle_increment_rad (float): The increment (in radians) between each
        range within the LiDAR scan's ranges array.
        num_ranges (int): The length of the ranges array from the LaserScan
        message. I.e., the number of range measurements.

    Returns:
        List[int]: List of indices from ranges array that span the range of the
        angles provided.
    """
    start_index = get_index_from_angle(angle_rad=start_angle_rad,
                                       angle_min_rad=angle_min_rad,
                                       angle_max_rad=angle_max_rad,
                                       angle_increment_rad=angle_increment_rad,
                                       num_ranges=num_ranges)
    end_index = get_index_from_angle(angle_rad=end_angle_rad,
                                     angle_min_rad=angle_min_rad,
                                     angle_max_rad=angle_max_rad,
                                     angle_increment_rad=angle_increment_rad,
                                     num_ranges=num_ranges)
    return list(range(start_index, end_index+1))

def get_angle_from_index(index: int, 
                         angle_increment_rad: float, 
                         angle_min_rad: float) -> float:
    """Returns the angle that corresponds to the provided index in an array of
    ranges from a LaserScan message (where zero degrees is not at index 0).

    Args:
        index (int): The index of the range value to get the angle of.
        angle_increment_rad (float): The angle increment (in radians) from one
        range value to the next.

    Returns:
        float: The angle (in radians) that approximately corresponds with the
        provided index.
    """
    return float(index*angle_increment_rad + angle_min_rad)

def clip_range_values(ranges: List[float],
                      min: float,
                      max: float) -> None:
    """clips each value in the provided ranges array between the provided
    minimum and maximum. Note that this is an IN-PLACE operation. That is, the
    ranges list provided will be modified.

    Args:
        ranges (List[float]): List of range values from LaserScan.
        min (float): Minimum value that ranges below that value will get set to.
        max (float): Maximum value that ranges above that value will get set to.
    """
    for i in range(len(ranges)):
        if ranges[i] < min:
            ranges[i] = min
        elif ranges[i] > max:
            ranges[i] = max
    return