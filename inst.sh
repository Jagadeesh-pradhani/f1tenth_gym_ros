# add new maps in "maps" folder of f1tenth_gym_ros
# replace the new sim.yaml at "f1tenth_gym_ros/config"

# YasMarina_map
# FOr this map the robot initial positions are
sx: -25.3
sy: 16.0
stheta: 0.0

# levine_obs
# FOr this map the robot initial positions are
sx: 10.0
sy: 2.0
stheta: 1.57

# Uncomment only the present map in sim.yaml

# Change name of the map in sim.yaml
map_path: '/sim_ws/src/f1tenth_gym_ros/maps/levine_obs'



# after every change in the map or sim.yaml
# Build the rerun the simulation
```
cd ~/sim_ws
colcon build
source install/setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```


# Autonomous movement
# Paste the FG.py in a desired location, possibly in home
# Run the script
python3 FG.py



# The script will run the simulation and the robot will move autonomously

