#!/bin/bash


echo "Building packages..."
colcon build --symlink-install

echo "Sourcing setup files..."
source /simulator_ws/install/setup.bash
source /opt/ros/humble/setup.bash


# Wait indefinitely
# python3 ../main.py
echo "Script is now waiting indefinitely..."
sleep infinity
