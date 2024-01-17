#!/bin/bash


echo "Building packages..."
colcon build --symlink-install

echo "Sourcing setup files..."
source /simulator_ws/install/setup.bash
source /opt/ros/humble/setup.bash


# Wait indefinitely
python3 ../state.py
