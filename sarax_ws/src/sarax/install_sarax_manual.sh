#!/bin/bash

# SARAX Installation Script for Ubuntu 20.04
# Run with: chmod +x install_sarax_manual.sh && ./install_sarax_manual.sh

set -e

echo "=== Step 1: Installing ROS Noetic ==="
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172D4FFF42CF6桂5EAD07
sudo apt update
sudo apt install ros-noetic-desktop-full -y
sudo rosdep init
rosdep update
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

echo "=== Step 2: Installing PX4 Toolchain ==="
cd ~
wget -q https://raw.githubusercontent.com/PX4/PX4-Autopilot/main/Tools/setup/ubuntu.sh
wget -q https://raw.githubusercontent.com/PX4/PX4-Autopilot/main/Tools/setup/requirements.txt
chmod +x ubuntu.sh
./ubuntu.sh

echo "=== Step 3: Installing MAVROS and GeographicLib ==="
sudo apt-get install ros-noetic-mavros ros-noetic-mavros-extras -y
wget -q https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh
chmod +x install_geometriclib_datasets.sh
sudo ./install_geometriclib_datasets.sh

echo "=== Step 4: Installing QGroundControl Dependencies ==="
sudo usermod -a -G dialout $USER
sudo apt install gstreamer1.0-plugins-bad gstreamer1.0-plugins-good libqt5svg5 libqt5xmlpatterns5 -y
wget -q https://d176tv9ibo4kno.cloudfront.net/downloads/QGroundControl-stable.AppImage
chmod +x QGroundControl-stable.AppImage

echo "=== Step 5: Creating SARAX Workspace ==="
mkdir -p ~/sarax_ws/src && cd ~/sarax_ws
catkin init
wstool init src

echo "Cloning PX4-Autopilot..."
git clone --recursive -b v1.13.2-sarax-sim https://github.com/SaxionMechatronics/PX4-Autopilot.git

echo "Cloning SARAX..."
cd src && git clone https://github.com/SaxionMechatronics/sarax.git

echo "Installing ROS dependencies..."
cd ~/sarax_ws
source /opt/ros/noetic/setup.bash
rosdep install --from-paths src --ignore-src -r -y --skip-keys="python-scipy"

echo "Building SARAX..."
catkin build

echo "=== Setting up environment variables ==="
cd ~/sarax_ws
echo "export SARAX_WS=$PWD" >> ~/.bashrc
echo "source \$SARAX_WS/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc

echo "=== Installation Complete! ==="
echo "To run SARAX simulation:"
echo "1. Open QGroundControl: ./QGroundControl-stable.AppImage"
echo "2. Terminal 1: cd \$SARAX_WS/PX4-Autopilot && ./sarax_plus_sitl.bash"
echo "3. Terminal 2: roslaunch m4e_mani_base sarax_plus_sitl.launch"
