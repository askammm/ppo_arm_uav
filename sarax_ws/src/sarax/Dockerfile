# Pull px4-ros-noetic image
FROM px4io/px4-dev-ros-noetic

# Update the linux environment
RUN echo "Updating environment" \
    && sudo apt install nano -y \
    && sudo apt install figlet -y \
    && sudo apt update -y \
    && sudo apt upgrade -y

# Check environment
RUN echo "Checking environment" \
    && export \
    && ulimit -a

# Perform rosdep
RUN echo "Use rosdep to update dependencies" \
    && . /opt/ros/$ROS_DISTRO/setup.sh \
    && rosdep update

# Create a dependency workspace
RUN mkdir -p ~/dependencies \
    && cd ~/dependencies

# Install PX4 Dependencies
RUN cd ~/dependencies \
    && wget https://raw.githubusercontent.com/PX4/PX4-Autopilot/main/Tools/setup/ubuntu.sh \
    && wget https://raw.githubusercontent.com/PX4/PX4-Autopilot/main/Tools/setup/requirements.txt \
    && chmod +x ~/dependencies/ubuntu.sh && . ~/dependencies/ubuntu.sh

# Install MAVROS using binaries
RUN cd ~/dependencies \
    && sudo apt-get install ros-$ROS_DISTRO-mavros ros-$ROS_DISTRO-mavros-extras -y \
    && wget https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh \
    && chmod +x install_geographiclib_datasets.sh && ./install_geographiclib_datasets.sh

# Install ROS build dependencies
RUN sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key F42ED6FBAB17C654 \
    && sudo apt-get update \
    && sudo apt-get install -y \
        python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential \
        python3-catkin-tools python3-osrf-pycommon \
        ros-$ROS_DISTRO-rqt ros-$ROS_DISTRO-rqt-common-plugins \
        # 补上缺失的控制核心包
        ros-$ROS_DISTRO-controller-manager \
        ros-$ROS_DISTRO-control-toolbox \
        ros-$ROS_DISTRO-transmission-interface \
        ros-$ROS_DISTRO-joint-limits-interface \
        ros-$ROS_DISTRO-gazebo-ros-control \
        ros-$ROS_DISTRO-joint-state-controller \
        ros-$ROS_DISTRO-effort-controllers \
        ros-$ROS_DISTRO-velocity-controllers \
        ros-$ROS_DISTRO-position-controllers \
        # 在 RUN sudo apt-get install ... 列表中添加以下包
	ros-$ROS_DISTRO-socketcan-bridge \
	ros-$ROS_DISTRO-can-msgs \
	ros-$ROS_DISTRO-socketcan-interface\
    && pip install scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
# Create a workspace
RUN echo "Creating sarax workspace" \
    && cd /home \
    && mkdir -p user/sarax_ws/src/sarax
    
RUN git config --global http.proxy http://192.168.122.1:7897 && \
    git config --global https.proxy http://192.168.122.1:7897 && \
    # 核心：限制子模块并发拉取数为 1（虽然慢，但最稳），并增加重试次数
    git config --global submodule.fetchJobs 1 && \
    git config --global http.lowSpeedLimit 1000 && \
    git config --global http.lowSpeedTime 60
# Clone the sarax branch from the PX4-Autopilot repository
RUN cd /home/user/sarax_ws/ \
    && git clone --recursive -b v1.13.2-sarax-sim https://github.com/SaxionMechatronics/PX4-Autopilot.git

# Initialise catkin workspace
RUN cd /home/user/sarax_ws/ \
    && . /opt/ros/$ROS_DISTRO/setup.sh \
    && catkin init \
    && wstool init src

# Copy the source into workspace source directory
COPY . /home/user/sarax_ws/src/sarax/

# Install dependencies using rosdep
RUN cd /home/user/sarax_ws/ \
    && . /opt/ros/$ROS_DISTRO/setup.sh \
    && rosdep install --from-paths src --ignore-src -r -y --skip-keys="python-scipy"

# Build the workspace
RUN cd /home/user/sarax_ws/ \
    && . /opt/ros/$ROS_DISTRO/setup.sh \
    && sudo apt-get update \
    && catkin build

# Export graphic libraries
RUN echo "# If running on WSL2, uncomment the following lines" >> ~/.bashrc \
    && echo "# export LD_LIBRARY_PATH=/usr/lib/wsl/lib" >> ~/.bashrc \
    && echo "# export LIBVA_DRIVER_NAME=d3d12" >> ~/.bashrc

# Export worskpace and update sourcing
RUN cd /home/user/sarax_ws/ \
    && echo "export SARAX_WS=$PWD" >> ~/.bashrc \
    && echo "source \$SARAX_WS/devel/setup.bash" >> ~/.bashrc \
    && echo "source \$SARAX_WS/src/sarax/scripts/container/docker_entrypoint.sh" >> ~/.bashrc

# Build the simulator to speed up running within the container
RUN cd /home/user/sarax_ws/PX4-Autopilot \
    && ./sarax_plus_sitl.bash
