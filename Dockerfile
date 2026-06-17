# Start from the official OSRF ROS 2 Jazzy desktop image
FROM osrf/ros:jazzy-desktop

# Avoid interactive prompts during apt-get installations
ENV DEBIAN_FRONTEND=noninteractive

# Update the package manager and install system-level compilation tools
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install all requested ROS 2 Jazzy packages and Python libraries
RUN apt-get update && apt-get install -y \
    ros-jazzy-usb-cam \
    ros-jazzy-camera-calibration \
    ros-jazzy-image-proc \
    ros-jazzy-apriltag-ros \
    ros-jazzy-apriltag-msgs \
    ros-jazzy-slam-toolbox \
    ros-jazzy-navigation2 \
    ros-jazzy-joint-state-publisher \
    ros-jazzy-xacro \
    ros-jazzy-imu-complementary-filter \
    python3-transforms3d \
    python3-pydantic \
    && rm -rf /var/lib/apt/lists/*

# Set up the exact workspace directory structure requested
# We place it in /root/workspace/ros2_ws to follow standard root containers,
# but your host mapping will bind your local laptop files here.
WORKDIR /root/workspace/ros2_ws/src

# Set the required robot architecture environment variables
ENV MACHINE_TYPE=MentorPi_Mecanum

# Return to the workspace root directory
WORKDIR /root/workspace/ros2_ws

# Initialize the workspace using colcon build
RUN . /opt/ros/jazzy/setup.sh && colcon build

# Inject automatic sourcing commands into the container's root bash profile
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc && \
    echo "source /root/workspace/ros2_ws/install/local_setup.bash" >> /root/.bashrc && \
    echo "export MACHINE_TYPE=MentorPi_Mecanum" >> /root/.bashrc

# Set the default shell command when logging into the container
CMD ["bash"]