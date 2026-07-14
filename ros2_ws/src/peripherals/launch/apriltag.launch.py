from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    apriltag_parameters = PathJoinSubstitution([
        FindPackageShare('peripherals'),
        'config',
        'apriltag_config.yaml',
    ])

    return LaunchDescription([
        Node(
            package='apriltag_ros',
            executable='apriltag_node',
            name='apriltag_detector',
            output='screen',
            parameters=[
                apriltag_parameters
            ],
            remappings=[
                (
                    '/image_rect',
                    '/ascamera/camera_publisher/'
                    'rgb0/image_rect',
                ),
                (
                    '/camera_info',
                    '/ascamera/camera_publisher/'
                    'rgb0/camera_info',
                ),
                (
                    '/detections',
                    '/apriltag_detections',
                ),
            ],
        ),
    ])
