from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_name = 'darth_maul_control'
    params_file = PathJoinSubstitution([
        FindPackageShare(package_name),
        'config',
        'control_params.yaml',
    ])

    return LaunchDescription([
        Node(
            package=package_name,
            executable='control_node',
            name='darth_maul_control',
            parameters=[params_file],
            output='screen',
        ),
    ])
