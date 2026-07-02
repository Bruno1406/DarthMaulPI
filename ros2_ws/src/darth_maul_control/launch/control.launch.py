from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_name = 'darth_maul_control'
    params_file = PathJoinSubstitution([
        FindPackageShare(package_name),
        'config',
        'control_params.yaml',
    ])

    grid_alignment_control_enabled = LaunchConfiguration(
        'grid_alignment_control_enabled'
    )
    grid_yaw_correction_enabled = LaunchConfiguration(
        'grid_yaw_correction_enabled'
    )
    k_grid_yaw = LaunchConfiguration('k_grid_yaw')
    max_grid_yaw_correction_radps = LaunchConfiguration(
        'max_grid_yaw_correction_radps'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'grid_alignment_control_enabled',
            default_value='false',
            description=(
                'Enable LiDAR grid-alignment control terms. '
                'When false, grid alignment remains diagnostic only.'
            ),
        ),
        DeclareLaunchArgument(
            'grid_yaw_correction_enabled',
            default_value='false',
            description=(
                'Enable low-gain LiDAR grid-yaw correction during forward translation.'
            ),
        ),
        DeclareLaunchArgument(
            'k_grid_yaw',
            default_value='0.70',
            description='Grid yaw correction proportional gain.',
        ),
        DeclareLaunchArgument(
            'max_grid_yaw_correction_radps',
            default_value='0.045',
            description='Absolute cap for grid yaw correction in rad/s.',
        ),
        Node(
            package=package_name,
            executable='control_node',
            name='darth_maul_control',
            parameters=[
                params_file,
                {
                    'grid_alignment_control_enabled': ParameterValue(
                        grid_alignment_control_enabled,
                        value_type=bool,
                    ),
                    'grid_yaw_correction_enabled': ParameterValue(
                        grid_yaw_correction_enabled,
                        value_type=bool,
                    ),
                    'k_grid_yaw': ParameterValue(
                        k_grid_yaw,
                        value_type=float,
                    ),
                    'max_grid_yaw_correction_radps': ParameterValue(
                        max_grid_yaw_correction_radps,
                        value_type=float,
                    ),
                },
            ],
            output='screen',
        ),
    ])
