from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    maze_nr = LaunchConfiguration('maze_nr')
    cell_length_m = LaunchConfiguration('cell_length_m')
    max_commands_to_execute = LaunchConfiguration('max_commands_to_execute')
    execute_motions = LaunchConfiguration('execute_motions')
    maze_service_timeout_s = LaunchConfiguration('maze_service_timeout_s')
    shutdown_on_fatal_error = LaunchConfiguration('shutdown_on_fatal_error')

    return LaunchDescription([
        DeclareLaunchArgument('maze_nr', default_value='1'),
        DeclareLaunchArgument('cell_length_m', default_value='0.254'),
        DeclareLaunchArgument('max_commands_to_execute', default_value='0'),
        DeclareLaunchArgument('execute_motions', default_value='false'),
        DeclareLaunchArgument('maze_service_timeout_s', default_value='15.0'),
        DeclareLaunchArgument('shutdown_on_fatal_error', default_value='true'),

        Node(
            package='maze_solver',
            executable='maze_solver_node',
            name='maze_solver_node',
            output='screen',
            parameters=[{
                'maze_nr': maze_nr,
                'cell_length_m': cell_length_m,
                'max_commands_to_execute': max_commands_to_execute,
                'execute_motions': execute_motions,
                'maze_service_name': '/get_ros_maze',
                'maze_service_timeout_s': maze_service_timeout_s,
                'shutdown_on_fatal_error': shutdown_on_fatal_error,
            }],
        ),
    ])
