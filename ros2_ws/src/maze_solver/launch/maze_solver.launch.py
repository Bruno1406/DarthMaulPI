from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    maze_nr = LaunchConfiguration('maze_nr')
    cell_length_m = LaunchConfiguration('cell_length_m')
    max_cells_per_drive = LaunchConfiguration('max_cells_per_drive')
    max_commands_to_execute = LaunchConfiguration('max_commands_to_execute')
    execute_motions = LaunchConfiguration('execute_motions')
    start_maze_server = LaunchConfiguration('start_maze_server')

    return LaunchDescription([
        DeclareLaunchArgument('maze_nr', default_value='1'),
        DeclareLaunchArgument('cell_length_m', default_value='0.254'),
        DeclareLaunchArgument('max_cells_per_drive', default_value='2'),
        DeclareLaunchArgument('max_commands_to_execute', default_value='0'),
        DeclareLaunchArgument('execute_motions', default_value='true'),
        DeclareLaunchArgument('start_maze_server', default_value='true'),

        Node(
            package='maze_publisher_node',
            executable='maze_server',
            name='maze_publisher_server',
            output='screen',
            condition=IfCondition(start_maze_server),
        ),

        Node(
            package='maze_solver',
            executable='maze_solver_node',
            name='maze_solver_node',
            output='screen',
            parameters=[{
                'maze_nr': maze_nr,
                'cell_length_m': cell_length_m,
                'max_cells_per_drive': max_cells_per_drive,
                'max_commands_to_execute': max_commands_to_execute,
                'execute_motions': execute_motions,
            }],
        ),
    ])
