from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    maze_nr = LaunchConfiguration('maze_nr')
    cell_length_m = LaunchConfiguration('cell_length_m')
    max_commands_to_execute = LaunchConfiguration('max_commands_to_execute')
    execute_motions = LaunchConfiguration('execute_motions')
    maze_service_timeout_s = LaunchConfiguration('maze_service_timeout_s')
    shutdown_on_fatal_error = LaunchConfiguration('shutdown_on_fatal_error')
    camera_face_down = LaunchConfiguration('camera_face_down')
    camera_vertical_servo_id = LaunchConfiguration('camera_vertical_servo_id')
    camera_down_position = LaunchConfiguration('camera_down_position')

    return LaunchDescription([
        DeclareLaunchArgument('maze_nr', default_value='1'),
        DeclareLaunchArgument('cell_length_m', default_value='0.254'),
        DeclareLaunchArgument('max_commands_to_execute', default_value='0'),
        DeclareLaunchArgument('execute_motions', default_value='false'),
        DeclareLaunchArgument('maze_service_timeout_s', default_value='15.0'),
        DeclareLaunchArgument('shutdown_on_fatal_error', default_value='true'),
        DeclareLaunchArgument('camera_face_down', default_value='true'),
        DeclareLaunchArgument('camera_vertical_servo_id', default_value='1'),
        DeclareLaunchArgument('camera_down_position', default_value='2000'),

        Node(
            package='maze_solver',
            executable='camera_tilt_node',
            name='task1_camera_tilt_node',
            output='screen',
            condition=IfCondition(camera_face_down),
            parameters=[{
                'servo_id': camera_vertical_servo_id,
                'position': camera_down_position,
                'duration': 0.5,
                'publish_for_s': 3.0,
            }],
        ),

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
