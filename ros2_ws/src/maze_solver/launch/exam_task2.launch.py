from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    scan_topic = LaunchConfiguration('scan_topic')
    cell_length_m = LaunchConfiguration('cell_length_m')
    start_heading = LaunchConfiguration('start_heading')
    wall_threshold_m = LaunchConfiguration('wall_threshold_m')
    open_threshold_m = LaunchConfiguration('open_threshold_m')
    drive_max_linear_x_mps = LaunchConfiguration('drive_max_linear_x_mps')
    reverse_max_linear_x_mps = LaunchConfiguration('reverse_max_linear_x_mps')
    reverse_backtracking_enabled = LaunchConfiguration('reverse_backtracking_enabled')
    reverse_backtracking_max_consecutive_cells = LaunchConfiguration(
        'reverse_backtracking_max_consecutive_cells'
    )
    reverse_position_tolerance_m = LaunchConfiguration('reverse_position_tolerance_m')
    reverse_heading_tolerance_rad = LaunchConfiguration('reverse_heading_tolerance_rad')
    rotate_max_angular_z_radps = LaunchConfiguration('rotate_max_angular_z_radps')
    max_cells_to_visit = LaunchConfiguration('max_cells_to_visit')
    output_maze_file = LaunchConfiguration('output_maze_file')
    shutdown_on_complete = LaunchConfiguration('shutdown_on_complete')
    direction_priority = LaunchConfiguration('direction_priority')

    camera_face_down = LaunchConfiguration('camera_face_down')
    camera_vertical_servo_id = LaunchConfiguration('camera_vertical_servo_id')
    camera_down_position = LaunchConfiguration('camera_down_position')
    camera_horizontal_servo_id = LaunchConfiguration('camera_horizontal_servo_id')
    camera_left_position = LaunchConfiguration('camera_left_position')

    return LaunchDescription([
        DeclareLaunchArgument('scan_topic', default_value='/ldlidar_node/scan'),
        DeclareLaunchArgument('cell_length_m', default_value='0.254'),
        DeclareLaunchArgument('start_heading', default_value='1'),
        DeclareLaunchArgument('wall_threshold_m', default_value='0.18'),
        DeclareLaunchArgument('open_threshold_m', default_value='0.30'),
        DeclareLaunchArgument('drive_max_linear_x_mps', default_value='0.18'),
        DeclareLaunchArgument('reverse_max_linear_x_mps', default_value='0.075'),
        DeclareLaunchArgument('reverse_backtracking_enabled', default_value='true'),
        DeclareLaunchArgument('reverse_backtracking_max_consecutive_cells', default_value='1'),
        DeclareLaunchArgument('reverse_position_tolerance_m', default_value='0.018'),
        DeclareLaunchArgument('reverse_heading_tolerance_rad', default_value='0.180'),
        DeclareLaunchArgument('rotate_max_angular_z_radps', default_value='0.85'),
        DeclareLaunchArgument('max_cells_to_visit', default_value='0'),
        DeclareLaunchArgument('output_maze_file', default_value=''),
        DeclareLaunchArgument('shutdown_on_complete', default_value='true'),
        DeclareLaunchArgument(
            'direction_priority',
            default_value='left_straight_right_back',
        ),

        DeclareLaunchArgument('camera_face_down', default_value='true'),
        DeclareLaunchArgument('camera_vertical_servo_id', default_value='1'),
        DeclareLaunchArgument('camera_down_position', default_value='2000'),
        DeclareLaunchArgument('camera_horizontal_servo_id', default_value='2'),
        DeclareLaunchArgument('camera_left_position', default_value='2000'),

        Node(
            package='maze_solver',
            executable='camera_tilt_node',
            name='task2_camera_tilt_node',
            output='screen',
            condition=IfCondition(camera_face_down),
            parameters=[{
                'servo_id': camera_vertical_servo_id,
                'position': camera_down_position,
                'horizontal_servo_id': camera_horizontal_servo_id,
                'left_position': camera_left_position,
                'duration': 0.5,
                'publish_for_s': 3.0,
            }],
        ),

        Node(
            package='maze_solver',
            executable='maze_explorer_node',
            name='maze_explorer_node',
            output='screen',
            parameters=[{
                'scan_topic': scan_topic,
                'cell_length_m': cell_length_m,
                'start_heading': start_heading,
                'wall_threshold_m': wall_threshold_m,
                'open_threshold_m': open_threshold_m,
                'drive_max_linear_x_mps': drive_max_linear_x_mps,
                'reverse_max_linear_x_mps': reverse_max_linear_x_mps,
                'reverse_backtracking_enabled': reverse_backtracking_enabled,
                'reverse_backtracking_max_consecutive_cells': (
                    reverse_backtracking_max_consecutive_cells
                ),
                'reverse_position_tolerance_m': reverse_position_tolerance_m,
                'reverse_heading_tolerance_rad': reverse_heading_tolerance_rad,
                'rotate_max_angular_z_radps': rotate_max_angular_z_radps,
                'max_cells_to_visit': max_cells_to_visit,
                'output_maze_file': output_maze_file,
                'shutdown_on_complete': shutdown_on_complete,
                'direction_priority': direction_priority,
            }],
        ),
    ])
