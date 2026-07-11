from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable


def generate_launch_description():
    # --- 1. Parameter Configurations ---
    scan_topic = LaunchConfiguration('scan_topic')
    cell_length_m = LaunchConfiguration('cell_length_m')
    start_heading = LaunchConfiguration('start_heading')
    wall_threshold_m = LaunchConfiguration('wall_threshold_m')
    open_threshold_m = LaunchConfiguration('open_threshold_m')
    drive_max_linear_x_mps = LaunchConfiguration('drive_max_linear_x_mps')
    reverse_max_linear_x_mps = LaunchConfiguration('reverse_max_linear_x_mps')
    reverse_backtracking_enabled = LaunchConfiguration('reverse_backtracking_enabled')
    reverse_backtracking_max_consecutive_cells = LaunchConfiguration('reverse_backtracking_max_consecutive_cells')
    reverse_position_tolerance_m = LaunchConfiguration('reverse_position_tolerance_m')
    reverse_heading_tolerance_rad = LaunchConfiguration('reverse_heading_tolerance_rad')
    rotate_max_angular_z_radps = LaunchConfiguration('rotate_max_angular_z_radps')
    max_cells_to_visit = LaunchConfiguration('max_cells_to_visit')
    output_maze_file = LaunchConfiguration('output_maze_file')
    shutdown_on_complete = LaunchConfiguration('shutdown_on_complete')
    direction_priority = LaunchConfiguration('direction_priority')
    race_planner_enabled = LaunchConfiguration('race_planner_enabled')
    race_compact_known_transit = LaunchConfiguration('race_compact_known_transit')
    race_turn_reverse_at_safe_junction = LaunchConfiguration('race_turn_reverse_at_safe_junction')
    race_reverse_into_new_cells = LaunchConfiguration('race_reverse_into_new_cells')
    race_cost_drive_forward = LaunchConfiguration('race_cost_drive_forward')
    race_cost_drive_backward = LaunchConfiguration('race_cost_drive_backward')
    race_cost_turn_90 = LaunchConfiguration('race_cost_turn_90')
    race_cost_turn_180 = LaunchConfiguration('race_cost_turn_180')
    race_cost_unvisited_info_bonus = LaunchConfiguration('race_cost_unvisited_info_bonus')
    race_cost_straight_bonus = LaunchConfiguration('race_cost_straight_bonus')
    race_lidar_lookahead_bonus = LaunchConfiguration('race_lidar_lookahead_bonus')

    submit_maze_to_grader = LaunchConfiguration('submit_maze_to_grader')
    grade_service_name = LaunchConfiguration('grade_service_name')
    grade_maze_nr = LaunchConfiguration('grade_maze_nr')
    grade_service_timeout_s = LaunchConfiguration('grade_service_timeout_s')
    grade_result_required = LaunchConfiguration('grade_result_required')

    camera_face_down = LaunchConfiguration('camera_face_down')
    camera_vertical_servo_id = LaunchConfiguration('camera_vertical_servo_id')
    camera_down_position = LaunchConfiguration('camera_down_position')
    camera_horizontal_servo_id = LaunchConfiguration('camera_horizontal_servo_id')
    camera_left_position = LaunchConfiguration('camera_left_position')

    return LaunchDescription([
        # --- 2. Declare Launch Arguments ---
        DeclareLaunchArgument('scan_topic', default_value='/ldlidar_node/scan'),
        DeclareLaunchArgument('cell_length_m', default_value='0.254'),
        DeclareLaunchArgument('start_heading', default_value='1'),
        DeclareLaunchArgument('wall_threshold_m', default_value='0.18'),
        DeclareLaunchArgument('open_threshold_m', default_value='0.30'),
        DeclareLaunchArgument('drive_max_linear_x_mps', default_value='0.22'),
        DeclareLaunchArgument('reverse_max_linear_x_mps', default_value='0.085'),
        DeclareLaunchArgument('reverse_backtracking_enabled', default_value='true'),
        DeclareLaunchArgument('reverse_backtracking_max_consecutive_cells', default_value='0'),
        DeclareLaunchArgument('reverse_position_tolerance_m', default_value='0.020'),
        DeclareLaunchArgument('reverse_heading_tolerance_rad', default_value='0.070'),
        DeclareLaunchArgument('rotate_max_angular_z_radps', default_value='0.85'),
        DeclareLaunchArgument('max_cells_to_visit', default_value='0'),
        DeclareLaunchArgument('output_maze_file', default_value=''),
        DeclareLaunchArgument('shutdown_on_complete', default_value='true'),
        DeclareLaunchArgument('direction_priority', default_value='left_straight_right_back'),
        DeclareLaunchArgument('race_planner_enabled', default_value='true'),
        DeclareLaunchArgument('race_compact_known_transit', default_value='true'),
        DeclareLaunchArgument('race_turn_reverse_at_safe_junction', default_value='true'),
        DeclareLaunchArgument('race_reverse_into_new_cells', default_value='false'),
        DeclareLaunchArgument('race_cost_drive_forward', default_value='3.4'),
        DeclareLaunchArgument('race_cost_drive_backward', default_value='5.2'),
        DeclareLaunchArgument('race_cost_turn_90', default_value='3.4'),
        DeclareLaunchArgument('race_cost_turn_180', default_value='6.8'),
        DeclareLaunchArgument('race_cost_unvisited_info_bonus', default_value='0.20'),
        DeclareLaunchArgument('race_cost_straight_bonus', default_value='0.25'),
        DeclareLaunchArgument('race_lidar_lookahead_bonus', default_value='0.20'),

        DeclareLaunchArgument('submit_maze_to_grader', default_value='true'),
        DeclareLaunchArgument('grade_service_name', default_value='/grade_maze'),
        DeclareLaunchArgument('grade_maze_nr', default_value='1'),
        DeclareLaunchArgument('grade_service_timeout_s', default_value='10.0'),
        DeclareLaunchArgument('grade_result_required', default_value='false'),

        # 11111
        DeclareLaunchArgument('camera_face_down', default_value='false'),
        DeclareLaunchArgument('camera_vertical_servo_id', default_value='1'),
        DeclareLaunchArgument('camera_down_position', default_value='2000'),
        DeclareLaunchArgument('camera_horizontal_servo_id', default_value='2'),
        DeclareLaunchArgument('camera_left_position', default_value='2000'),

        SetEnvironmentVariable('MACHINE_TYPE', 'MentorPi_Mecanum'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('controller'),
                    'launch',
                    'controller.launch.py',
                ])
            )
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('ldlidar_node'),
                    'launch',
                    'ldlidar.launch.py',
                ])
            )
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('darth_maul_control'),
                    'launch',
                    'control.launch.py',
                ])
            )
        ),

        # --- 3. Include Peripheral Launch Files (from your XML) ---
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([FindPackageShare('peripherals'), 'launch', 'image_pipeline.launch.py'])
            )
        ),
        
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([FindPackageShare('peripherals'), 'launch', 'apriltag.launch.py'])
            )
        ),

        # --- 4. Camera Tilt Node ---
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
                'publish_for_s': 8.0,
            }],
        ),

        # --- 5. Search and Rescue Node (Merged with Explorer parameters) ---
        Node(
            package='search_rescue',
            executable='search_rescue_node',
            name='search_rescue_node',
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
                'reverse_backtracking_max_consecutive_cells': reverse_backtracking_max_consecutive_cells,
                'reverse_position_tolerance_m': reverse_position_tolerance_m,
                'reverse_heading_tolerance_rad': reverse_heading_tolerance_rad,
                'rotate_max_angular_z_radps': rotate_max_angular_z_radps,
                'max_cells_to_visit': max_cells_to_visit,
                'output_maze_file': output_maze_file,
                'shutdown_on_complete': shutdown_on_complete,
                'direction_priority': direction_priority,
                'race_planner_enabled': race_planner_enabled,
                'race_compact_known_transit': race_compact_known_transit,
                'race_turn_reverse_at_safe_junction': race_turn_reverse_at_safe_junction,
                'race_reverse_into_new_cells': race_reverse_into_new_cells,
                'race_cost_drive_forward': race_cost_drive_forward,
                'race_cost_drive_backward': race_cost_drive_backward,
                'race_cost_turn_90': race_cost_turn_90,
                'race_cost_turn_180': race_cost_turn_180,
                'race_cost_unvisited_info_bonus': race_cost_unvisited_info_bonus,
                'race_cost_straight_bonus': race_cost_straight_bonus,
                'race_lidar_lookahead_bonus': race_lidar_lookahead_bonus,

                'submit_maze_to_grader': submit_maze_to_grader,
                'grade_service_name': grade_service_name,
                'grade_maze_nr': grade_maze_nr,
                'grade_service_timeout_s': grade_service_timeout_s,
                'grade_result_required': grade_result_required,
            }],
        ),
    ])