from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import (
    PythonLaunchDescriptionSource,
)
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    launch_camera_pipeline = LaunchConfiguration(
        'launch_camera_pipeline'
    )
    launch_apriltag = LaunchConfiguration(
        'launch_apriltag'
    )

    scan_topic = LaunchConfiguration(
        'scan_topic'
    )
    motion_action_name = LaunchConfiguration(
        'motion_action_name'
    )
    cell_length_m = LaunchConfiguration(
        'cell_length_m'
    )
    start_heading = LaunchConfiguration(
        'start_heading'
    )
    wall_threshold_m = LaunchConfiguration(
        'wall_threshold_m'
    )
    open_threshold_m = LaunchConfiguration(
        'open_threshold_m'
    )
    drive_max_linear_x_mps = LaunchConfiguration(
        'drive_max_linear_x_mps'
    )
    reverse_max_linear_x_mps = LaunchConfiguration(
        'reverse_max_linear_x_mps'
    )
    reverse_backtracking_enabled = LaunchConfiguration(
        'reverse_backtracking_enabled'
    )
    reverse_backtracking_max_consecutive_cells = (
        LaunchConfiguration(
            'reverse_backtracking_max_consecutive_cells'
        )
    )
    reverse_position_tolerance_m = LaunchConfiguration(
        'reverse_position_tolerance_m'
    )
    reverse_heading_tolerance_rad = LaunchConfiguration(
        'reverse_heading_tolerance_rad'
    )
    rotate_max_angular_z_radps = LaunchConfiguration(
        'rotate_max_angular_z_radps'
    )
    rotate_heading_tolerance_rad = LaunchConfiguration(
        'rotate_heading_tolerance_rad'
    )
    max_cells_to_visit = LaunchConfiguration(
        'max_cells_to_visit'
    )
    output_maze_file = LaunchConfiguration(
        'output_maze_file'
    )
    shutdown_on_complete = LaunchConfiguration(
        'shutdown_on_complete'
    )
    direction_priority = LaunchConfiguration(
        'direction_priority'
    )
    race_planner_enabled = LaunchConfiguration(
        'race_planner_enabled'
    )
    race_compact_known_transit = LaunchConfiguration(
        'race_compact_known_transit'
    )
    race_turn_reverse_at_safe_junction = (
        LaunchConfiguration(
            'race_turn_reverse_at_safe_junction'
        )
    )
    race_reverse_into_new_cells = LaunchConfiguration(
        'race_reverse_into_new_cells'
    )
    race_cost_drive_forward = LaunchConfiguration(
        'race_cost_drive_forward'
    )
    race_cost_drive_backward = LaunchConfiguration(
        'race_cost_drive_backward'
    )
    race_cost_turn_90 = LaunchConfiguration(
        'race_cost_turn_90'
    )
    race_cost_turn_180 = LaunchConfiguration(
        'race_cost_turn_180'
    )
    race_cost_unvisited_info_bonus = LaunchConfiguration(
        'race_cost_unvisited_info_bonus'
    )
    race_cost_straight_bonus = LaunchConfiguration(
        'race_cost_straight_bonus'
    )
    race_lidar_lookahead_bonus = LaunchConfiguration(
        'race_lidar_lookahead_bonus'
    )

    rect_image_topic = LaunchConfiguration(
        'rect_image_topic'
    )
    raw_image_topic = LaunchConfiguration(
        'raw_image_topic'
    )
    camera_info_topic = LaunchConfiguration(
        'camera_info_topic'
    )
    apriltag_topic = LaunchConfiguration(
        'apriltag_topic'
    )
    tag_size_m = LaunchConfiguration(
        'tag_size_m'
    )
    tag_min_side_px = LaunchConfiguration(
        'tag_min_side_px'
    )
    tag_min_aspect_ratio = LaunchConfiguration(
        'tag_min_aspect_ratio'
    )
    tag_max_aspect_ratio = LaunchConfiguration(
        'tag_max_aspect_ratio'
    )
    cube_crop_scale = LaunchConfiguration(
        'cube_crop_scale'
    )
    camera_data_timeout_s = LaunchConfiguration(
        'camera_data_timeout_s'
    )
    distance_scale = LaunchConfiguration(
        'distance_scale'
    )
    distance_bias_cm = LaunchConfiguration(
        'distance_bias_cm'
    )
    cube_center_offset_cm = LaunchConfiguration(
        'cube_center_offset_cm'
    )
    minimum_detection_distance_cm = (
        LaunchConfiguration(
            'minimum_detection_distance_cm'
        )
    )
    maximum_detection_distance_cm = (
        LaunchConfiguration(
            'maximum_detection_distance_cm'
        )
    )
    camera_forward_offset_cm = LaunchConfiguration(
        'camera_forward_offset_cm'
    )
    camera_left_offset_cm = LaunchConfiguration(
        'camera_left_offset_cm'
    )
    cube_merge_distance_cm = LaunchConfiguration(
        'cube_merge_distance_cm'
    )
    cube_cell_assignment_tolerance_cm = (
        LaunchConfiguration(
            'cube_cell_assignment_tolerance_cm'
        )
    )
    block_cube_cells = LaunchConfiguration(
        'block_cube_cells'
    )
    observation_hold_s = LaunchConfiguration(
        'observation_hold_s'
    )

    submit_cubes_to_grader = LaunchConfiguration(
        'submit_cubes_to_grader'
    )
    cube_grade_service_name = LaunchConfiguration(
        'cube_grade_service_name'
    )
    cube_grade_service_timeout_s = (
        LaunchConfiguration(
            'cube_grade_service_timeout_s'
        )
    )
    cube_grade_result_required = LaunchConfiguration(
        'cube_grade_result_required'
    )

    search_rescue_parameters = {
        # Unchanged v87 Task 2 explorer parameters.
        'scan_topic': scan_topic,
        'motion_action_name': motion_action_name,
        'cell_length_m': cell_length_m,
        'start_heading': start_heading,
        'wall_threshold_m': wall_threshold_m,
        'open_threshold_m': open_threshold_m,
        'drive_max_linear_x_mps': (
            drive_max_linear_x_mps
        ),
        'reverse_max_linear_x_mps': (
            reverse_max_linear_x_mps
        ),
        'reverse_backtracking_enabled': (
            reverse_backtracking_enabled
        ),
        'reverse_backtracking_max_consecutive_cells': (
            reverse_backtracking_max_consecutive_cells
        ),
        'reverse_position_tolerance_m': (
            reverse_position_tolerance_m
        ),
        'reverse_heading_tolerance_rad': (
            reverse_heading_tolerance_rad
        ),
        'rotate_max_angular_z_radps': (
            rotate_max_angular_z_radps
        ),
        'rotate_heading_tolerance_rad': (
            rotate_heading_tolerance_rad
        ),
        'max_cells_to_visit': max_cells_to_visit,
        'output_maze_file': output_maze_file,
        'shutdown_on_complete': shutdown_on_complete,
        'direction_priority': direction_priority,
        'race_planner_enabled': race_planner_enabled,
        'race_compact_known_transit': (
            race_compact_known_transit
        ),
        'race_turn_reverse_at_safe_junction': (
            race_turn_reverse_at_safe_junction
        ),
        'race_reverse_into_new_cells': (
            race_reverse_into_new_cells
        ),
        'race_cost_drive_forward': (
            race_cost_drive_forward
        ),
        'race_cost_drive_backward': (
            race_cost_drive_backward
        ),
        'race_cost_turn_90': race_cost_turn_90,
        'race_cost_turn_180': race_cost_turn_180,
        'race_cost_unvisited_info_bonus': (
            race_cost_unvisited_info_bonus
        ),
        'race_cost_straight_bonus': (
            race_cost_straight_bonus
        ),
        'race_lidar_lookahead_bonus': (
            race_lidar_lookahead_bonus
        ),

        # Task 3 submits cubes, not the Task 2 maze.
        'submit_maze_to_grader': False,

        'rect_image_topic': rect_image_topic,
        'raw_image_topic': raw_image_topic,
        'camera_info_topic': camera_info_topic,
        'apriltag_topic': apriltag_topic,
        'tag_size_m': tag_size_m,
        'tag_min_side_px': tag_min_side_px,
        'tag_min_aspect_ratio': tag_min_aspect_ratio,
        'tag_max_aspect_ratio': tag_max_aspect_ratio,
        'cube_crop_scale': cube_crop_scale,
        'camera_data_timeout_s': camera_data_timeout_s,
        'distance_scale': distance_scale,
        'distance_bias_cm': distance_bias_cm,
        'cube_center_offset_cm': (
            cube_center_offset_cm
        ),
        'minimum_detection_distance_cm': (
            minimum_detection_distance_cm
        ),
        'maximum_detection_distance_cm': (
            maximum_detection_distance_cm
        ),
        'camera_forward_offset_cm': (
            camera_forward_offset_cm
        ),
        'camera_left_offset_cm': (
            camera_left_offset_cm
        ),
        'cube_merge_distance_cm': (
            cube_merge_distance_cm
        ),
        'cube_cell_assignment_tolerance_cm': (
            cube_cell_assignment_tolerance_cm
        ),
        'block_cube_cells': block_cube_cells,
        'observation_hold_s': observation_hold_s,
        'submit_cubes_to_grader': (
            submit_cubes_to_grader
        ),
        'cube_grade_service_name': (
            cube_grade_service_name
        ),
        'cube_grade_service_timeout_s': (
            cube_grade_service_timeout_s
        ),
        'cube_grade_result_required': (
            cube_grade_result_required
        ),
    }

    return LaunchDescription([
        DeclareLaunchArgument(
            'launch_camera_pipeline',
            default_value='true',
        ),
        DeclareLaunchArgument(
            'launch_apriltag',
            default_value='true',
        ),

        # These defaults match the v87 Task 2 launch unless Task 3 requires
        # stopping at every cell for camera observation.
        DeclareLaunchArgument(
            'scan_topic',
            default_value='/ldlidar_node/scan',
        ),
        DeclareLaunchArgument(
            'motion_action_name',
            default_value=(
                '/darth_maul_control/'
                'execute_motion_primitive'
            ),
        ),
        DeclareLaunchArgument(
            'cell_length_m',
            default_value='0.254',
        ),
        DeclareLaunchArgument(
            'start_heading',
            default_value='1',
        ),
        DeclareLaunchArgument(
            'wall_threshold_m',
            default_value='0.18',
        ),
        DeclareLaunchArgument(
            'open_threshold_m',
            default_value='0.30',
        ),
        DeclareLaunchArgument(
            'drive_max_linear_x_mps',
            default_value='0.25',
        ),
        DeclareLaunchArgument(
            'reverse_max_linear_x_mps',
            default_value='0.085',
        ),
        DeclareLaunchArgument(
            'reverse_backtracking_enabled',
            default_value='true',
        ),
        DeclareLaunchArgument(
            'reverse_backtracking_max_consecutive_cells',
            default_value='0',
        ),
        DeclareLaunchArgument(
            'reverse_position_tolerance_m',
            default_value='0.020',
        ),
        DeclareLaunchArgument(
            'reverse_heading_tolerance_rad',
            default_value='0.070',
        ),
        DeclareLaunchArgument(
            'rotate_max_angular_z_radps',
            default_value='0.70',
        ),
        DeclareLaunchArgument(
            'rotate_heading_tolerance_rad',
            default_value='0.045',
        ),
        DeclareLaunchArgument(
            'max_cells_to_visit',
            default_value='0',
        ),
        DeclareLaunchArgument(
            'output_maze_file',
            default_value='',
        ),
        DeclareLaunchArgument(
            'shutdown_on_complete',
            default_value='true',
        ),
        DeclareLaunchArgument(
            'direction_priority',
            default_value='left_straight_right_back',
        ),
        DeclareLaunchArgument(
            'race_planner_enabled',
            default_value='true',
        ),

        # Task 3 deliberately disables multi-cell compaction. The robot stops
        # after each cell so the stationary camera gets an observation period.
        DeclareLaunchArgument(
            'race_compact_known_transit',
            default_value='false',
        ),
        DeclareLaunchArgument(
            'race_turn_reverse_at_safe_junction',
            default_value='true',
        ),
        DeclareLaunchArgument(
            'race_reverse_into_new_cells',
            default_value='false',
        ),
        DeclareLaunchArgument(
            'race_cost_drive_forward',
            default_value='3.4',
        ),
        DeclareLaunchArgument(
            'race_cost_drive_backward',
            default_value='5.2',
        ),
        DeclareLaunchArgument(
            'race_cost_turn_90',
            default_value='3.4',
        ),
        DeclareLaunchArgument(
            'race_cost_turn_180',
            default_value='6.8',
        ),
        DeclareLaunchArgument(
            'race_cost_unvisited_info_bonus',
            default_value='0.20',
        ),
        DeclareLaunchArgument(
            'race_cost_straight_bonus',
            default_value='0.25',
        ),
        DeclareLaunchArgument(
            'race_lidar_lookahead_bonus',
            default_value='0.20',
        ),

        DeclareLaunchArgument(
            'rect_image_topic',
            default_value=(
                '/ascamera/camera_publisher/'
                'rgb0/image_rect'
            ),
        ),
        DeclareLaunchArgument(
            'raw_image_topic',
            default_value=(
                '/ascamera/camera_publisher/'
                'rgb0/image'
            ),
        ),
        DeclareLaunchArgument(
            'camera_info_topic',
            default_value=(
                '/ascamera/camera_publisher/'
                'rgb0/camera_info'
            ),
        ),
        DeclareLaunchArgument(
            'apriltag_topic',
            default_value='/apriltag_detections',
        ),
        DeclareLaunchArgument(
            'tag_size_m',
            default_value='0.0162',
        ),
        DeclareLaunchArgument(
            'tag_min_side_px',
            default_value='15.0',
        ),
        DeclareLaunchArgument(
            'tag_min_aspect_ratio',
            default_value='0.60',
        ),
        DeclareLaunchArgument(
            'tag_max_aspect_ratio',
            default_value='1.50',
        ),
        DeclareLaunchArgument(
            'cube_crop_scale',
            default_value='2.30',
        ),
        DeclareLaunchArgument(
            'camera_data_timeout_s',
            default_value='0.75',
        ),

        # The newer v1 distance calibration.
        DeclareLaunchArgument(
            'distance_scale',
            default_value='1.18',
        ),
        DeclareLaunchArgument(
            'distance_bias_cm',
            default_value='-3.19',
        ),
        DeclareLaunchArgument(
            'cube_center_offset_cm',
            default_value='1.50',
        ),
        DeclareLaunchArgument(
            'minimum_detection_distance_cm',
            default_value='8.0',
        ),
        DeclareLaunchArgument(
            'maximum_detection_distance_cm',
            default_value='35.0',
        ),

        DeclareLaunchArgument(
            'camera_forward_offset_cm',
            default_value='8.0',
        ),
        DeclareLaunchArgument(
            'camera_left_offset_cm',
            default_value='0.0',
        ),
        DeclareLaunchArgument(
            'cube_merge_distance_cm',
            default_value='20.0',
        ),
        DeclareLaunchArgument(
            'cube_cell_assignment_tolerance_cm',
            default_value='13.0',
        ),
        DeclareLaunchArgument(
            'block_cube_cells',
            default_value='true',
        ),
        DeclareLaunchArgument(
            'observation_hold_s',
            default_value='0.60',
        ),

        DeclareLaunchArgument(
            'submit_cubes_to_grader',
            default_value='true',
        ),
        DeclareLaunchArgument(
            'cube_grade_service_name',
            default_value='/grade_cubes',
        ),
        DeclareLaunchArgument(
            'cube_grade_service_timeout_s',
            default_value='10.0',
        ),
        DeclareLaunchArgument(
            'cube_grade_result_required',
            default_value='false',
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('peripherals'),
                    'launch',
                    'image_pipeline.launch.py',
                ])
            ),
            condition=IfCondition(
                launch_camera_pipeline
            ),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('peripherals'),
                    'launch',
                    'apriltag.launch.py',
                ])
            ),
            condition=IfCondition(
                launch_apriltag
            ),
        ),

        Node(
            package='search_rescue',
            executable='search_rescue_node',
            name='search_rescue_node',
            output='screen',
            parameters=[
                search_rescue_parameters
            ],
        ),
    ])
