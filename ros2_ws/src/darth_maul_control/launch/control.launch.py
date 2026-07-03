from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _declare(name: str, default_value: str, description: str = ''):
    return DeclareLaunchArgument(
        name,
        default_value=default_value,
        description=description,
    )


def generate_launch_description():
    package_name = 'darth_maul_control'
    params_file = PathJoinSubstitution([
        FindPackageShare(package_name),
        'config',
        'control_params.yaml',
    ])

    translation_progress_source = LaunchConfiguration('translation_progress_source')
    translation_lidar_required_invalid_max_consecutive_samples = LaunchConfiguration(
        'translation_lidar_required_invalid_max_consecutive_samples'
    )
    lidar_progress_temporal_filter_enabled = LaunchConfiguration(
        'lidar_progress_temporal_filter_enabled'
    )
    lidar_progress_temporal_max_backtrack_m = LaunchConfiguration(
        'lidar_progress_temporal_max_backtrack_m'
    )
    lidar_progress_temporal_max_jump_m = LaunchConfiguration(
        'lidar_progress_temporal_max_jump_m'
    )
    lidar_progress_temporal_max_degraded_samples = LaunchConfiguration(
        'lidar_progress_temporal_max_degraded_samples'
    )

    grid_live_control_enabled = LaunchConfiguration('grid_live_control_enabled')
    k_grid_lateral = LaunchConfiguration('k_grid_lateral')
    max_grid_lateral_mps = LaunchConfiguration('max_grid_lateral_mps')
    grid_lateral_min_confidence = LaunchConfiguration('grid_lateral_min_confidence')
    k_grid_live_yaw = LaunchConfiguration('k_grid_live_yaw')
    max_grid_live_yaw_correction_radps = LaunchConfiguration(
        'max_grid_live_yaw_correction_radps'
    )
    grid_manhattan_yaw_enabled = LaunchConfiguration('grid_manhattan_yaw_enabled')
    grid_manhattan_yaw_min_range_m = LaunchConfiguration('grid_manhattan_yaw_min_range_m')
    grid_manhattan_yaw_max_range_m = LaunchConfiguration('grid_manhattan_yaw_max_range_m')
    grid_manhattan_yaw_max_point_gap_m = LaunchConfiguration(
        'grid_manhattan_yaw_max_point_gap_m'
    )
    grid_manhattan_yaw_max_range_jump_m = LaunchConfiguration(
        'grid_manhattan_yaw_max_range_jump_m'
    )
    grid_manhattan_yaw_min_cluster_points = LaunchConfiguration(
        'grid_manhattan_yaw_min_cluster_points'
    )
    grid_manhattan_yaw_min_segment_points = LaunchConfiguration(
        'grid_manhattan_yaw_min_segment_points'
    )
    grid_manhattan_yaw_min_segment_length_m = LaunchConfiguration(
        'grid_manhattan_yaw_min_segment_length_m'
    )
    grid_manhattan_yaw_max_line_rms_m = LaunchConfiguration(
        'grid_manhattan_yaw_max_line_rms_m'
    )
    grid_manhattan_yaw_min_line_count = LaunchConfiguration(
        'grid_manhattan_yaw_min_line_count'
    )
    grid_manhattan_yaw_min_total_weight = LaunchConfiguration(
        'grid_manhattan_yaw_min_total_weight'
    )
    grid_manhattan_yaw_min_concentration = LaunchConfiguration(
        'grid_manhattan_yaw_min_concentration'
    )
    grid_manhattan_yaw_min_confidence = LaunchConfiguration(
        'grid_manhattan_yaw_min_confidence'
    )
    grid_manhattan_yaw_max_abs_error_rad = LaunchConfiguration(
        'grid_manhattan_yaw_max_abs_error_rad'
    )
    grid_live_yaw_min_confidence = LaunchConfiguration(
        'grid_live_yaw_min_confidence'
    )
    grid_live_max_abs_yaw_error_rad = LaunchConfiguration(
        'grid_live_max_abs_yaw_error_rad'
    )
    grid_live_min_span_x_m = LaunchConfiguration('grid_live_min_span_x_m')
    grid_live_min_support_count = LaunchConfiguration('grid_live_min_support_count')
    grid_live_max_rms_error_m = LaunchConfiguration('grid_live_max_rms_error_m')
    grid_virtual_cell_boundary_margin_m = LaunchConfiguration(
        'grid_virtual_cell_boundary_margin_m'
    )
    grid_reacquire_stable_samples = LaunchConfiguration(
        'grid_reacquire_stable_samples'
    )
    grid_reacquire_small_yaw_rad = LaunchConfiguration(
        'grid_reacquire_small_yaw_rad'
    )
    grid_reacquire_large_yaw_rad = LaunchConfiguration(
        'grid_reacquire_large_yaw_rad'
    )
    grid_reacquire_speed_scale = LaunchConfiguration('grid_reacquire_speed_scale')
    rotate_timeout_accept_heading_error_rad = LaunchConfiguration(
        'rotate_timeout_accept_heading_error_rad'
    )

    return LaunchDescription([
        _declare(
            'translation_progress_source',
            'lidar_required',
            'Translation distance authority: lidar_required, lidar_when_consistent, or odom_only.',
        ),
        _declare(
            'translation_lidar_required_invalid_max_consecutive_samples',
            '2',
            'Consecutive invalid LiDAR progress samples allowed in lidar_required mode.',
        ),
        _declare('lidar_progress_temporal_filter_enabled', 'false'),
        _declare('lidar_progress_temporal_max_backtrack_m', '0.015'),
        _declare('lidar_progress_temporal_max_jump_m', '0.080'),
        _declare('lidar_progress_temporal_max_degraded_samples', '2'),
        _declare('grid_live_control_enabled', 'true'),
        _declare('k_grid_lateral', '1.40'),
        _declare('max_grid_lateral_mps', '0.025'),
        _declare('grid_lateral_min_confidence', '0.55'),
        _declare('k_grid_live_yaw', '2.20'),
        _declare('max_grid_live_yaw_correction_radps', '0.080'),
        _declare('grid_manhattan_yaw_enabled', 'true'),
        _declare('grid_manhattan_yaw_min_range_m', '0.08'),
        _declare('grid_manhattan_yaw_max_range_m', '2.50'),
        _declare('grid_manhattan_yaw_max_point_gap_m', '0.055'),
        _declare('grid_manhattan_yaw_max_range_jump_m', '0.080'),
        _declare('grid_manhattan_yaw_min_cluster_points', '8'),
        _declare('grid_manhattan_yaw_min_segment_points', '8'),
        _declare('grid_manhattan_yaw_min_segment_length_m', '0.120'),
        _declare('grid_manhattan_yaw_max_line_rms_m', '0.020'),
        _declare('grid_manhattan_yaw_min_line_count', '1'),
        _declare('grid_manhattan_yaw_min_total_weight', '0.20'),
        _declare('grid_manhattan_yaw_min_concentration', '0.70'),
        _declare('grid_manhattan_yaw_min_confidence', '0.55'),
        _declare('grid_manhattan_yaw_max_abs_error_rad', '0.140'),
        _declare('grid_live_yaw_min_confidence', '0.55'),
        _declare('grid_live_max_abs_yaw_error_rad', '0.120'),
        _declare('grid_live_min_span_x_m', '0.120'),
        _declare('grid_live_min_support_count', '8'),
        _declare('grid_live_max_rms_error_m', '0.025'),
        _declare('grid_virtual_cell_boundary_margin_m', '0.025'),
        _declare('grid_reacquire_stable_samples', '3'),
        _declare('grid_reacquire_small_yaw_rad', '0.040'),
        _declare('grid_reacquire_large_yaw_rad', '0.100'),
        _declare('grid_reacquire_speed_scale', '0.65'),
        _declare(
            'rotate_timeout_accept_heading_error_rad',
            '0.090',
            'Accept ROTATE_RELATIVE timeout as success near the odom target.',
        ),
        Node(
            package=package_name,
            executable='control_node',
            name='darth_maul_control',
            parameters=[
                params_file,
                {
                    'translation_progress_source': ParameterValue(
                        translation_progress_source,
                        value_type=str,
                    ),
                    'translation_lidar_required_invalid_max_consecutive_samples': ParameterValue(
                        translation_lidar_required_invalid_max_consecutive_samples,
                        value_type=int,
                    ),
                    'lidar_progress_temporal_filter_enabled': ParameterValue(
                        lidar_progress_temporal_filter_enabled,
                        value_type=bool,
                    ),
                    'lidar_progress_temporal_max_backtrack_m': ParameterValue(
                        lidar_progress_temporal_max_backtrack_m,
                        value_type=float,
                    ),
                    'lidar_progress_temporal_max_jump_m': ParameterValue(
                        lidar_progress_temporal_max_jump_m,
                        value_type=float,
                    ),
                    'lidar_progress_temporal_max_degraded_samples': ParameterValue(
                        lidar_progress_temporal_max_degraded_samples,
                        value_type=int,
                    ),
                    'grid_live_control_enabled': ParameterValue(
                        grid_live_control_enabled,
                        value_type=bool,
                    ),
                    'k_grid_lateral': ParameterValue(
                        k_grid_lateral,
                        value_type=float,
                    ),
                    'max_grid_lateral_mps': ParameterValue(
                        max_grid_lateral_mps,
                        value_type=float,
                    ),
                    'grid_lateral_min_confidence': ParameterValue(
                        grid_lateral_min_confidence,
                        value_type=float,
                    ),
                    'k_grid_live_yaw': ParameterValue(
                        k_grid_live_yaw,
                        value_type=float,
                    ),
                    'max_grid_live_yaw_correction_radps': ParameterValue(
                        max_grid_live_yaw_correction_radps,
                        value_type=float,
                    ),
                    'grid_manhattan_yaw_enabled': ParameterValue(
                        grid_manhattan_yaw_enabled,
                        value_type=bool,
                    ),
                    'grid_manhattan_yaw_min_range_m': ParameterValue(
                        grid_manhattan_yaw_min_range_m,
                        value_type=float,
                    ),
                    'grid_manhattan_yaw_max_range_m': ParameterValue(
                        grid_manhattan_yaw_max_range_m,
                        value_type=float,
                    ),
                    'grid_manhattan_yaw_max_point_gap_m': ParameterValue(
                        grid_manhattan_yaw_max_point_gap_m,
                        value_type=float,
                    ),
                    'grid_manhattan_yaw_max_range_jump_m': ParameterValue(
                        grid_manhattan_yaw_max_range_jump_m,
                        value_type=float,
                    ),
                    'grid_manhattan_yaw_min_cluster_points': ParameterValue(
                        grid_manhattan_yaw_min_cluster_points,
                        value_type=int,
                    ),
                    'grid_manhattan_yaw_min_segment_points': ParameterValue(
                        grid_manhattan_yaw_min_segment_points,
                        value_type=int,
                    ),
                    'grid_manhattan_yaw_min_segment_length_m': ParameterValue(
                        grid_manhattan_yaw_min_segment_length_m,
                        value_type=float,
                    ),
                    'grid_manhattan_yaw_max_line_rms_m': ParameterValue(
                        grid_manhattan_yaw_max_line_rms_m,
                        value_type=float,
                    ),
                    'grid_manhattan_yaw_min_line_count': ParameterValue(
                        grid_manhattan_yaw_min_line_count,
                        value_type=int,
                    ),
                    'grid_manhattan_yaw_min_total_weight': ParameterValue(
                        grid_manhattan_yaw_min_total_weight,
                        value_type=float,
                    ),
                    'grid_manhattan_yaw_min_concentration': ParameterValue(
                        grid_manhattan_yaw_min_concentration,
                        value_type=float,
                    ),
                    'grid_manhattan_yaw_min_confidence': ParameterValue(
                        grid_manhattan_yaw_min_confidence,
                        value_type=float,
                    ),
                    'grid_manhattan_yaw_max_abs_error_rad': ParameterValue(
                        grid_manhattan_yaw_max_abs_error_rad,
                        value_type=float,
                    ),
                    'grid_live_yaw_min_confidence': ParameterValue(
                        grid_live_yaw_min_confidence,
                        value_type=float,
                    ),
                    'grid_live_max_abs_yaw_error_rad': ParameterValue(
                        grid_live_max_abs_yaw_error_rad,
                        value_type=float,
                    ),
                    'grid_live_min_span_x_m': ParameterValue(
                        grid_live_min_span_x_m,
                        value_type=float,
                    ),
                    'grid_live_min_support_count': ParameterValue(
                        grid_live_min_support_count,
                        value_type=int,
                    ),
                    'grid_live_max_rms_error_m': ParameterValue(
                        grid_live_max_rms_error_m,
                        value_type=float,
                    ),
                    'grid_virtual_cell_boundary_margin_m': ParameterValue(
                        grid_virtual_cell_boundary_margin_m,
                        value_type=float,
                    ),
                    'grid_reacquire_stable_samples': ParameterValue(
                        grid_reacquire_stable_samples,
                        value_type=int,
                    ),
                    'grid_reacquire_small_yaw_rad': ParameterValue(
                        grid_reacquire_small_yaw_rad,
                        value_type=float,
                    ),
                    'grid_reacquire_large_yaw_rad': ParameterValue(
                        grid_reacquire_large_yaw_rad,
                        value_type=float,
                    ),
                    'grid_reacquire_speed_scale': ParameterValue(
                        grid_reacquire_speed_scale,
                        value_type=float,
                    ),
                    'rotate_timeout_accept_heading_error_rad': ParameterValue(
                        rotate_timeout_accept_heading_error_rad,
                        value_type=float,
                    ),
                },
            ],
            output='screen',
        ),
    ])
