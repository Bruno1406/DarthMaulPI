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
    grid_yaw_active_heading_hold_scale = LaunchConfiguration(
        'grid_yaw_active_heading_hold_scale'
    )
    translation_progress_source = LaunchConfiguration('translation_progress_source')
    pre_translation_grid_yaw_align_enabled = LaunchConfiguration(
        'pre_translation_grid_yaw_align_enabled'
    )
    pre_translation_grid_yaw_align_start_threshold_rad = LaunchConfiguration(
        'pre_translation_grid_yaw_align_start_threshold_rad'
    )
    pre_translation_grid_yaw_align_target_rad = LaunchConfiguration(
        'pre_translation_grid_yaw_align_target_rad'
    )
    pre_translation_grid_yaw_align_stable_samples = LaunchConfiguration(
        'pre_translation_grid_yaw_align_stable_samples'
    )
    pre_translation_grid_yaw_align_max_invalid_samples = LaunchConfiguration(
        'pre_translation_grid_yaw_align_max_invalid_samples'
    )
    translation_lidar_required_invalid_max_consecutive_samples = LaunchConfiguration(
        'translation_lidar_required_invalid_max_consecutive_samples'
    )
    lidar_progress_geometry_validation_enabled = LaunchConfiguration(
        'lidar_progress_geometry_validation_enabled'
    )
    lidar_progress_axial_wall_min_distance_m = LaunchConfiguration(
        'lidar_progress_axial_wall_min_distance_m'
    )
    lidar_progress_axial_wall_max_distance_m = LaunchConfiguration(
        'lidar_progress_axial_wall_max_distance_m'
    )
    lidar_progress_axial_wall_max_abs_lateral_m = LaunchConfiguration(
        'lidar_progress_axial_wall_max_abs_lateral_m'
    )
    lidar_progress_axial_wall_min_points = LaunchConfiguration(
        'lidar_progress_axial_wall_min_points'
    )
    lidar_progress_axial_wall_min_span_y_m = LaunchConfiguration(
        'lidar_progress_axial_wall_min_span_y_m'
    )
    lidar_progress_axial_wall_max_rms_error_m = LaunchConfiguration(
        'lidar_progress_axial_wall_max_rms_error_m'
    )
    lidar_progress_axial_wall_max_abs_yaw_error_rad = LaunchConfiguration(
        'lidar_progress_axial_wall_max_abs_yaw_error_rad'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'grid_alignment_control_enabled',
            default_value='true',
            description=(
                'Enable LiDAR grid-alignment control terms. '
                'When false, grid alignment remains diagnostic only.'
            ),
        ),
        DeclareLaunchArgument(
            'grid_yaw_correction_enabled',
            default_value='true',
            description=(
                'Enable low-gain LiDAR grid-yaw correction during forward translation.'
            ),
        ),
        DeclareLaunchArgument(
            'k_grid_yaw',
            default_value='1.20',
            description='Grid yaw correction proportional gain.',
        ),
        DeclareLaunchArgument(
            'max_grid_yaw_correction_radps',
            default_value='0.060',
            description='Absolute cap for grid yaw correction in rad/s.',
        ),
        DeclareLaunchArgument(
            'grid_yaw_active_heading_hold_scale',
            default_value='0.0',
            description='Heading-hold scale while grid yaw correction is active.',
        ),
        DeclareLaunchArgument(
            'translation_progress_source',
            default_value='lidar_required',
            description=(
                'Translation distance authority: lidar_required, '
                'lidar_when_consistent, or odom_only.'
            ),
        ),
        DeclareLaunchArgument(
            'pre_translation_grid_yaw_align_enabled',
            default_value='false',
            description=(
                'Enable experimental pre-translation LiDAR grid-yaw alignment. '
                'Default false because raw front/rear distance progress must not be '
                'contaminated by rotate-in-place side-wall aliasing.'
            ),
        ),
        DeclareLaunchArgument(
            'pre_translation_grid_yaw_align_start_threshold_rad',
            default_value='0.035',
            description='Starting grid-yaw error threshold for pre-alignment.',
        ),
        DeclareLaunchArgument(
            'pre_translation_grid_yaw_align_target_rad',
            default_value='0.015',
            description='Target grid-yaw error for pre-alignment completion.',
        ),
        DeclareLaunchArgument(
            'pre_translation_grid_yaw_align_stable_samples',
            default_value='3',
            description='Consecutive in-target samples required after pre-alignment.',
        ),
        DeclareLaunchArgument(
            'pre_translation_grid_yaw_align_max_invalid_samples',
            default_value='3',
            description='Consecutive inactive/invalid pre-alignment samples before failure.',
        ),
        DeclareLaunchArgument(
            'translation_lidar_required_invalid_max_consecutive_samples',
            default_value='2',
            description='Consecutive invalid LiDAR progress samples allowed in lidar_required mode.',
        ),
        DeclareLaunchArgument(
            'lidar_progress_geometry_validation_enabled',
            default_value='true',
            description='Require axial front/rear wall geometry for LiDAR progress.',
        ),
        DeclareLaunchArgument(
            'lidar_progress_axial_wall_min_distance_m',
            default_value='0.05',
            description='Minimum +/-x distance for front/rear axial wall candidates.',
        ),
        DeclareLaunchArgument(
            'lidar_progress_axial_wall_max_distance_m',
            default_value='2.00',
            description='Maximum +/-x distance for front/rear axial wall candidates.',
        ),
        DeclareLaunchArgument(
            'lidar_progress_axial_wall_max_abs_lateral_m',
            default_value='0.35',
            description='Maximum absolute lateral y for front/rear axial wall candidates.',
        ),
        DeclareLaunchArgument(
            'lidar_progress_axial_wall_min_points',
            default_value='8',
            description='Minimum point support for front/rear axial wall fitting.',
        ),
        DeclareLaunchArgument(
            'lidar_progress_axial_wall_min_span_y_m',
            default_value='0.08',
            description='Minimum lateral span for front/rear axial wall fitting.',
        ),
        DeclareLaunchArgument(
            'lidar_progress_axial_wall_max_rms_error_m',
            default_value='0.025',
            description='Maximum RMS residual for front/rear axial wall fitting.',
        ),
        DeclareLaunchArgument(
            'lidar_progress_axial_wall_max_abs_yaw_error_rad',
            default_value='0.35',
            description='Maximum axial wall yaw error accepted for LiDAR progress.',
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
                    'grid_yaw_active_heading_hold_scale': ParameterValue(
                        grid_yaw_active_heading_hold_scale,
                        value_type=float,
                    ),
                    'translation_progress_source': ParameterValue(
                        translation_progress_source,
                        value_type=str,
                    ),
                    'pre_translation_grid_yaw_align_enabled': ParameterValue(
                        pre_translation_grid_yaw_align_enabled,
                        value_type=bool,
                    ),
                    'pre_translation_grid_yaw_align_start_threshold_rad': ParameterValue(
                        pre_translation_grid_yaw_align_start_threshold_rad,
                        value_type=float,
                    ),
                    'pre_translation_grid_yaw_align_target_rad': ParameterValue(
                        pre_translation_grid_yaw_align_target_rad,
                        value_type=float,
                    ),
                    'pre_translation_grid_yaw_align_stable_samples': ParameterValue(
                        pre_translation_grid_yaw_align_stable_samples,
                        value_type=int,
                    ),
                    'pre_translation_grid_yaw_align_max_invalid_samples': ParameterValue(
                        pre_translation_grid_yaw_align_max_invalid_samples,
                        value_type=int,
                    ),
                    'translation_lidar_required_invalid_max_consecutive_samples': ParameterValue(
                        translation_lidar_required_invalid_max_consecutive_samples,
                        value_type=int,
                    ),
                    'lidar_progress_geometry_validation_enabled': ParameterValue(
                        lidar_progress_geometry_validation_enabled,
                        value_type=bool,
                    ),
                    'lidar_progress_axial_wall_min_distance_m': ParameterValue(
                        lidar_progress_axial_wall_min_distance_m,
                        value_type=float,
                    ),
                    'lidar_progress_axial_wall_max_distance_m': ParameterValue(
                        lidar_progress_axial_wall_max_distance_m,
                        value_type=float,
                    ),
                    'lidar_progress_axial_wall_max_abs_lateral_m': ParameterValue(
                        lidar_progress_axial_wall_max_abs_lateral_m,
                        value_type=float,
                    ),
                    'lidar_progress_axial_wall_min_points': ParameterValue(
                        lidar_progress_axial_wall_min_points,
                        value_type=int,
                    ),
                    'lidar_progress_axial_wall_min_span_y_m': ParameterValue(
                        lidar_progress_axial_wall_min_span_y_m,
                        value_type=float,
                    ),
                    'lidar_progress_axial_wall_max_rms_error_m': ParameterValue(
                        lidar_progress_axial_wall_max_rms_error_m,
                        value_type=float,
                    ),
                    'lidar_progress_axial_wall_max_abs_yaw_error_rad': ParameterValue(
                        lidar_progress_axial_wall_max_abs_yaw_error_rad,
                        value_type=float,
                    ),
                },
            ],
            output='screen',
        ),
    ])
