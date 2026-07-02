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
    grid_yaw_control_require_strong_evidence = LaunchConfiguration(
        'grid_yaw_control_require_strong_evidence'
    )
    grid_yaw_control_allow_single_wall = LaunchConfiguration(
        'grid_yaw_control_allow_single_wall'
    )
    grid_yaw_control_single_wall_min_confidence = LaunchConfiguration(
        'grid_yaw_control_single_wall_min_confidence'
    )
    grid_yaw_control_single_wall_max_abs_yaw_error_rad = LaunchConfiguration(
        'grid_yaw_control_single_wall_max_abs_yaw_error_rad'
    )
    grid_yaw_control_single_wall_max_rms_error_m = LaunchConfiguration(
        'grid_yaw_control_single_wall_max_rms_error_m'
    )
    grid_yaw_control_single_wall_min_span_x_m = LaunchConfiguration(
        'grid_yaw_control_single_wall_min_span_x_m'
    )
    grid_yaw_control_single_wall_min_support_count = LaunchConfiguration(
        'grid_yaw_control_single_wall_min_support_count'
    )
    lidar_parallelity_control_enabled = LaunchConfiguration(
        'lidar_parallelity_control_enabled'
    )
    lidar_parallelity_heading_hold_scale = LaunchConfiguration(
        'lidar_parallelity_heading_hold_scale'
    )
    lidar_parallelity_min_progress_for_control_m = LaunchConfiguration(
        'lidar_parallelity_min_progress_for_control_m'
    )
    lidar_parallelity_min_stable_samples = LaunchConfiguration(
        'lidar_parallelity_min_stable_samples'
    )
    lidar_parallelity_max_yaw_jump_rad = LaunchConfiguration(
        'lidar_parallelity_max_yaw_jump_rad'
    )
    lidar_parallelity_max_offset_jump_m = LaunchConfiguration(
        'lidar_parallelity_max_offset_jump_m'
    )
    lidar_parallelity_max_abs_yaw_error_rad = LaunchConfiguration(
        'lidar_parallelity_max_abs_yaw_error_rad'
    )
    lidar_parallelity_min_confidence = LaunchConfiguration(
        'lidar_parallelity_min_confidence'
    )
    lidar_parallelity_max_rms_error_m = LaunchConfiguration(
        'lidar_parallelity_max_rms_error_m'
    )
    lidar_parallelity_min_span_x_m = LaunchConfiguration(
        'lidar_parallelity_min_span_x_m'
    )
    lidar_parallelity_min_support_count = LaunchConfiguration(
        'lidar_parallelity_min_support_count'
    )
    lidar_parallelity_drift_validation_enabled = LaunchConfiguration(
        'lidar_parallelity_drift_validation_enabled'
    )
    lidar_parallelity_max_drift_per_m = LaunchConfiguration(
        'lidar_parallelity_max_drift_per_m'
    )
    k_lidar_parallelity_yaw = LaunchConfiguration('k_lidar_parallelity_yaw')
    k_lidar_parallelity_drift = LaunchConfiguration('k_lidar_parallelity_drift')
    max_lidar_parallelity_correction_radps = LaunchConfiguration(
        'max_lidar_parallelity_correction_radps'
    )
    lidar_parallelity_require_yaw_drift_consistency = LaunchConfiguration(
        'lidar_parallelity_require_yaw_drift_consistency'
    )
    lidar_parallelity_max_yaw_drift_disagreement_rad = LaunchConfiguration(
        'lidar_parallelity_max_yaw_drift_disagreement_rad'
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
    post_rotation_grid_yaw_refine_enabled = LaunchConfiguration(
        'post_rotation_grid_yaw_refine_enabled'
    )
    post_rotation_grid_yaw_refine_start_threshold_rad = LaunchConfiguration(
        'post_rotation_grid_yaw_refine_start_threshold_rad'
    )
    post_rotation_grid_yaw_refine_target_rad = LaunchConfiguration(
        'post_rotation_grid_yaw_refine_target_rad'
    )
    post_rotation_grid_yaw_refine_stable_samples = LaunchConfiguration(
        'post_rotation_grid_yaw_refine_stable_samples'
    )
    post_rotation_grid_yaw_refine_timeout_s = LaunchConfiguration(
        'post_rotation_grid_yaw_refine_timeout_s'
    )
    post_rotation_grid_yaw_refine_max_invalid_samples = LaunchConfiguration(
        'post_rotation_grid_yaw_refine_max_invalid_samples'
    )
    post_rotation_grid_yaw_refine_max_abs_error_rad = LaunchConfiguration(
        'post_rotation_grid_yaw_refine_max_abs_error_rad'
    )
    post_rotation_grid_yaw_refine_min_confidence = LaunchConfiguration(
        'post_rotation_grid_yaw_refine_min_confidence'
    )
    post_rotation_grid_yaw_refine_kp = LaunchConfiguration(
        'post_rotation_grid_yaw_refine_kp'
    )
    post_rotation_grid_yaw_refine_max_angular_z_radps = LaunchConfiguration(
        'post_rotation_grid_yaw_refine_max_angular_z_radps'
    )
    post_rotation_grid_yaw_refine_require_valid = LaunchConfiguration(
        'post_rotation_grid_yaw_refine_require_valid'
    )
    rotate_timeout_accept_heading_error_rad = LaunchConfiguration(
        'rotate_timeout_accept_heading_error_rad'
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
            'grid_yaw_control_require_strong_evidence',
            default_value='true',
            description='Require strong evidence before active DRIVE_FORWARD grid yaw correction.',
        ),
        DeclareLaunchArgument(
            'grid_yaw_control_allow_single_wall',
            default_value='true',
            description='Allow single-wall grid yaw to actively control DRIVE_FORWARD yaw.',
        ),
        DeclareLaunchArgument(
            'grid_yaw_control_single_wall_min_confidence',
            default_value='0.60',
            description='Minimum confidence for experimental single-wall grid yaw control.',
        ),
        DeclareLaunchArgument(
            'grid_yaw_control_single_wall_max_abs_yaw_error_rad',
            default_value='0.100',
            description='Maximum single-wall yaw error allowed for active grid yaw control.',
        ),
        DeclareLaunchArgument(
            'grid_yaw_control_single_wall_max_rms_error_m',
            default_value='0.020',
            description='Maximum contributing wall RMS error for single-wall yaw control.',
        ),
        DeclareLaunchArgument(
            'grid_yaw_control_single_wall_min_span_x_m',
            default_value='0.220',
            description='Minimum contributing wall x-span for single-wall yaw control.',
        ),
        DeclareLaunchArgument(
            'grid_yaw_control_single_wall_min_support_count',
            default_value='80',
            description='Minimum contributing wall support points for single-wall yaw control.',
        ),
        DeclareLaunchArgument('lidar_parallelity_control_enabled', default_value='true'),
        DeclareLaunchArgument('lidar_parallelity_heading_hold_scale', default_value='1.0'),
        DeclareLaunchArgument('lidar_parallelity_min_progress_for_control_m', default_value='0.040'),
        DeclareLaunchArgument('lidar_parallelity_min_stable_samples', default_value='3'),
        DeclareLaunchArgument('lidar_parallelity_max_yaw_jump_rad', default_value='0.025'),
        DeclareLaunchArgument('lidar_parallelity_max_offset_jump_m', default_value='0.035'),
        DeclareLaunchArgument('lidar_parallelity_max_abs_yaw_error_rad', default_value='0.100'),
        DeclareLaunchArgument('lidar_parallelity_min_confidence', default_value='0.60'),
        DeclareLaunchArgument('lidar_parallelity_max_rms_error_m', default_value='0.025'),
        DeclareLaunchArgument('lidar_parallelity_min_span_x_m', default_value='0.220'),
        DeclareLaunchArgument('lidar_parallelity_min_support_count', default_value='80'),
        DeclareLaunchArgument('lidar_parallelity_drift_validation_enabled', default_value='true'),
        DeclareLaunchArgument('lidar_parallelity_max_drift_per_m', default_value='0.35'),
        DeclareLaunchArgument('k_lidar_parallelity_yaw', default_value='0.80'),
        DeclareLaunchArgument('k_lidar_parallelity_drift', default_value='0.20'),
        DeclareLaunchArgument('max_lidar_parallelity_correction_radps', default_value='0.060'),
        DeclareLaunchArgument('lidar_parallelity_require_yaw_drift_consistency', default_value='true'),
        DeclareLaunchArgument('lidar_parallelity_max_yaw_drift_disagreement_rad', default_value='0.060'),
        DeclareLaunchArgument('lidar_progress_temporal_filter_enabled', default_value='false'),
        DeclareLaunchArgument('lidar_progress_temporal_max_backtrack_m', default_value='0.015'),
        DeclareLaunchArgument('lidar_progress_temporal_max_jump_m', default_value='0.080'),
        DeclareLaunchArgument('lidar_progress_temporal_max_degraded_samples', default_value='2'),
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
            'post_rotation_grid_yaw_refine_enabled',
            default_value='false',
            description='Enable LiDAR side-wall yaw refinement after ROTATE_RELATIVE.',
        ),
        DeclareLaunchArgument(
            'post_rotation_grid_yaw_refine_start_threshold_rad',
            default_value='0.030',
            description='Grid yaw error threshold that triggers post-rotation refinement.',
        ),
        DeclareLaunchArgument(
            'post_rotation_grid_yaw_refine_target_rad',
            default_value='0.015',
            description='Target grid yaw error for post-rotation refinement completion.',
        ),
        DeclareLaunchArgument(
            'post_rotation_grid_yaw_refine_stable_samples',
            default_value='3',
            description='Consecutive in-target samples required after post-rotation refinement.',
        ),
        DeclareLaunchArgument(
            'post_rotation_grid_yaw_refine_timeout_s',
            default_value='2.0',
            description='Maximum seconds spent in post-rotation refinement.',
        ),
        DeclareLaunchArgument(
            'post_rotation_grid_yaw_refine_max_invalid_samples',
            default_value='8',
            description='Invalid grid-yaw samples tolerated during post-rotation refinement.',
        ),
        DeclareLaunchArgument(
            'post_rotation_grid_yaw_refine_max_abs_error_rad',
            default_value='0.20',
            description='Maximum grid yaw error considered safe to correct after rotation.',
        ),
        DeclareLaunchArgument(
            'post_rotation_grid_yaw_refine_min_confidence',
            default_value='0.60',
            description='Minimum grid alignment confidence for post-rotation refinement.',
        ),
        DeclareLaunchArgument(
            'post_rotation_grid_yaw_refine_kp',
            default_value='1.00',
            description='Post-rotation grid yaw refinement proportional gain.',
        ),
        DeclareLaunchArgument(
            'post_rotation_grid_yaw_refine_max_angular_z_radps',
            default_value='0.080',
            description='Max angular speed during post-rotation grid yaw refinement.',
        ),
        DeclareLaunchArgument(
            'post_rotation_grid_yaw_refine_require_valid',
            default_value='false',
            description='If true, fail ROTATE_RELATIVE when post-rotation grid yaw is unavailable.',
        ),
        DeclareLaunchArgument(
            'rotate_timeout_accept_heading_error_rad',
            default_value='0.090',
            description=(
                'Accept ROTATE_RELATIVE timeout as success when odom heading error '
                'is at or below this threshold.'
            ),
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
                    'grid_yaw_control_require_strong_evidence': ParameterValue(
                        grid_yaw_control_require_strong_evidence,
                        value_type=bool,
                    ),
                    'grid_yaw_control_allow_single_wall': ParameterValue(
                        grid_yaw_control_allow_single_wall,
                        value_type=bool,
                    ),
                    'grid_yaw_control_single_wall_min_confidence': ParameterValue(
                        grid_yaw_control_single_wall_min_confidence,
                        value_type=float,
                    ),
                    'grid_yaw_control_single_wall_max_abs_yaw_error_rad': ParameterValue(
                        grid_yaw_control_single_wall_max_abs_yaw_error_rad,
                        value_type=float,
                    ),
                    'grid_yaw_control_single_wall_max_rms_error_m': ParameterValue(
                        grid_yaw_control_single_wall_max_rms_error_m,
                        value_type=float,
                    ),
                    'grid_yaw_control_single_wall_min_span_x_m': ParameterValue(
                        grid_yaw_control_single_wall_min_span_x_m,
                        value_type=float,
                    ),
                    'grid_yaw_control_single_wall_min_support_count': ParameterValue(
                        grid_yaw_control_single_wall_min_support_count,
                        value_type=int,
                    ),
                    'lidar_parallelity_control_enabled': ParameterValue(
                        lidar_parallelity_control_enabled, value_type=bool
                    ),
                    'lidar_parallelity_heading_hold_scale': ParameterValue(
                        lidar_parallelity_heading_hold_scale, value_type=float
                    ),
                    'lidar_parallelity_min_progress_for_control_m': ParameterValue(
                        lidar_parallelity_min_progress_for_control_m, value_type=float
                    ),
                    'lidar_parallelity_min_stable_samples': ParameterValue(
                        lidar_parallelity_min_stable_samples, value_type=int
                    ),
                    'lidar_parallelity_max_yaw_jump_rad': ParameterValue(
                        lidar_parallelity_max_yaw_jump_rad, value_type=float
                    ),
                    'lidar_parallelity_max_offset_jump_m': ParameterValue(
                        lidar_parallelity_max_offset_jump_m, value_type=float
                    ),
                    'lidar_parallelity_max_abs_yaw_error_rad': ParameterValue(
                        lidar_parallelity_max_abs_yaw_error_rad, value_type=float
                    ),
                    'lidar_parallelity_min_confidence': ParameterValue(
                        lidar_parallelity_min_confidence, value_type=float
                    ),
                    'lidar_parallelity_max_rms_error_m': ParameterValue(
                        lidar_parallelity_max_rms_error_m, value_type=float
                    ),
                    'lidar_parallelity_min_span_x_m': ParameterValue(
                        lidar_parallelity_min_span_x_m, value_type=float
                    ),
                    'lidar_parallelity_min_support_count': ParameterValue(
                        lidar_parallelity_min_support_count, value_type=int
                    ),
                    'lidar_parallelity_drift_validation_enabled': ParameterValue(
                        lidar_parallelity_drift_validation_enabled, value_type=bool
                    ),
                    'lidar_parallelity_max_drift_per_m': ParameterValue(
                        lidar_parallelity_max_drift_per_m, value_type=float
                    ),
                    'k_lidar_parallelity_yaw': ParameterValue(
                        k_lidar_parallelity_yaw, value_type=float
                    ),
                    'k_lidar_parallelity_drift': ParameterValue(
                        k_lidar_parallelity_drift, value_type=float
                    ),
                    'max_lidar_parallelity_correction_radps': ParameterValue(
                        max_lidar_parallelity_correction_radps, value_type=float
                    ),
                    'lidar_parallelity_require_yaw_drift_consistency': ParameterValue(
                        lidar_parallelity_require_yaw_drift_consistency, value_type=bool
                    ),
                    'lidar_parallelity_max_yaw_drift_disagreement_rad': ParameterValue(
                        lidar_parallelity_max_yaw_drift_disagreement_rad,
                        value_type=float,
                    ),
                    'lidar_progress_temporal_filter_enabled': ParameterValue(
                        lidar_progress_temporal_filter_enabled, value_type=bool
                    ),
                    'lidar_progress_temporal_max_backtrack_m': ParameterValue(
                        lidar_progress_temporal_max_backtrack_m, value_type=float
                    ),
                    'lidar_progress_temporal_max_jump_m': ParameterValue(
                        lidar_progress_temporal_max_jump_m, value_type=float
                    ),
                    'lidar_progress_temporal_max_degraded_samples': ParameterValue(
                        lidar_progress_temporal_max_degraded_samples, value_type=int
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
                    'post_rotation_grid_yaw_refine_enabled': ParameterValue(
                        post_rotation_grid_yaw_refine_enabled,
                        value_type=bool,
                    ),
                    'post_rotation_grid_yaw_refine_start_threshold_rad': ParameterValue(
                        post_rotation_grid_yaw_refine_start_threshold_rad,
                        value_type=float,
                    ),
                    'post_rotation_grid_yaw_refine_target_rad': ParameterValue(
                        post_rotation_grid_yaw_refine_target_rad,
                        value_type=float,
                    ),
                    'post_rotation_grid_yaw_refine_stable_samples': ParameterValue(
                        post_rotation_grid_yaw_refine_stable_samples,
                        value_type=int,
                    ),
                    'post_rotation_grid_yaw_refine_timeout_s': ParameterValue(
                        post_rotation_grid_yaw_refine_timeout_s,
                        value_type=float,
                    ),
                    'post_rotation_grid_yaw_refine_max_invalid_samples': ParameterValue(
                        post_rotation_grid_yaw_refine_max_invalid_samples,
                        value_type=int,
                    ),
                    'post_rotation_grid_yaw_refine_max_abs_error_rad': ParameterValue(
                        post_rotation_grid_yaw_refine_max_abs_error_rad,
                        value_type=float,
                    ),
                    'post_rotation_grid_yaw_refine_min_confidence': ParameterValue(
                        post_rotation_grid_yaw_refine_min_confidence,
                        value_type=float,
                    ),
                    'post_rotation_grid_yaw_refine_kp': ParameterValue(
                        post_rotation_grid_yaw_refine_kp,
                        value_type=float,
                    ),
                    'post_rotation_grid_yaw_refine_max_angular_z_radps': ParameterValue(
                        post_rotation_grid_yaw_refine_max_angular_z_radps,
                        value_type=float,
                    ),
                    'post_rotation_grid_yaw_refine_require_valid': ParameterValue(
                        post_rotation_grid_yaw_refine_require_valid,
                        value_type=bool,
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
