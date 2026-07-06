from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_VALIDATION_PARAM = 'lidar_progress_geometry_validation' + '_enabled'
AXIAL_WALL_PARAM_PREFIX = 'lidar_progress_' + 'axial_wall'
LONGITUDINAL_WALL_SNAPSHOT = 'Longitudinal' + 'WallSnapshot'
LONGITUDINAL_WALL_PREFIX = 'longitudinal' + '_wall'


def test_control_params_default_to_lidar_required():
    text = (PACKAGE_ROOT / 'config' / 'control_params.yaml').read_text()

    assert 'translation_progress_source: lidar_required' in text
    assert 'front_stop_distance_m: 0.09' in text
    assert 'translation_lidar_required_invalid_max_consecutive_samples: 4' in text
    assert 'lidar_progress_max_disagreement_m: 0.035' in text
    assert 'lidar_progress_max_ahead_of_odom_m: 0.075' in text
    assert 'lidar_odom_warning_threshold_m: 0.030' in text
    assert 'lidar_progress_temporal_filter_enabled: true' in text
    assert 'lidar_progress_temporal_max_jump_m: 0.080' in text
    assert 'lidar_progress_temporal_max_degraded_samples: 6' in text
    assert 'rotate_timeout_accept_heading_error_rad: 0.090' in text
    assert 'grid_cell_settle_enabled: true' in text
    assert 'grid_cell_settle_abort_position_error_m: 0.120' in text
    assert 'grid_center_expected_front_distance_m: 0.125' in text
    assert 'grid_center_expected_rear_distance_m: 0.125' in text
    assert 'translation_lidar_required_invalid_grace_s' not in text
    assert GEOMETRY_VALIDATION_PARAM not in text
    assert AXIAL_WALL_PARAM_PREFIX not in text
    assert 'It must not complete a drive when LiDAR progress is missing' in text
    assert 'grid_live_yaw_min_confidence' not in text


def test_control_launch_uses_yaml_as_single_default_source():
    text = (PACKAGE_ROOT / 'launch' / 'control.launch.py').read_text()

    assert "'params_file'" in text
    assert "'control_params.yaml'" in text
    assert 'parameters=[params_file]' in text
    assert 'ParameterValue' not in text
    assert 'translation_progress_source' not in text
    assert 'default_linear_speed_mps' not in text
    assert 'timeout_margin_sec' not in text
    assert 'translation_lidar_required_invalid_grace_s' not in text


def test_legacy_yaw_bloat_removed_from_control_node():
    text = (PACKAGE_ROOT / 'darth_maul_control' / 'control_node.py').read_text()

    forbidden = [
        '_update_lidar_parallelity_tracker',
        '_reset_lidar_parallelity_tracker',
        '_begin_grid_yaw_control_memory',
        '_end_grid_yaw_control_memory',
        '_grid_yaw_control_memory_accepting',
        '_post_rotation_refine_correction',
        '_refine_post_rotation_grid_yaw',
        'pre_translation_grid_yaw_align_enabled',
        'lidar_parallelity_control_enabled',
        'post_rotation_grid_yaw_refine_enabled',
        'DRIVE_FORWARD does not use one-frame grid yaw',
    ]

    for token in forbidden:
        assert token not in text


def test_live_grid_control_defaults_present():
    text = (PACKAGE_ROOT / 'config' / 'control_params.yaml').read_text()

    assert 'translation_progress_source: lidar_required' in text
    assert 'grid_live_control_enabled: true' in text
    assert 'k_grid_lateral: 1.60' in text
    assert 'max_linear_x_mps: 0.150' in text
    assert 'max_grid_lateral_mps: 0.045' in text
    assert 'grid_manhattan_yaw_enabled: true' in text
    assert 'k_grid_live_yaw: 2.40' in text
    assert 'max_grid_live_yaw_correction_radps: 0.140' in text
    assert 'grid_virtual_cell_boundary_margin_m: 0.025' in text
    assert 'grid_reacquire_large_yaw_rad: 0.200' in text
    assert 'grid_cell_settle_enabled: true' in text
    assert 'grid_cell_settle_timeout_sec: 2.0' in text
    assert 'grid_cell_settle_position_tolerance_m: 0.025' in text
    assert 'grid_center_front_rear_agreement_tolerance_m: 0.035' in text
    assert 'lidar_parallelity_' not in text
    assert 'pre_translation_grid_yaw_align_' not in text
    assert 'post_rotation_grid_yaw_refine_' not in text


def test_manhattan_yaw_defaults_present():
    text = (PACKAGE_ROOT / 'config' / 'control_params.yaml').read_text()

    assert 'grid_manhattan_yaw_enabled: true' in text
    assert 'grid_manhattan_yaw_min_segment_length_m: 0.120' in text
    assert 'grid_manhattan_yaw_max_line_rms_m: 0.020' in text
    assert 'grid_manhattan_yaw_min_concentration: 0.70' in text
    assert 'grid_manhattan_yaw_max_abs_error_rad: 0.140' in text


def test_control_node_splits_yaw_and_lateral_observations():
    text = (PACKAGE_ROOT / 'darth_maul_control' / 'control_node.py').read_text()

    assert 'estimate_manhattan_grid_yaw(' in text
    assert 'observe_centering_from_expected_side_walls(' in text
    assert 'observation.yaw.valid' in text
    assert 'observation.centering.valid' in text
    assert 'cmd.linear.y = live_command.linear_y_mps' in text


def test_control_node_has_cell_settle_phase():
    text = (PACKAGE_ROOT / 'darth_maul_control' / 'control_node.py').read_text()

    assert '_settle_grid_cell_after_translation(' in text
    assert 'choose_axial_cell_centering(' in text
    assert 'grid_cell_settle_abort_position_error_m' in text
    assert 'front_rear_rejected' in text


def test_control_node_uses_live_lateral_command_not_hardcoded_zero():
    text = (PACKAGE_ROOT / 'darth_maul_control' / 'control_node.py').read_text()

    assert 'cmd.linear.y = live_command.linear_y_mps' in text
    assert 'cmd.linear.y = 0.0' in text
    assert 'live_command.lateral_active' in text


def test_control_node_has_no_4g_geometry_validation_runtime_path():
    text = (PACKAGE_ROOT / 'darth_maul_control' / 'control_node.py').read_text()

    assert GEOMETRY_VALIDATION_PARAM not in text
    assert LONGITUDINAL_WALL_SNAPSHOT not in text
    assert LONGITUDINAL_WALL_PREFIX not in text
