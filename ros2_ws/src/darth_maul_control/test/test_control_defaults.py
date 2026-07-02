from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_VALIDATION_PARAM = 'lidar_progress_geometry_validation' + '_enabled'
AXIAL_WALL_PARAM_PREFIX = 'lidar_progress_' + 'axial_wall'
LONGITUDINAL_WALL_SNAPSHOT = 'Longitudinal' + 'WallSnapshot'
LONGITUDINAL_WALL_PREFIX = 'longitudinal' + '_wall'


def test_control_params_default_to_lidar_required():
    text = (PACKAGE_ROOT / 'config' / 'control_params.yaml').read_text()

    assert 'translation_progress_source: lidar_required' in text
    assert 'grid_alignment_control_enabled: true' in text
    assert 'grid_yaw_correction_enabled: true' in text
    assert 'pre_translation_grid_yaw_align_enabled: false' in text
    assert 'front_stop_distance_m: 0.08' in text
    assert GEOMETRY_VALIDATION_PARAM not in text
    assert AXIAL_WALL_PARAM_PREFIX not in text
    assert 'pre_translation_grid_yaw_align_start_threshold_rad: 0.035' in text
    assert 'pre_translation_grid_yaw_align_target_rad: 0.015' in text
    assert 'pre_translation_grid_yaw_align_stable_samples: 3' in text
    assert 'pre_translation_grid_yaw_align_max_invalid_samples: 3' in text
    assert 'translation_lidar_required_invalid_max_consecutive_samples: 2' in text
    assert 'post_rotation_grid_yaw_refine_enabled: false' in text
    assert 'post_rotation_grid_yaw_refine_start_threshold_rad: 0.030' in text
    assert 'post_rotation_grid_yaw_refine_target_rad: 0.015' in text
    assert 'post_rotation_grid_yaw_refine_stable_samples: 3' in text
    assert 'post_rotation_grid_yaw_refine_timeout_s: 2.0' in text
    assert 'post_rotation_grid_yaw_refine_max_invalid_samples: 8' in text
    assert 'post_rotation_grid_yaw_refine_max_abs_error_rad: 0.20' in text
    assert 'post_rotation_grid_yaw_refine_min_confidence: 0.60' in text
    assert 'post_rotation_grid_yaw_refine_kp: 1.00' in text
    assert 'post_rotation_grid_yaw_refine_max_angular_z_radps: 0.080' in text
    assert 'post_rotation_grid_yaw_refine_require_valid: false' in text
    assert 'grid_yaw_control_require_strong_evidence: true' in text
    assert 'grid_yaw_control_allow_single_wall: true' in text
    assert 'grid_yaw_control_single_wall_min_confidence: 0.60' in text
    assert 'grid_yaw_control_single_wall_max_abs_yaw_error_rad: 0.100' in text
    assert 'grid_yaw_control_single_wall_max_rms_error_m: 0.020' in text
    assert 'grid_yaw_control_single_wall_min_span_x_m: 0.220' in text
    assert 'grid_yaw_control_single_wall_min_support_count: 80' in text
    assert 'lidar_parallelity_control_enabled: true' in text
    assert 'lidar_parallelity_heading_hold_scale: 1.0' in text
    assert 'lidar_parallelity_min_progress_for_control_m: 0.040' in text
    assert 'lidar_parallelity_min_stable_samples: 3' in text
    assert 'lidar_parallelity_max_yaw_jump_rad: 0.025' in text
    assert 'lidar_parallelity_max_offset_jump_m: 0.035' in text
    assert 'lidar_parallelity_max_abs_yaw_error_rad: 0.100' in text
    assert 'lidar_parallelity_min_confidence: 0.60' in text
    assert 'lidar_parallelity_max_rms_error_m: 0.020' in text
    assert 'lidar_parallelity_min_span_x_m: 0.220' in text
    assert 'lidar_parallelity_min_support_count: 80' in text
    assert 'lidar_parallelity_drift_validation_enabled: true' in text
    assert 'k_lidar_parallelity_yaw: 0.80' in text
    assert 'k_lidar_parallelity_drift: 0.20' in text
    assert 'max_lidar_parallelity_correction_radps: 0.040' in text
    assert 'lidar_progress_temporal_filter_enabled: true' in text
    assert 'translation_lidar_required_invalid_grace_s' not in text
    assert 'Odom is diagnostic only' in text


def test_control_launch_defaults_for_exam_mode():
    text = (PACKAGE_ROOT / 'launch' / 'control.launch.py').read_text()

    assert "'translation_progress_source'" in text
    assert "default_value='lidar_required'" in text
    assert "'translation_progress_source': ParameterValue(" in text
    assert "'grid_alignment_control_enabled'" in text
    assert "'grid_yaw_correction_enabled'" in text
    assert "'pre_translation_grid_yaw_align_enabled'" in text
    assert "default_value='false'" in text
    assert GEOMETRY_VALIDATION_PARAM not in text
    assert AXIAL_WALL_PARAM_PREFIX not in text
    assert "'pre_translation_grid_yaw_align_start_threshold_rad'" in text
    assert "default_value='0.035'" in text
    assert "'pre_translation_grid_yaw_align_target_rad'" in text
    assert "default_value='0.015'" in text
    assert "'pre_translation_grid_yaw_align_stable_samples'" in text
    assert "default_value='3'" in text
    assert "'pre_translation_grid_yaw_align_max_invalid_samples'" in text
    assert "'translation_lidar_required_invalid_max_consecutive_samples'" in text
    assert "default_value='2'" in text
    assert "'post_rotation_grid_yaw_refine_enabled'" in text
    assert "'post_rotation_grid_yaw_refine_start_threshold_rad'" in text
    assert "'post_rotation_grid_yaw_refine_target_rad'" in text
    assert "'post_rotation_grid_yaw_refine_stable_samples'" in text
    assert "'post_rotation_grid_yaw_refine_timeout_s'" in text
    assert "'post_rotation_grid_yaw_refine_max_invalid_samples'" in text
    assert "'post_rotation_grid_yaw_refine_max_abs_error_rad'" in text
    assert "'post_rotation_grid_yaw_refine_min_confidence'" in text
    assert "'post_rotation_grid_yaw_refine_kp'" in text
    assert "'post_rotation_grid_yaw_refine_max_angular_z_radps'" in text
    assert "'post_rotation_grid_yaw_refine_require_valid'" in text
    assert "'grid_yaw_control_require_strong_evidence'" in text
    assert "'grid_yaw_control_allow_single_wall'" in text
    assert "'grid_yaw_control_single_wall_min_confidence'" in text
    assert "'grid_yaw_control_single_wall_max_abs_yaw_error_rad'" in text
    assert "'grid_yaw_control_single_wall_max_rms_error_m'" in text
    assert "'grid_yaw_control_single_wall_min_span_x_m'" in text
    assert "'grid_yaw_control_single_wall_min_support_count'" in text
    assert "'lidar_parallelity_control_enabled'" in text
    assert "'lidar_parallelity_heading_hold_scale'" in text
    assert "'lidar_parallelity_min_progress_for_control_m'" in text
    assert "'lidar_parallelity_min_stable_samples'" in text
    assert "'lidar_parallelity_max_yaw_jump_rad'" in text
    assert "'lidar_parallelity_max_offset_jump_m'" in text
    assert "'lidar_parallelity_max_abs_yaw_error_rad'" in text
    assert "'lidar_parallelity_min_confidence'" in text
    assert "'lidar_parallelity_max_rms_error_m'" in text
    assert "'lidar_parallelity_min_span_x_m'" in text
    assert "'lidar_parallelity_min_support_count'" in text
    assert "'lidar_progress_temporal_filter_enabled'" in text
    assert 'translation_lidar_required_invalid_grace_s' not in text


def test_pre_align_skips_when_grid_yaw_correction_disabled():
    text = (
        PACKAGE_ROOT
        / 'darth_maul_control'
        / 'control_node.py'
    ).read_text()

    assert 'not self.grid_alignment_control_enabled' in text
    assert 'not self.grid_yaw_correction_enabled' in text
    assert "pre_align_reason = 'grid yaw correction disabled'" in text


def test_control_node_defaults_to_grid_yaw_correction_enabled():
    text = (
        PACKAGE_ROOT
        / 'darth_maul_control'
        / 'control_node.py'
    ).read_text()

    assert "self._bool_param(\n            'grid_alignment_control_enabled',\n            True" in text
    assert "self._bool_param(\n            'grid_yaw_correction_enabled',\n            True" in text
    assert 'grid_yaw_control_require_strong_evidence' in text
    assert 'grid_yaw_control_allow_single_wall' in text
    assert 'def _grid_yaw_has_strong_control_evidence(' in text
    assert 'def _grid_yaw_single_wall_quality_ok(' in text
    assert 'grid yaw diagnostic only:' in text
    assert "self._bool_param(\n            'pre_translation_grid_yaw_align_enabled',\n            False" in text
    assert "self._bool_param(\n            'post_rotation_grid_yaw_refine_enabled',\n            False" in text
    assert "self._bool_param(\n            'post_rotation_grid_yaw_refine_require_valid',\n            False" in text


def test_drive_forward_uses_lidar_parallelity_observer_not_direct_grid_yaw():
    text = (
        PACKAGE_ROOT
        / 'darth_maul_control'
        / 'control_node.py'
    ).read_text()

    assert 'def _update_lidar_parallelity_tracker(' in text
    assert '_update_lidar_parallelity_tracker(' in text
    assert 'DRIVE_FORWARD does not use one-frame grid yaw' in text


def test_parallelity_drift_uses_tracking_progress_baseline():
    text = (
        PACKAGE_ROOT
        / 'darth_maul_control'
        / 'control_node.py'
    ).read_text()

    assert '_parallel_wall_start_progress_m = 0.0' in text
    assert 'control_progress_m - self._parallel_wall_start_progress_m' in text
    assert 'drift_per_m = offset_drift_m / tracking_progress_m' in text
    assert 'offset_drift_m / max(control_progress_m' not in text

    baseline_progress_m = 0.20
    current_progress_m = 0.25
    offset_drift_m = 0.01
    tracking_progress_m = current_progress_m - baseline_progress_m

    assert offset_drift_m / tracking_progress_m == pytest.approx(0.20)
    assert offset_drift_m / current_progress_m == pytest.approx(0.04)


def test_control_node_has_post_rotation_refine_hook_in_rotate_only():
    text = (
        PACKAGE_ROOT
        / 'darth_maul_control'
        / 'control_node.py'
    ).read_text()

    assert 'def _refine_post_rotation_grid_yaw(' in text
    assert 'def _post_rotation_refine_correction(' in text
    assert 'ROTATE_RELATIVE_POST_GRID_REFINE' in text
    assert '_refine_post_rotation_grid_yaw(' in text
    assert 'post_rotation_refine=' in text


def test_control_node_has_no_4g_geometry_validation_runtime_path():
    text = (
        PACKAGE_ROOT
        / 'darth_maul_control'
        / 'control_node.py'
    ).read_text()

    assert GEOMETRY_VALIDATION_PARAM not in text
    assert LONGITUDINAL_WALL_SNAPSHOT not in text
    assert LONGITUDINAL_WALL_PREFIX not in text
