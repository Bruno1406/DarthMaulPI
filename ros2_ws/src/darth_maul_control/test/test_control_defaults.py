from pathlib import Path


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
    assert GEOMETRY_VALIDATION_PARAM not in text
    assert AXIAL_WALL_PARAM_PREFIX not in text
    assert 'pre_translation_grid_yaw_align_start_threshold_rad: 0.035' in text
    assert 'pre_translation_grid_yaw_align_target_rad: 0.015' in text
    assert 'pre_translation_grid_yaw_align_stable_samples: 3' in text
    assert 'pre_translation_grid_yaw_align_max_invalid_samples: 3' in text
    assert 'translation_lidar_required_invalid_max_consecutive_samples: 2' in text
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
    assert "self._bool_param(\n            'pre_translation_grid_yaw_align_enabled',\n            False" in text


def test_control_node_has_no_4g_geometry_validation_runtime_path():
    text = (
        PACKAGE_ROOT
        / 'darth_maul_control'
        / 'control_node.py'
    ).read_text()

    assert GEOMETRY_VALIDATION_PARAM not in text
    assert LONGITUDINAL_WALL_SNAPSHOT not in text
    assert LONGITUDINAL_WALL_PREFIX not in text
