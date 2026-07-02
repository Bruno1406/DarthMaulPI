from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_control_params_default_to_lidar_required():
    text = (PACKAGE_ROOT / 'config' / 'control_params.yaml').read_text()

    assert 'translation_progress_source: lidar_required' in text
    assert 'translation_lidar_required_invalid_grace_s: 0.20' in text
    assert 'Odom is diagnostic only in this mode' in text


def test_control_launch_default_to_lidar_required_with_debug_override():
    text = (PACKAGE_ROOT / 'launch' / 'control.launch.py').read_text()

    assert "'translation_progress_source'" in text
    assert "default_value='lidar_required'" in text
    assert "'translation_progress_source': ParameterValue(" in text
