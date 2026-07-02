import pytest

from darth_maul_control.scan_geometry import (
    choose_heading_validation_error,
    choose_lidar_progress,
    choose_translation_progress,
    compose_angular_command,
    grid_yaw_pre_align_complete,
    grid_lateral_drift,
    grid_yaw_correction_radps,
    should_pre_align_grid_yaw,
)


def test_geometry_qualified_front_rear_progress_is_averaged():
    lidar = choose_lidar_progress(
        front_valid=True,
        front_progress_m=1.668 - 0.903,
        rear_valid=True,
        rear_progress_m=0.863 - 0.106,
        max_disagreement_m=0.025,
        min_progress_m=-0.010,
        allow_single_source=True,
        front_reason='front axial wall valid',
        rear_reason='rear axial wall valid',
    )

    assert lidar.valid
    assert lidar.source == 'front_rear'
    assert lidar.disagreement_m == pytest.approx(0.008)
    assert lidar.progress_m == pytest.approx(0.761)


def test_lab_side_wall_alias_uses_rear_only_when_allowed():
    lidar = choose_lidar_progress(
        front_valid=False,
        front_progress_m=0.0,
        rear_valid=True,
        rear_progress_m=0.014,
        max_disagreement_m=0.025,
        min_progress_m=-0.010,
        allow_single_source=True,
        front_reason='front axial wall invalid: y-span too small; side-wall-like geometry',
        rear_reason='rear axial wall valid',
    )

    assert lidar.valid
    assert lidar.source == 'rear'
    assert lidar.progress_m == pytest.approx(0.014)
    assert 'front rejected' in lidar.reason
    assert 'side-wall-like' in lidar.reason


def test_lab_side_wall_alias_never_selects_raw_front_jump():
    lidar = choose_lidar_progress(
        front_valid=False,
        front_progress_m=0.506,
        rear_valid=False,
        rear_progress_m=0.014,
        max_disagreement_m=0.025,
        min_progress_m=-0.010,
        allow_single_source=True,
        front_reason='front axial wall invalid: y-span too small; side-wall-like geometry',
        rear_reason='rear axial wall invalid: not enough candidate points',
    )

    selection = choose_translation_progress(
        odom_progress_m=0.029,
        lidar_estimate=lidar,
        mode='lidar_required',
        max_lidar_ahead_of_odom_m=0.060,
    )

    assert not lidar.valid
    assert lidar.progress_m == pytest.approx(0.0)
    assert not selection.valid
    assert selection.source == 'lidar_required_unavailable'
    assert selection.progress_m == pytest.approx(0.0)
    assert 'no valid LiDAR progress source' in selection.reason
    assert 'side-wall-like' in selection.reason
    assert 'odom progress 0.029 m ignored' in selection.reason


def test_raw_cardinal_progress_fallback_is_explicit_geometry_disabled_path():
    lidar = choose_lidar_progress(
        front_valid=True,
        front_progress_m=0.506,
        rear_valid=True,
        rear_progress_m=0.500,
        max_disagreement_m=0.025,
        min_progress_m=-0.010,
        allow_single_source=True,
        front_reason='raw cardinal front range valid',
        rear_reason='raw cardinal rear range valid',
    )

    assert lidar.valid
    assert lidar.source == 'front_rear'
    assert lidar.progress_m == pytest.approx(0.503)


def test_lidar_required_invalid_geometry_never_falls_back_to_odom():
    lidar = choose_lidar_progress(
        front_valid=False,
        front_progress_m=0.0,
        rear_valid=False,
        rear_progress_m=0.0,
        max_disagreement_m=0.025,
        min_progress_m=-0.010,
        allow_single_source=True,
        front_reason='front axial wall invalid',
        rear_reason='rear axial wall invalid',
    )

    selection = choose_translation_progress(
        odom_progress_m=0.25,
        lidar_estimate=lidar,
        mode='lidar_required',
        max_lidar_ahead_of_odom_m=0.060,
    )

    assert not selection.valid
    assert selection.source == 'lidar_required_unavailable'
    assert selection.progress_m == pytest.approx(0.0)
    assert 'odom progress 0.250 m ignored' in selection.reason


def test_geometry_single_source_rear_allowed():
    lidar = choose_lidar_progress(
        front_valid=False,
        front_progress_m=0.0,
        rear_valid=True,
        rear_progress_m=0.757,
        max_disagreement_m=0.025,
        min_progress_m=-0.010,
        allow_single_source=True,
        front_reason='front axial wall invalid',
        rear_reason='rear axial wall valid',
    )

    assert lidar.valid
    assert lidar.source == 'rear'
    assert lidar.progress_m == pytest.approx(0.757)


def test_geometry_single_source_disabled_rejects_rear_only():
    lidar = choose_lidar_progress(
        front_valid=False,
        front_progress_m=0.0,
        rear_valid=True,
        rear_progress_m=0.757,
        max_disagreement_m=0.025,
        min_progress_m=-0.010,
        allow_single_source=False,
        front_reason='front axial wall invalid',
        rear_reason='rear axial wall valid',
    )

    assert not lidar.valid
    assert lidar.source == 'none'
    assert 'single-source LiDAR progress disabled' in lidar.reason


def test_lateral_drift_computes_signed_drift_per_meter():
    estimate = grid_lateral_drift(
        start_valid=True,
        start_error_m=0.000,
        end_valid=True,
        end_error_m=-0.020,
        progress_m=0.762,
    )

    assert estimate.drift_valid
    assert estimate.drift_m == pytest.approx(-0.020)
    assert estimate.drift_per_m == pytest.approx(-0.0262, abs=0.0001)


def test_lateral_drift_rejects_invalid_start():
    estimate = grid_lateral_drift(
        start_valid=False,
        start_error_m=0.000,
        end_valid=True,
        end_error_m=-0.020,
        progress_m=0.762,
    )

    assert not estimate.start_valid
    assert estimate.end_valid
    assert not estimate.drift_valid
    assert estimate.drift_m == pytest.approx(0.0)


def test_lateral_drift_rejects_invalid_end():
    estimate = grid_lateral_drift(
        start_valid=True,
        start_error_m=0.000,
        end_valid=False,
        end_error_m=-0.020,
        progress_m=0.762,
    )

    assert estimate.start_valid
    assert not estimate.end_valid
    assert not estimate.drift_valid
    assert estimate.drift_m == pytest.approx(0.0)


def test_grid_yaw_correction_uses_positive_gain_sign():
    correction = grid_yaw_correction_radps(
        yaw_error_rad=-0.05,
        k_yaw=0.70,
        max_correction_radps=0.045,
    )

    assert correction == pytest.approx(-0.035)


def test_grid_yaw_correction_is_capped():
    correction = grid_yaw_correction_radps(
        yaw_error_rad=0.20,
        k_yaw=0.70,
        max_correction_radps=0.045,
    )

    assert correction == pytest.approx(0.045)


def test_compose_angular_command_uses_heading_hold_when_grid_inactive():
    assert compose_angular_command(
        heading_correction_radps=0.04,
        grid_yaw_correction_radps=0.0,
        grid_yaw_active=False,
        grid_yaw_active_heading_hold_scale=0.0,
    ) == pytest.approx(0.04)


def test_compose_angular_command_grid_active_disables_heading_hold_by_default():
    assert compose_angular_command(
        heading_correction_radps=0.04,
        grid_yaw_correction_radps=-0.06,
        grid_yaw_active=True,
        grid_yaw_active_heading_hold_scale=0.0,
    ) == pytest.approx(-0.06)


def test_compose_angular_command_grid_active_can_blend_heading_hold():
    assert compose_angular_command(
        heading_correction_radps=0.04,
        grid_yaw_correction_radps=-0.06,
        grid_yaw_active=True,
        grid_yaw_active_heading_hold_scale=0.25,
    ) == pytest.approx(-0.05)


def test_compose_angular_command_clamps_heading_hold_scale():
    assert compose_angular_command(
        heading_correction_radps=0.04,
        grid_yaw_correction_radps=-0.06,
        grid_yaw_active=True,
        grid_yaw_active_heading_hold_scale=2.0,
    ) == pytest.approx(-0.02)


def test_heading_validation_uses_odom_when_grid_correction_unused():
    error, source = choose_heading_validation_error(
        odom_heading_error_rad=0.073,
        grid_yaw_error_rad=0.001,
        grid_yaw_valid=True,
        grid_yaw_correction_used=False,
        grid_alignment_confidence=0.6,
        min_grid_confidence=0.5,
        max_grid_yaw_abs_error_rad=0.10,
        grid_alignment_source='right',
    )

    assert error == pytest.approx(0.073)
    assert source == 'odom'


def test_heading_validation_uses_valid_confident_grid_yaw():
    error, source = choose_heading_validation_error(
        odom_heading_error_rad=0.073,
        grid_yaw_error_rad=-0.001,
        grid_yaw_valid=True,
        grid_yaw_correction_used=True,
        grid_alignment_confidence=0.6,
        min_grid_confidence=0.5,
        max_grid_yaw_abs_error_rad=0.10,
        grid_alignment_source='right',
    )

    assert error == pytest.approx(0.001)
    assert source == 'grid_yaw/right'


def test_heading_validation_uses_odom_when_grid_confidence_low():
    error, source = choose_heading_validation_error(
        odom_heading_error_rad=0.073,
        grid_yaw_error_rad=0.001,
        grid_yaw_valid=True,
        grid_yaw_correction_used=True,
        grid_alignment_confidence=0.4,
        min_grid_confidence=0.5,
        max_grid_yaw_abs_error_rad=0.10,
        grid_alignment_source='right',
    )

    assert error == pytest.approx(0.073)
    assert source == 'odom'


def test_heading_validation_uses_odom_when_grid_invalid():
    error, source = choose_heading_validation_error(
        odom_heading_error_rad=0.073,
        grid_yaw_error_rad=0.001,
        grid_yaw_valid=False,
        grid_yaw_correction_used=True,
        grid_alignment_confidence=0.6,
        min_grid_confidence=0.5,
        max_grid_yaw_abs_error_rad=0.10,
        grid_alignment_source='right',
    )

    assert error == pytest.approx(0.073)
    assert source == 'odom'


def test_heading_validation_uses_odom_when_grid_yaw_too_large():
    error, source = choose_heading_validation_error(
        odom_heading_error_rad=0.073,
        grid_yaw_error_rad=0.12,
        grid_yaw_valid=True,
        grid_yaw_correction_used=True,
        grid_alignment_confidence=0.6,
        min_grid_confidence=0.5,
        max_grid_yaw_abs_error_rad=0.10,
        grid_alignment_source='right',
    )

    assert error == pytest.approx(0.073)
    assert source == 'odom'


def test_should_pre_align_grid_yaw_accepts_lab_failure_case():
    should_align, reason = should_pre_align_grid_yaw(
        enabled=True,
        direction=1.0,
        alignment_valid=True,
        yaw_valid=True,
        yaw_error_rad=-0.105,
        confidence=0.6,
        min_confidence=0.6,
        start_threshold_rad=0.035,
        max_control_error_rad=0.20,
    )

    assert should_align
    assert 'required' in reason


def test_should_pre_align_grid_yaw_skips_small_start_error():
    should_align, reason = should_pre_align_grid_yaw(
        enabled=True,
        direction=1.0,
        alignment_valid=True,
        yaw_valid=True,
        yaw_error_rad=-0.002,
        confidence=0.6,
        min_confidence=0.6,
        start_threshold_rad=0.035,
        max_control_error_rad=0.20,
    )

    assert not should_align
    assert 'within threshold' in reason


def test_should_pre_align_grid_yaw_skips_reverse_translation():
    should_align, reason = should_pre_align_grid_yaw(
        enabled=True,
        direction=-1.0,
        alignment_valid=True,
        yaw_valid=True,
        yaw_error_rad=-0.105,
        confidence=0.6,
        min_confidence=0.6,
        start_threshold_rad=0.035,
        max_control_error_rad=0.20,
    )

    assert not should_align
    assert 'reverse translation' in reason


def test_should_pre_align_grid_yaw_skips_invalid_yaw():
    should_align, reason = should_pre_align_grid_yaw(
        enabled=True,
        direction=1.0,
        alignment_valid=False,
        yaw_valid=False,
        yaw_error_rad=0.0,
        confidence=0.0,
        min_confidence=0.6,
        start_threshold_rad=0.035,
        max_control_error_rad=0.20,
    )

    assert not should_align
    assert 'no valid grid yaw' in reason


def test_should_pre_align_grid_yaw_rejects_low_confidence():
    should_align, reason = should_pre_align_grid_yaw(
        enabled=True,
        direction=1.0,
        alignment_valid=True,
        yaw_valid=True,
        yaw_error_rad=-0.105,
        confidence=0.5,
        min_confidence=0.6,
        start_threshold_rad=0.035,
        max_control_error_rad=0.20,
    )

    assert not should_align
    assert 'confidence too low' in reason


def test_should_pre_align_grid_yaw_rejects_error_above_control_limit():
    should_align, reason = should_pre_align_grid_yaw(
        enabled=True,
        direction=1.0,
        alignment_valid=True,
        yaw_valid=True,
        yaw_error_rad=0.25,
        confidence=0.6,
        min_confidence=0.6,
        start_threshold_rad=0.035,
        max_control_error_rad=0.20,
    )

    assert not should_align
    assert 'too large' in reason


def test_grid_yaw_pre_align_complete_accepts_valid_target_sample():
    complete, reason = grid_yaw_pre_align_complete(
        alignment_valid=True,
        yaw_valid=True,
        yaw_error_rad=-0.012,
        confidence=0.6,
        min_confidence=0.6,
        target_rad=0.015,
    )

    assert complete
    assert 'within target' in reason


@pytest.mark.parametrize(
    'alignment_valid,yaw_valid,yaw_error_rad,confidence,reason_text',
    [
        (False, False, 0.0, 0.0, 'no valid grid yaw'),
        (True, True, 0.0, 0.5, 'confidence too low'),
        (True, True, 0.030, 0.6, 'outside target'),
    ],
)
def test_grid_yaw_pre_align_complete_rejects_unusable_samples(
    alignment_valid,
    yaw_valid,
    yaw_error_rad,
    confidence,
    reason_text,
):
    complete, reason = grid_yaw_pre_align_complete(
        alignment_valid=alignment_valid,
        yaw_valid=yaw_valid,
        yaw_error_rad=yaw_error_rad,
        confidence=confidence,
        min_confidence=0.6,
        target_rad=0.015,
    )

    assert not complete
    assert reason_text in reason
