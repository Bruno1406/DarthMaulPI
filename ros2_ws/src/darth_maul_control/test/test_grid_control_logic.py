import pytest

from darth_maul_control.scan_geometry import (
    LidarProgressEstimate,
    choose_heading_validation_error,
    choose_lidar_progress,
    choose_translation_progress,
    grid_lateral_drift,
    grid_yaw_correction_radps,
    rotation_timeout_accepts_heading_error,
)


def test_raw_lidar_progress_accepts_consistent_three_cell_measurement():
    lidar = choose_lidar_progress(
        front_valid=True,
        front_progress_m=0.7610,
        rear_valid=True,
        rear_progress_m=0.7630,
        max_disagreement_m=0.025,
        min_progress_m=-0.010,
        allow_single_source=True,
    )

    assert lidar.valid
    assert lidar.source == 'front_rear'
    assert lidar.progress_m == pytest.approx(0.7620)
    assert lidar.disagreement_m == pytest.approx(0.0020)


def test_raw_lidar_progress_rejects_side_wall_alias_by_front_rear_disagreement():
    estimate = choose_lidar_progress(
        front_valid=True,
        front_progress_m=0.5065,
        rear_valid=True,
        rear_progress_m=0.0140,
        max_disagreement_m=0.025,
        min_progress_m=-0.010,
        allow_single_source=True,
    )

    assert not estimate.valid
    assert estimate.source == 'front_rear_rejected'
    assert estimate.disagreement_m == pytest.approx(0.4925)
    assert 'disagreement' in estimate.reason


def test_raw_lidar_progress_allows_single_source_when_configured():
    lidar = choose_lidar_progress(
        front_valid=False,
        front_progress_m=0.0,
        rear_valid=True,
        rear_progress_m=0.757,
        max_disagreement_m=0.025,
        min_progress_m=-0.010,
        allow_single_source=True,
    )

    assert lidar.valid
    assert lidar.source == 'rear'
    assert lidar.progress_m == pytest.approx(0.757)


def test_raw_lidar_progress_rejects_single_source_if_disabled():
    lidar = choose_lidar_progress(
        front_valid=False,
        front_progress_m=0.0,
        rear_valid=True,
        rear_progress_m=0.757,
        max_disagreement_m=0.025,
        min_progress_m=-0.010,
        allow_single_source=False,
    )

    assert not lidar.valid
    assert lidar.source == 'none'
    assert 'single-source LiDAR progress disabled' in lidar.reason


def test_lidar_required_rejects_invalid_lidar_even_when_odom_progress_exists():
    lidar = LidarProgressEstimate(
        valid=False,
        progress_m=0.0,
        source='front_rear_rejected',
        disagreement_m=0.49,
        reason='front/rear progress disagreement',
    )

    selection = choose_translation_progress(
        odom_progress_m=0.25,
        lidar_estimate=lidar,
        mode='lidar_required',
        max_lidar_ahead_of_odom_m=0.060,
    )

    assert not selection.valid
    assert selection.source == 'lidar_required_unavailable'
    assert 'odom progress 0.250 m ignored' in selection.reason


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


def test_rotation_timeout_accepts_near_target_error():
    assert rotation_timeout_accepts_heading_error(
        final_heading_error_rad=0.084,
        accept_threshold_rad=0.090,
    )


def test_rotation_timeout_rejects_large_residual_error():
    assert not rotation_timeout_accepts_heading_error(
        final_heading_error_rad=0.120,
        accept_threshold_rad=0.090,
    )


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
