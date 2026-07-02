import pytest

from darth_maul_control.scan_geometry import (
    compose_angular_command,
    grid_lateral_drift,
    grid_yaw_correction_radps,
)


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
