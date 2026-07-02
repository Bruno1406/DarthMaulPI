import pytest

from darth_maul_control.scan_geometry import (
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
