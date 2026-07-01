from types import SimpleNamespace
import math

import pytest

from darth_maul_control.scan_geometry import (
    cardinal_sector_ranges,
    choose_lidar_progress,
    choose_translation_progress,
    sector_range,
    valid_ranges_in_sector,
)


def make_scan(ranges, angle_min=-math.pi, angle_increment=None):
    if angle_increment is None:
        angle_increment = (2.0 * math.pi) / len(ranges)
    return SimpleNamespace(
        angle_min=angle_min,
        angle_increment=angle_increment,
        range_min=0.05,
        range_max=12.0,
        ranges=ranges,
    )


def test_sector_range_uses_median_not_min():
    scan = make_scan([5.0, 4.0, 1.0, 3.0], angle_increment=math.pi / 2.0)
    measurement = sector_range(scan, 'front', 0.0, width_deg=181.0, min_samples=2)

    assert measurement.valid
    assert valid_ranges_in_sector(scan, 0.0, 181.0) == pytest.approx([4.0, 1.0, 3.0])
    assert measurement.min_m == pytest.approx(1.0)
    assert measurement.median_m == pytest.approx(3.0)


def test_sector_range_rejects_invalid_values():
    scan = make_scan([float('inf'), float('nan'), 0.01, 99.0])
    measurement = sector_range(scan, 'front', 0.0, width_deg=360.0, min_samples=3)

    assert not measurement.valid
    assert measurement.count == 0


def test_cardinal_sector_ranges_return_expected_keys():
    scan = make_scan([1.0] * 360, angle_min=-math.pi, angle_increment=math.radians(1.0))
    measurements = cardinal_sector_ranges(scan, width_deg=10.0, min_samples=3)

    assert set(measurements.keys()) == {'front', 'rear', 'left', 'right'}
    assert all(value.valid for value in measurements.values())


def test_rear_sector_wraparound_is_valid():
    scan = make_scan([2.0] * 360, angle_min=-math.pi, angle_increment=math.radians(1.0))
    measurement = sector_range(scan, 'rear', math.pi, width_deg=10.0, min_samples=3)

    assert measurement.valid
    assert measurement.median_m == pytest.approx(2.0)


def test_choose_lidar_progress_averages_consistent_front_and_rear():
    estimate = choose_lidar_progress(
        True,
        0.213,
        True,
        0.216,
        max_disagreement_m=0.025,
        min_progress_m=-0.010,
        allow_single_source=True,
    )

    assert estimate.valid
    assert estimate.progress_m == pytest.approx(0.2145)
    assert estimate.source == 'front_rear'
    assert estimate.disagreement_m == pytest.approx(0.003)


def test_choose_lidar_progress_rejects_front_rear_disagreement():
    estimate = choose_lidar_progress(
        True,
        0.213,
        True,
        0.270,
        max_disagreement_m=0.025,
        min_progress_m=-0.010,
        allow_single_source=True,
    )

    assert not estimate.valid
    assert estimate.source == 'front_rear_rejected'
    assert estimate.disagreement_m == pytest.approx(0.057)


def test_choose_lidar_progress_uses_front_only_when_allowed():
    estimate = choose_lidar_progress(
        True,
        0.213,
        False,
        0.0,
        max_disagreement_m=0.025,
        min_progress_m=-0.010,
        allow_single_source=True,
    )

    assert estimate.valid
    assert estimate.progress_m == pytest.approx(0.213)
    assert estimate.source == 'front'


def test_choose_lidar_progress_uses_rear_only_when_allowed():
    estimate = choose_lidar_progress(
        False,
        0.0,
        True,
        0.216,
        max_disagreement_m=0.025,
        min_progress_m=-0.010,
        allow_single_source=True,
    )

    assert estimate.valid
    assert estimate.progress_m == pytest.approx(0.216)
    assert estimate.source == 'rear'


def test_choose_lidar_progress_rejects_single_source_when_disabled():
    estimate = choose_lidar_progress(
        True,
        0.213,
        False,
        0.0,
        max_disagreement_m=0.025,
        min_progress_m=-0.010,
        allow_single_source=False,
    )

    assert not estimate.valid
    assert estimate.source == 'none'


def test_choose_lidar_progress_rejects_negative_progress_below_threshold():
    estimate = choose_lidar_progress(
        True,
        -0.030,
        False,
        0.0,
        max_disagreement_m=0.025,
        min_progress_m=-0.010,
        allow_single_source=True,
    )

    assert not estimate.valid
    assert estimate.source == 'none'


def test_choose_translation_progress_uses_lidar_when_consistent():
    lidar = choose_lidar_progress(
        True,
        0.213,
        True,
        0.216,
        max_disagreement_m=0.025,
        min_progress_m=-0.010,
        allow_single_source=True,
    )

    selection = choose_translation_progress(
        odom_progress_m=0.249,
        lidar_estimate=lidar,
        mode='lidar_when_consistent',
        max_lidar_ahead_of_odom_m=0.060,
    )

    assert selection.valid
    assert selection.source == 'lidar'
    assert selection.progress_m == pytest.approx(0.2145)


def test_choose_translation_progress_falls_back_to_odom_when_lidar_invalid():
    lidar = choose_lidar_progress(
        True,
        0.213,
        True,
        0.270,
        max_disagreement_m=0.025,
        min_progress_m=-0.010,
        allow_single_source=True,
    )

    selection = choose_translation_progress(
        odom_progress_m=0.249,
        lidar_estimate=lidar,
        mode='lidar_when_consistent',
        max_lidar_ahead_of_odom_m=0.060,
    )

    assert selection.valid
    assert selection.source == 'odom'
    assert selection.progress_m == pytest.approx(0.249)


def test_choose_translation_progress_rejects_when_lidar_required_and_invalid():
    lidar = choose_lidar_progress(
        True,
        0.213,
        True,
        0.270,
        max_disagreement_m=0.025,
        min_progress_m=-0.010,
        allow_single_source=True,
    )

    selection = choose_translation_progress(
        odom_progress_m=0.249,
        lidar_estimate=lidar,
        mode='lidar_required',
        max_lidar_ahead_of_odom_m=0.060,
    )

    assert not selection.valid
    assert selection.source == 'none'


def test_choose_translation_progress_rejects_lidar_implausibly_ahead_of_odom():
    lidar = choose_lidar_progress(
        True,
        0.400,
        True,
        0.405,
        max_disagreement_m=0.025,
        min_progress_m=-0.010,
        allow_single_source=True,
    )

    selection = choose_translation_progress(
        odom_progress_m=0.250,
        lidar_estimate=lidar,
        mode='lidar_when_consistent',
        max_lidar_ahead_of_odom_m=0.060,
    )

    assert selection.valid
    assert selection.source == 'odom'
    assert selection.progress_m == pytest.approx(0.250)


def test_choose_translation_progress_odom_only_ignores_valid_lidar():
    lidar = choose_lidar_progress(
        True,
        0.213,
        True,
        0.216,
        max_disagreement_m=0.025,
        min_progress_m=-0.010,
        allow_single_source=True,
    )

    selection = choose_translation_progress(
        odom_progress_m=0.249,
        lidar_estimate=lidar,
        mode='odom_only',
        max_lidar_ahead_of_odom_m=0.060,
    )

    assert selection.valid
    assert selection.source == 'odom'
    assert selection.progress_m == pytest.approx(0.249)
