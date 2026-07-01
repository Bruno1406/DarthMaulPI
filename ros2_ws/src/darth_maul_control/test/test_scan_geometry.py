from types import SimpleNamespace
import math

import pytest

from darth_maul_control.scan_geometry import (
    cardinal_sector_ranges,
    combine_lidar_progress_candidates,
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


def test_combine_lidar_progress_averages_front_and_rear():
    valid, progress = combine_lidar_progress_candidates(True, 0.220, True, 0.250)

    assert valid
    assert progress == pytest.approx(0.235)


def test_combine_lidar_progress_uses_single_valid_candidate():
    front_valid, front_progress = combine_lidar_progress_candidates(True, 0.220, False, 0.250)
    rear_valid, rear_progress = combine_lidar_progress_candidates(False, 0.220, True, 0.250)

    assert front_valid
    assert front_progress == pytest.approx(0.220)
    assert rear_valid
    assert rear_progress == pytest.approx(0.250)


def test_combine_lidar_progress_invalid_without_candidates():
    valid, progress = combine_lidar_progress_candidates(False, 0.220, False, 0.250)

    assert not valid
    assert progress == pytest.approx(0.0)
