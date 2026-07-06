import math
from types import SimpleNamespace

import pytest

from darth_maul_control.grid_yaw import (
    estimate_manhattan_grid_yaw,
    estimate_manhattan_grid_yaw_from_segments,
    fit_line_segment_from_xy,
    normalize_manhattan_axis_error,
)


def make_points_on_line(angle_rad, offset_normal_m=0.0, start=-0.30, stop=0.30, n=40):
    ux = math.cos(angle_rad)
    uy = math.sin(angle_rad)
    nx = -uy
    ny = ux
    points = []
    for i in range(n):
        t = start + (stop - start) * i / max(1, n - 1)
        x = t * ux + offset_normal_m * nx
        y = t * uy + offset_normal_m * ny
        points.append((x, y))
    return points


def segment_from_line(angle_rad, offset=0.0, length=0.60, n=40):
    return fit_line_segment_from_xy(
        make_points_on_line(angle_rad, offset, -length / 2.0, length / 2.0, n),
        min_segment_points=8,
        min_segment_length_m=0.12,
        max_line_rms_m=0.020,
    )


def estimate_from_angles(angles):
    segments = [segment_from_line(angle) for angle in angles]
    return estimate_manhattan_grid_yaw_from_segments(
        segments,
        min_line_count=1,
        min_total_weight=0.20,
        min_concentration=0.70,
        max_abs_yaw_error_rad=0.140,
    )


def make_laserscan_from_wall_segments(
    wall_segments,
    *,
    num_samples=720,
    angle_min=-math.pi,
    angle_max=math.pi,
    range_min=0.05,
    range_max=12.0,
):
    """Create a synthetic LaserScan from finite wall segments.

    wall_segments entries:
      ('x', x_value, y_min, y_max) for vertical wall x = constant
      ('y', y_value, x_min, x_max) for horizontal wall y = constant
    """
    angle_increment = (angle_max - angle_min) / num_samples
    ranges = [float('inf')] * num_samples

    for i in range(num_samples):
        theta = angle_min + i * angle_increment
        c = math.cos(theta)
        s = math.sin(theta)
        best = float('inf')

        for kind, value, a_min, a_max in wall_segments:
            if kind == 'x':
                if abs(c) < 1.0e-9:
                    continue
                r = value / c
                other = r * s
            elif kind == 'y':
                if abs(s) < 1.0e-9:
                    continue
                r = value / s
                other = r * c
            else:
                raise ValueError(kind)

            if not math.isfinite(r) or r < range_min or r > range_max:
                continue
            if a_min <= other <= a_max:
                best = min(best, r)

        ranges[i] = best

    return SimpleNamespace(
        angle_min=angle_min,
        angle_increment=angle_increment,
        range_min=range_min,
        range_max=range_max,
        ranges=ranges,
    )


def estimate_yaw_from_scan(scan):
    return estimate_manhattan_grid_yaw(
        scan,
        min_range_m=0.08,
        max_range_m=2.50,
        max_point_gap_m=0.055,
        max_range_jump_m=0.080,
        min_cluster_points=8,
        min_segment_points=8,
        min_segment_length_m=0.120,
        max_line_rms_m=0.020,
        min_line_count=1,
        min_total_weight=0.20,
        min_concentration=0.70,
        max_abs_yaw_error_rad=0.140,
    )


def test_normalize_manhattan_axis_error():
    assert normalize_manhattan_axis_error(0.0) == pytest.approx(0.0)
    assert normalize_manhattan_axis_error(math.pi / 2.0) == pytest.approx(0.0)
    assert normalize_manhattan_axis_error(0.10) == pytest.approx(0.10)
    assert normalize_manhattan_axis_error(math.pi / 2.0 + 0.10) == pytest.approx(0.10)
    assert normalize_manhattan_axis_error(math.pi / 2.0 - 0.10) == pytest.approx(-0.10)


def test_horizontal_wall_line_gives_zero_yaw():
    obs = estimate_from_angles([0.0])
    assert obs.valid
    assert obs.yaw_error_rad == pytest.approx(0.0, abs=0.01)


def test_front_wall_vertical_line_gives_zero_yaw():
    obs = estimate_from_angles([math.pi / 2.0])
    assert obs.valid
    assert obs.yaw_error_rad == pytest.approx(0.0, abs=0.01)


def test_rotated_front_wall_gives_yaw_error():
    obs = estimate_from_angles([math.pi / 2.0 + 0.08])
    assert obs.valid
    assert obs.yaw_error_rad == pytest.approx(0.08, abs=0.015)


def test_rotated_side_wall_gives_yaw_error():
    obs = estimate_from_angles([0.08])
    assert obs.valid
    assert obs.yaw_error_rad == pytest.approx(0.08, abs=0.015)


def test_orthogonal_lines_average_to_same_yaw():
    obs = estimate_from_angles([0.06, math.pi / 2.0 + 0.06])
    assert obs.valid
    assert obs.yaw_error_rad == pytest.approx(0.06, abs=0.015)
    assert obs.line_count == 2


def test_conflicting_lines_rejected_by_concentration():
    segments = [
        segment_from_line(0.00),
        segment_from_line(0.30),
    ]
    obs = estimate_manhattan_grid_yaw_from_segments(
        segments,
        min_line_count=2,
        min_total_weight=0.20,
        min_concentration=0.95,
        max_abs_yaw_error_rad=0.140,
    )
    assert not obs.valid
    assert 'concentration' in obs.reason


def test_short_segment_rejected_before_yaw_estimation():
    segment = fit_line_segment_from_xy(
        make_points_on_line(0.0, 0.0, -0.03, 0.03, 10),
        min_segment_points=8,
        min_segment_length_m=0.12,
        max_line_rms_m=0.020,
    )
    obs = estimate_manhattan_grid_yaw_from_segments(
        [segment],
        min_line_count=1,
        min_total_weight=0.20,
        min_concentration=0.70,
        max_abs_yaw_error_rad=0.140,
    )
    assert not obs.valid


def test_manhattan_yaw_from_front_wall_laserscan():
    scan = make_laserscan_from_wall_segments([
        ('x', 0.35, -0.30, 0.30),
    ])

    obs = estimate_yaw_from_scan(scan)

    assert obs.valid
    assert obs.yaw_error_rad == pytest.approx(0.0, abs=0.01)
    assert obs.line_count >= 1


def test_manhattan_yaw_from_side_wall_laserscan():
    scan = make_laserscan_from_wall_segments([
        ('y', 0.20, -0.30, 0.45),
    ])

    obs = estimate_yaw_from_scan(scan)

    assert obs.valid
    assert obs.yaw_error_rad == pytest.approx(0.0, abs=0.01)
    assert obs.line_count >= 1


def test_manhattan_yaw_rejects_too_short_laserscan_segment():
    scan = make_laserscan_from_wall_segments([
        ('x', 0.35, -0.03, 0.03),
    ])

    obs = estimate_yaw_from_scan(scan)

    assert not obs.valid


def test_manhattan_yaw_from_front_and_side_laserscan():
    scan = make_laserscan_from_wall_segments([
        ('x', 0.35, -0.30, 0.30),
        ('y', 0.20, -0.30, 0.20),
    ])

    obs = estimate_yaw_from_scan(scan)

    assert obs.valid
    assert obs.yaw_error_rad == pytest.approx(0.0, abs=0.01)
    assert obs.line_count >= 2
