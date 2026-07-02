from types import SimpleNamespace
import math

import pytest

from darth_maul_control.scan_geometry import (
    GridAlignmentEstimate,
    LidarProgressEstimate,
    WallLineEstimate,
    cardinal_sector_ranges,
    choose_lidar_progress,
    choose_temporal_lidar_progress,
    choose_translation_progress,
    estimate_grid_alignment,
    fit_side_wall_line,
    invalid_grid_alignment,
    invalid_wall_line,
    lidar_parallelity_correction_radps,
    sector_range,
    side_wall_observation_from_alignment,
    side_wall_candidate_points,
    valid_scan_points_xy,
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


def make_line_scan(
    lines,
    num_samples=720,
    angle_min=-math.pi,
    angle_max=math.pi,
    range_min=0.05,
    range_max=12.0,
):
    """Create a synthetic 2D scan from lines y = slope*x + intercept.

    lines: sequence of (slope, intercept, x_min, x_max)
    """
    angle_increment = (angle_max - angle_min) / num_samples
    ranges = [float('inf')] * num_samples

    for i in range(num_samples):
        theta = angle_min + i * angle_increment
        c = math.cos(theta)
        s = math.sin(theta)

        best = float('inf')
        for slope, intercept, x_min, x_max in lines:
            denom = s - slope * c
            if abs(denom) < 1e-9:
                continue

            r = intercept / denom
            if not math.isfinite(r) or r < range_min or r > range_max:
                continue

            x = r * c
            if x_min <= x <= x_max:
                best = min(best, r)

        ranges[i] = best

    return make_scan(
        ranges,
        angle_min=angle_min,
        angle_increment=angle_increment,
    )


def make_points_scan(
    points,
    num_samples=3600,
    angle_min=-math.pi,
    angle_max=math.pi,
    range_min=0.05,
    range_max=12.0,
):
    angle_increment = (angle_max - angle_min) / num_samples
    ranges = [float('inf')] * num_samples

    for x, y in points:
        r = math.hypot(x, y)
        if not math.isfinite(r) or r < range_min or r > range_max:
            continue

        theta = math.atan2(y, x)
        index = int(round((theta - angle_min) / angle_increment))
        if 0 <= index < num_samples:
            ranges[index] = min(ranges[index], r)

    return make_scan(
        ranges,
        angle_min=angle_min,
        angle_increment=angle_increment,
    )


def make_wall(side, offset, yaw=0.02, rms=0.012, span=0.30, count=120):
    return WallLineEstimate(
        valid=True,
        side=side,
        offset_m=offset,
        yaw_error_rad=yaw,
        slope=math.tan(yaw),
        intercept_m=offset,
        support_count=count,
        span_x_m=span,
        rms_error_m=rms,
        reason='test wall',
    )


def make_alignment(
    source='right',
    confidence=0.60,
    left=None,
    right=None,
    yaw=0.02,
):
    left = left if left is not None else invalid_wall_line('left', 'not used')
    right = right if right is not None else invalid_wall_line('right', 'not used')
    return GridAlignmentEstimate(
        valid=True,
        yaw_valid=True,
        yaw_error_rad=yaw,
        lateral_valid=True,
        lateral_error_m=0.0,
        source=source,
        confidence=confidence,
        left=left,
        right=right,
        reason='test alignment',
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


def test_choose_translation_progress_lidar_required_uses_valid_lidar():
    lidar = choose_lidar_progress(
        True,
        0.770,
        True,
        0.754,
        max_disagreement_m=0.025,
        min_progress_m=-0.010,
        allow_single_source=True,
    )

    selection = choose_translation_progress(
        odom_progress_m=0.822,
        lidar_estimate=lidar,
        mode='lidar_required',
        max_lidar_ahead_of_odom_m=0.060,
    )

    assert selection.valid
    assert selection.source == 'lidar'
    assert selection.progress_m == pytest.approx(0.762)


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
    assert 'falling back to odom' in selection.reason


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
    assert selection.source != 'odom'
    assert selection.source == 'lidar_required_unavailable'
    assert selection.progress_m == pytest.approx(0.0)
    assert 'LiDAR progress required' in selection.reason
    assert 'odom progress 0.249 m ignored' in selection.reason


def test_choose_translation_progress_lidar_required_rejects_lab_disagreement():
    lidar = choose_lidar_progress(
        True,
        0.731,
        True,
        0.689,
        max_disagreement_m=0.025,
        min_progress_m=-0.010,
        allow_single_source=True,
    )

    selection = choose_translation_progress(
        odom_progress_m=0.759,
        lidar_estimate=lidar,
        mode='lidar_required',
        max_lidar_ahead_of_odom_m=0.060,
    )

    assert not selection.valid
    assert selection.source == 'lidar_required_unavailable'
    assert selection.source != 'odom'
    assert selection.progress_m == pytest.approx(0.0)
    assert 'front/rear progress disagreement 0.042 m > 0.025 m' in selection.reason
    assert 'odom progress 0.759 m ignored' in selection.reason


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


def test_valid_scan_points_xy_and_side_candidates_use_robot_frame():
    scan = make_line_scan([(0.0, 0.20, -0.20, 0.45)])

    points = valid_scan_points_xy(scan)
    left_candidates = side_wall_candidate_points(
        scan,
        side='left',
        min_x_m=-0.18,
        max_x_m=0.45,
        min_side_distance_m=0.06,
        max_side_distance_m=0.45,
    )

    assert points
    assert len(left_candidates) >= 8
    assert all(y > 0.0 for _, y in left_candidates)


def test_fit_side_wall_line_left_parallel_wall():
    scan = make_line_scan([(0.0, 0.20, -0.20, 0.45)])

    estimate = fit_side_wall_line(
        scan,
        side='left',
        min_x_m=-0.18,
        max_x_m=0.45,
        min_side_distance_m=0.06,
        max_side_distance_m=0.45,
        min_points=8,
        min_span_x_m=0.12,
        max_rms_error_m=0.025,
        max_abs_yaw_error_rad=0.35,
    )

    assert estimate.valid
    assert estimate.offset_m == pytest.approx(0.20, abs=0.01)
    assert estimate.yaw_error_rad == pytest.approx(0.0, abs=0.02)
    assert estimate.support_count >= 8


def test_fit_side_wall_line_right_parallel_wall():
    scan = make_line_scan([(0.0, -0.18, -0.20, 0.45)])

    estimate = fit_side_wall_line(
        scan,
        side='right',
        min_x_m=-0.18,
        max_x_m=0.45,
        min_side_distance_m=0.06,
        max_side_distance_m=0.45,
        min_points=8,
        min_span_x_m=0.12,
        max_rms_error_m=0.025,
        max_abs_yaw_error_rad=0.35,
    )

    assert estimate.valid
    assert estimate.offset_m == pytest.approx(-0.18, abs=0.01)
    assert estimate.yaw_error_rad == pytest.approx(0.0, abs=0.02)


def test_fit_side_wall_line_rejects_far_wall_seen_through_opening():
    scan = make_line_scan([(0.0, 1.20, -0.20, 0.45)])

    estimate = fit_side_wall_line(
        scan,
        side='left',
        min_x_m=-0.18,
        max_x_m=0.45,
        min_side_distance_m=0.06,
        max_side_distance_m=0.45,
        min_points=8,
        min_span_x_m=0.12,
        max_rms_error_m=0.025,
        max_abs_yaw_error_rad=0.35,
    )

    assert not estimate.valid


def test_estimate_grid_alignment_both_walls_centered():
    scan = make_line_scan([
        (0.0, 0.20, -0.20, 0.45),
        (0.0, -0.20, -0.20, 0.45),
    ])

    estimate = estimate_grid_alignment(
        scan,
        expected_half_width_m=0.20,
        min_x_m=-0.18,
        max_x_m=0.45,
        min_side_distance_m=0.06,
        max_side_distance_m=0.45,
        min_points=8,
        min_span_x_m=0.12,
        max_rms_error_m=0.025,
        max_abs_yaw_error_rad=0.35,
        max_reported_error_m=0.30,
        max_reported_yaw_rad=0.50,
    )

    assert estimate.valid
    assert estimate.source == 'left_right'
    assert estimate.yaw_error_rad == pytest.approx(0.0, abs=0.02)
    assert estimate.lateral_error_m == pytest.approx(0.0, abs=0.02)


def test_estimate_grid_alignment_left_only_reports_centerline_left():
    scan = make_line_scan([(0.0, 0.25, -0.20, 0.45)])

    estimate = estimate_grid_alignment(
        scan,
        expected_half_width_m=0.20,
        min_x_m=-0.18,
        max_x_m=0.45,
        min_side_distance_m=0.06,
        max_side_distance_m=0.45,
        min_points=8,
        min_span_x_m=0.12,
        max_rms_error_m=0.025,
        max_abs_yaw_error_rad=0.35,
        max_reported_error_m=0.30,
        max_reported_yaw_rad=0.50,
    )

    assert estimate.valid
    assert estimate.source == 'left'
    assert estimate.lateral_error_m == pytest.approx(0.05, abs=0.02)


def test_estimate_grid_alignment_right_only_reports_centerline_right():
    scan = make_line_scan([(0.0, -0.25, -0.20, 0.45)])

    estimate = estimate_grid_alignment(
        scan,
        expected_half_width_m=0.20,
        min_x_m=-0.18,
        max_x_m=0.45,
        min_side_distance_m=0.06,
        max_side_distance_m=0.45,
        min_points=8,
        min_span_x_m=0.12,
        max_rms_error_m=0.025,
        max_abs_yaw_error_rad=0.35,
        max_reported_error_m=0.30,
        max_reported_yaw_rad=0.50,
    )

    assert estimate.valid
    assert estimate.source == 'right'
    assert estimate.lateral_error_m == pytest.approx(-0.05, abs=0.02)


def test_estimate_grid_alignment_reports_wall_yaw():
    scan = make_line_scan([(0.10, 0.20, -0.20, 0.45)])

    estimate = estimate_grid_alignment(
        scan,
        expected_half_width_m=0.20,
        min_x_m=-0.18,
        max_x_m=0.45,
        min_side_distance_m=0.06,
        max_side_distance_m=0.45,
        min_points=8,
        min_span_x_m=0.12,
        max_rms_error_m=0.025,
        max_abs_yaw_error_rad=0.35,
        max_reported_error_m=0.30,
        max_reported_yaw_rad=0.50,
    )

    assert estimate.valid
    assert estimate.yaw_error_rad == pytest.approx(math.atan(0.10), abs=0.03)


def estimate_grid_alignment_default(
    scan,
    expected_half_width_m=0.125,
    adjacent_wall_tolerance_m=0.080,
    pair_width_tolerance_m=0.080,
):
    return estimate_grid_alignment(
        scan,
        expected_half_width_m=expected_half_width_m,
        min_x_m=-0.18,
        max_x_m=0.45,
        min_side_distance_m=0.06,
        max_side_distance_m=0.45,
        min_points=8,
        min_span_x_m=0.12,
        max_rms_error_m=0.025,
        max_abs_yaw_error_rad=0.35,
        max_reported_error_m=0.30,
        max_reported_yaw_rad=0.50,
        adjacent_wall_tolerance_m=adjacent_wall_tolerance_m,
        pair_width_tolerance_m=pair_width_tolerance_m,
    )


def test_estimate_grid_alignment_uses_direct_right_not_far_left_pair():
    # Lab geometry:
    # - far/open left wall around +0.350 m
    # - adjacent right wall around -0.142 m
    # The raw left line may be geometrically valid, but it must not be paired
    # with the adjacent right wall as the current-cell corridor.
    scan = make_line_scan([
        (0.08, 0.350, -0.20, 0.45),
        (-0.03, -0.142, -0.20, 0.45),
    ])

    estimate = estimate_grid_alignment_default(scan)

    assert estimate.left.valid
    assert estimate.right.valid
    assert estimate.valid
    assert estimate.source == 'right'
    assert estimate.confidence == pytest.approx(0.6)
    assert estimate.yaw_error_rad == pytest.approx(math.atan(-0.03), abs=0.03)
    assert estimate.lateral_error_m == pytest.approx(-0.017, abs=0.02)
    assert 'left not adjacent' in estimate.reason


def test_estimate_grid_alignment_right_adjacent_lateral_error_sign():
    scan = make_line_scan([
        (0.0, -0.142, -0.20, 0.45),
    ])

    estimate = estimate_grid_alignment_default(scan)

    assert estimate.valid
    assert estimate.source == 'right'
    assert estimate.lateral_valid
    # right.offset + expected_half_width = -0.142 + 0.125 = -0.017.
    # Negative means desired centerline is to the robot's right.
    assert estimate.lateral_error_m == pytest.approx(-0.017, abs=0.02)


def test_estimate_grid_alignment_far_left_only_is_not_current_cell_alignment():
    scan = make_line_scan([
        (0.0, 0.350, -0.20, 0.45),
    ])

    estimate = estimate_grid_alignment_default(scan)

    assert estimate.left.valid
    assert not estimate.right.valid
    assert not estimate.valid
    assert not estimate.yaw_valid
    assert not estimate.lateral_valid
    assert estimate.source == 'none'
    assert 'no adjacent side walls' in estimate.reason


def test_estimate_grid_alignment_direct_pair_still_left_right():
    scan = make_line_scan([
        (0.0, 0.125, -0.20, 0.45),
        (0.0, -0.125, -0.20, 0.45),
    ])

    estimate = estimate_grid_alignment_default(scan)

    assert estimate.valid
    assert estimate.source == 'left_right'
    assert estimate.confidence == pytest.approx(1.0)
    assert estimate.yaw_error_rad == pytest.approx(0.0, abs=0.02)
    assert estimate.lateral_error_m == pytest.approx(0.0, abs=0.02)


def test_estimate_grid_alignment_rejects_bad_pair_width_and_uses_better_wall():
    # Both walls are barely individually adjacent, but together imply a corridor
    # much wider than the current cell. This must not become left_right.
    # Left has fewer points/span and worse geometry; right should be selected.
    scan = make_line_scan([
        (0.05, 0.204, -0.05, 0.20),
        (-0.02, -0.204, -0.20, 0.45),
    ])

    estimate = estimate_grid_alignment_default(scan)

    assert estimate.valid
    assert estimate.source in {'left', 'right'}
    assert estimate.source != 'left_right'
    assert 'pair rejected by width' in estimate.reason


def test_side_wall_observation_rejects_invalid_alignment():
    observation = side_wall_observation_from_alignment(
        invalid_grid_alignment('bad geometry')
    )

    assert not observation.valid
    assert 'grid alignment invalid' in observation.reason


def test_side_wall_observation_rejects_no_adjacent_source():
    alignment = make_alignment(source='none')
    observation = side_wall_observation_from_alignment(alignment)

    assert not observation.valid
    assert 'no adjacent side-wall candidate' in observation.reason


def test_side_wall_observation_accepts_right_adjacent_wall():
    alignment = make_alignment(right=make_wall('right', -0.125))
    observation = side_wall_observation_from_alignment(alignment)

    assert observation.valid
    assert observation.side == 'right'
    assert observation.confidence == pytest.approx(0.60)


def test_side_wall_observation_rejects_large_yaw():
    alignment = make_alignment(right=make_wall('right', -0.125, yaw=0.110))
    observation = side_wall_observation_from_alignment(alignment)

    assert not observation.valid
    assert 'yaw too large' in observation.reason


def test_side_wall_observation_rejects_large_rms():
    alignment = make_alignment(right=make_wall('right', -0.125, rms=0.021))
    observation = side_wall_observation_from_alignment(alignment)

    assert not observation.valid
    assert 'rms too high' in observation.reason


def test_side_wall_observation_rejects_small_span():
    alignment = make_alignment(right=make_wall('right', -0.125, span=0.219))
    observation = side_wall_observation_from_alignment(alignment)

    assert not observation.valid
    assert 'span too small' in observation.reason


def test_side_wall_observation_rejects_low_support():
    alignment = make_alignment(right=make_wall('right', -0.125, count=79))
    observation = side_wall_observation_from_alignment(alignment)

    assert not observation.valid
    assert 'support too low' in observation.reason


def test_side_wall_observation_left_right_chooses_better_candidate():
    alignment = make_alignment(
        source='left_right',
        confidence=1.0,
        left=make_wall('left', 0.125, span=0.35, count=140, rms=0.011),
        right=make_wall('right', -0.125, span=0.30, count=160, rms=0.008),
    )
    observation = side_wall_observation_from_alignment(alignment)

    assert observation.valid
    assert observation.side == 'left'


def test_side_wall_observation_honors_preferred_side():
    alignment = make_alignment(
        source='left_right',
        confidence=1.0,
        left=make_wall('left', 0.125, span=0.35),
        right=make_wall('right', -0.125, span=0.30),
    )
    observation = side_wall_observation_from_alignment(
        alignment,
        preferred_side='right',
    )

    assert observation.valid
    assert observation.side == 'right'


def test_lidar_parallelity_correction_accepts_consistent_yaw_and_drift():
    correction, drift_yaw, reason = lidar_parallelity_correction_radps(
        yaw_error_rad=0.05,
        drift_per_m=math.tan(0.04),
        k_yaw=0.80,
        k_drift=0.20,
        max_correction_radps=0.040,
        require_consistency=True,
        max_yaw_drift_disagreement_rad=0.060,
    )

    assert drift_yaw == pytest.approx(0.04)
    assert correction > 0.0
    assert 'fused LiDAR parallelity correction' in reason


def test_lidar_parallelity_correction_rejects_sign_disagreement():
    correction, _drift_yaw, reason = lidar_parallelity_correction_radps(
        yaw_error_rad=0.05,
        drift_per_m=math.tan(-0.04),
        k_yaw=0.80,
        k_drift=0.20,
        max_correction_radps=0.040,
        require_consistency=True,
        max_yaw_drift_disagreement_rad=0.060,
    )

    assert correction == pytest.approx(0.0)
    assert 'sign disagreement' in reason


def test_lidar_parallelity_correction_rejects_magnitude_disagreement():
    correction, _drift_yaw, reason = lidar_parallelity_correction_radps(
        yaw_error_rad=0.02,
        drift_per_m=math.tan(0.10),
        k_yaw=0.80,
        k_drift=0.20,
        max_correction_radps=0.040,
        require_consistency=True,
        max_yaw_drift_disagreement_rad=0.060,
    )

    assert correction == pytest.approx(0.0)
    assert 'magnitude disagreement' in reason


def test_lidar_parallelity_correction_is_capped():
    correction, _drift_yaw, _reason = lidar_parallelity_correction_radps(
        yaw_error_rad=0.20,
        drift_per_m=math.tan(0.20),
        k_yaw=0.80,
        k_drift=0.20,
        max_correction_radps=0.040,
        require_consistency=True,
        max_yaw_drift_disagreement_rad=0.060,
    )

    assert correction == pytest.approx(0.040)


def test_temporal_lidar_progress_raw_valid_resets_degraded_samples():
    raw = LidarProgressEstimate(True, 0.12, 'front_rear', 0.001, 'raw ok')
    estimate = choose_temporal_lidar_progress(
        raw=raw,
        previous_valid=True,
        previous_progress_m=0.10,
        previous_source='front_temporal',
        front_valid=True,
        front_progress_m=0.12,
        rear_valid=True,
        rear_progress_m=0.12,
        degraded_samples=2,
        max_backtrack_m=0.015,
        max_jump_m=0.080,
        max_degraded_samples=2,
    )

    assert estimate.valid
    assert not estimate.degraded
    assert estimate.degraded_samples == 0
    assert estimate.source == 'front_rear'


def test_temporal_lidar_progress_raw_invalid_without_previous_is_invalid():
    raw = LidarProgressEstimate(False, 0.0, 'none', 0.0, 'raw bad')
    estimate = choose_temporal_lidar_progress(
        raw=raw,
        previous_valid=False,
        previous_progress_m=0.0,
        previous_source='none',
        front_valid=True,
        front_progress_m=0.04,
        rear_valid=False,
        rear_progress_m=0.0,
        degraded_samples=0,
        max_backtrack_m=0.015,
        max_jump_m=0.080,
        max_degraded_samples=2,
    )

    assert not estimate.valid
    assert 'no previous LiDAR progress' in estimate.reason


def test_temporal_lidar_progress_recovers_with_monotonic_front_source():
    raw = LidarProgressEstimate(False, 0.0, 'front_rear_rejected', 0.10, 'raw bad')
    estimate = choose_temporal_lidar_progress(
        raw=raw,
        previous_valid=True,
        previous_progress_m=0.10,
        previous_source='front',
        front_valid=True,
        front_progress_m=0.14,
        rear_valid=True,
        rear_progress_m=0.20,
        degraded_samples=0,
        max_backtrack_m=0.015,
        max_jump_m=0.080,
        max_degraded_samples=2,
    )

    assert estimate.valid
    assert estimate.degraded
    assert estimate.source == 'front_temporal'
    assert estimate.progress_m == pytest.approx(0.14)


def test_temporal_lidar_progress_rejects_backtrack_candidates():
    raw = LidarProgressEstimate(False, 0.0, 'front_rear_rejected', 0.10, 'raw bad')
    estimate = choose_temporal_lidar_progress(
        raw=raw,
        previous_valid=True,
        previous_progress_m=0.10,
        previous_source='front',
        front_valid=True,
        front_progress_m=0.07,
        rear_valid=True,
        rear_progress_m=0.06,
        degraded_samples=0,
        max_backtrack_m=0.015,
        max_jump_m=0.080,
        max_degraded_samples=2,
    )

    assert not estimate.valid
    assert 'no temporal LiDAR candidate' in estimate.reason


def test_temporal_lidar_progress_rejects_when_degraded_exhausted():
    raw = LidarProgressEstimate(False, 0.0, 'front_rear_rejected', 0.10, 'raw bad')
    estimate = choose_temporal_lidar_progress(
        raw=raw,
        previous_valid=True,
        previous_progress_m=0.10,
        previous_source='front',
        front_valid=True,
        front_progress_m=0.11,
        rear_valid=False,
        rear_progress_m=0.0,
        degraded_samples=2,
        max_backtrack_m=0.015,
        max_jump_m=0.080,
        max_degraded_samples=2,
    )

    assert not estimate.valid
    assert estimate.source == 'temporal_rejected'
    assert 'exhausted' in estimate.reason


def test_temporal_lidar_progress_never_references_odom():
    raw = LidarProgressEstimate(False, 0.0, 'none', 0.0, 'raw bad')
    estimate = choose_temporal_lidar_progress(
        raw=raw,
        previous_valid=True,
        previous_progress_m=0.10,
        previous_source='front',
        front_valid=True,
        front_progress_m=0.11,
        rear_valid=False,
        rear_progress_m=0.0,
        degraded_samples=0,
        max_backtrack_m=0.015,
        max_jump_m=0.080,
        max_degraded_samples=2,
    )

    assert 'odom' not in estimate.reason.lower()
    assert 'odom' not in estimate.source.lower()
