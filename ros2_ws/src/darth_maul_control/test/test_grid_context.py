import math
from types import SimpleNamespace

import pytest

from darth_maul_control.grid_context import (
    GRID_YAW_LOCK,
    HEADING_COAST,
    LATERAL_COAST,
    WALL_LOCK,
    GridRunContext,
    advance_cell,
    expected_walls_for_cell,
    invalid_centering,
    live_grid_command,
    make_grid_observation,
    observe_centering_from_expected_side_walls,
    virtual_cell_for_progress,
)
from darth_maul_control.grid_yaw import GridYawObservation, invalid_grid_yaw
from darth_maul_control.scan_geometry import estimate_grid_alignment


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


def make_line_scan(lines, num_samples=720, angle_min=-math.pi, angle_max=math.pi):
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
            if not math.isfinite(r) or r < 0.05 or r > 12.0:
                continue
            x = r * c
            if x_min <= x <= x_max:
                best = min(best, r)

        ranges[i] = best

    return make_scan(ranges, angle_min=angle_min, angle_increment=angle_increment)


def alignment_from_lines(lines, half_width=0.20):
    scan = make_line_scan(lines)
    return estimate_grid_alignment(
        scan,
        expected_half_width_m=half_width,
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


def valid_yaw(error=0.0, confidence=0.8, source='manhattan_lines'):
    return GridYawObservation(
        valid=True,
        yaw_error_rad=error,
        confidence=confidence,
        source=source,
        line_count=1,
        dominant_axis_rad=error,
        total_weight=1.0,
        concentration=1.0,
        reason='test yaw',
    )


def observe_centering(context, alignment):
    return observe_centering_from_expected_side_walls(
        context=context,
        alignment=alignment,
        progress_m=0.10,
        cell_length_m=0.254,
        boundary_margin_m=0.025,
        expected_half_width_m=0.20,
        adjacent_wall_tolerance_m=0.08,
        pair_width_tolerance_m=0.08,
        max_abs_yaw_error_rad=0.12,
        max_rms_error_m=0.025,
        min_span_x_m=0.12,
        min_support_count=8,
        min_confidence=0.55,
    )


def test_advance_cell_uses_readme_indexing():
    assert advance_cell(1, 1, 7, 11) == 2
    assert advance_cell(1, 4, 7, 11) == 8
    assert advance_cell(8, 8, 7, 11) == 1
    assert advance_cell(1, 2, 7, 11) == 0


def test_virtual_cell_tracks_long_run_by_lidar_progress():
    context = GridRunContext(
        True,
        n=7,
        m=11,
        start_idx=1,
        heading=1,
        run_cells=4,
        l=tuple([0] * 77),
        reason='test',
    )

    first = virtual_cell_for_progress(context, 0.10, 0.254, 0.025)
    second = virtual_cell_for_progress(context, 0.260, 0.254, 0.025)
    fourth = virtual_cell_for_progress(context, 0.900, 0.254, 0.025)

    assert first.cell_idx == 1
    assert second.cell_idx == 2
    assert fourth.cell_idx == 4
    assert fourth.completed_cells == 3


def test_expected_walls_maps_global_to_robot_frame_heading_pos_x():
    context = GridRunContext(True, 2, 2, 1, 1, 1, tuple([12, 0, 0, 0]), 'test')
    expected = expected_walls_for_cell(context, 1)

    assert expected.left
    assert expected.right
    assert not expected.front
    assert not expected.rear


def test_observation_accepts_expected_both_side_walls():
    context = GridRunContext(True, 2, 2, 1, 1, 1, tuple([12, 0, 0, 0]), 'test')
    alignment = alignment_from_lines([
        (0.0, 0.20, -0.20, 0.45),
        (0.0, -0.20, -0.20, 0.45),
    ])

    virtual, expected, centering = observe_centering(context, alignment)
    obs = make_grid_observation(
        context=context,
        yaw=valid_yaw(),
        centering=centering,
        virtual=virtual,
        expected=expected,
    )

    assert obs.combined_mode == WALL_LOCK
    assert obs.yaw.valid
    assert obs.centering.valid
    assert obs.centering.source == 'left_right'
    assert obs.centering.lateral_error_m == pytest.approx(0.0, abs=0.02)


def test_observation_rejects_wall_when_map_says_open_side():
    context = GridRunContext(True, 2, 2, 1, 1, 1, tuple([0, 0, 0, 0]), 'test')
    alignment = alignment_from_lines([
        (0.0, 0.20, -0.20, 0.45),
        (0.0, -0.20, -0.20, 0.45),
    ])

    virtual, expected, centering = observe_centering(context, alignment)
    obs = make_grid_observation(
        context=context,
        yaw=invalid_grid_yaw('no lines'),
        centering=centering,
        virtual=virtual,
        expected=expected,
    )

    assert obs.combined_mode == HEADING_COAST
    assert not obs.yaw.valid
    assert not obs.centering.valid


def test_lateral_error_sign_left_wall_robot_right_of_center():
    context = GridRunContext(True, 2, 2, 1, 1, 1, tuple([4, 0, 0, 0]), 'test')
    alignment = alignment_from_lines([(0.0, 0.25, -0.20, 0.45)])

    _virtual, _expected, centering = observe_centering(context, alignment)

    assert centering.valid
    assert centering.source == 'left'
    assert centering.lateral_error_m > 0.0


def test_live_grid_command_uses_wall_yaw_and_lateral_when_locked():
    context = GridRunContext(True, 2, 2, 1, 1, 1, tuple([4, 0, 0, 0]), 'test')
    alignment = alignment_from_lines([(0.05, 0.25, -0.20, 0.45)])
    virtual, expected, centering = observe_centering(context, alignment)
    obs = make_grid_observation(
        context=context,
        yaw=valid_yaw(0.05),
        centering=centering,
        virtual=virtual,
        expected=expected,
    )

    cmd = live_grid_command(
        observation=obs,
        previous_yaw_mode=GRID_YAW_LOCK,
        odom_heading_correction_radps=0.0,
        k_yaw=2.0,
        max_yaw_correction_radps=0.12,
        k_lateral=1.4,
        max_lateral_mps=0.035,
        yaw_min_confidence=0.55,
        lateral_min_confidence=0.55,
        reacquire_stable_samples=3,
        current_reacquire_samples=3,
        small_reacquire_yaw_rad=0.04,
        large_reacquire_yaw_rad=0.10,
        reacquire_speed_scale=0.65,
    )

    assert cmd.mode == WALL_LOCK
    assert cmd.yaw_active
    assert cmd.lateral_active
    assert cmd.linear_y_mps > 0.0


def test_live_grid_command_falls_back_to_odom_heading_when_blind():
    context = GridRunContext(True, 2, 2, 1, 1, 1, tuple([0, 0, 0, 0]), 'test')
    alignment = alignment_from_lines([(0.0, 0.20, -0.20, 0.45)])
    virtual, expected, centering = observe_centering(context, alignment)
    obs = make_grid_observation(
        context=context,
        yaw=invalid_grid_yaw('no lines'),
        centering=centering,
        virtual=virtual,
        expected=expected,
    )

    cmd = live_grid_command(
        observation=obs,
        previous_yaw_mode=GRID_YAW_LOCK,
        odom_heading_correction_radps=0.07,
        k_yaw=2.0,
        max_yaw_correction_radps=0.12,
        k_lateral=1.4,
        max_lateral_mps=0.035,
        yaw_min_confidence=0.55,
        lateral_min_confidence=0.55,
        reacquire_stable_samples=3,
        current_reacquire_samples=0,
        small_reacquire_yaw_rad=0.04,
        large_reacquire_yaw_rad=0.10,
        reacquire_speed_scale=0.65,
    )

    assert cmd.mode == HEADING_COAST
    assert not cmd.yaw_active
    assert not cmd.lateral_active
    assert cmd.angular_z_radps == pytest.approx(0.07)
    assert cmd.linear_y_mps == pytest.approx(0.0)


def test_front_wall_yaw_can_be_valid_while_lateral_is_invalid():
    context = GridRunContext(True, 2, 2, 1, 1, 1, tuple([0, 0, 0, 0]), 'test')
    alignment = alignment_from_lines([])

    virtual, expected, centering = observe_centering(context, alignment)
    obs = make_grid_observation(
        context=context,
        yaw=valid_yaw(0.03),
        centering=centering,
        virtual=virtual,
        expected=expected,
    )

    assert obs.yaw.valid
    assert not obs.centering.valid
    assert obs.yaw_mode == GRID_YAW_LOCK
    assert obs.lateral_mode == LATERAL_COAST


def test_live_grid_command_uses_yaw_without_lateral():
    context = GridRunContext(True, 2, 2, 1, 1, 1, tuple([0, 0, 0, 0]), 'test')
    virtual = virtual_cell_for_progress(context, 0.10, 0.254, 0.025)
    expected = expected_walls_for_cell(context, virtual.cell_idx)
    obs = make_grid_observation(
        context=context,
        yaw=valid_yaw(0.04),
        centering=invalid_centering('no side wall'),
        virtual=virtual,
        expected=expected,
    )

    cmd = live_grid_command(
        observation=obs,
        previous_yaw_mode=GRID_YAW_LOCK,
        odom_heading_correction_radps=0.0,
        k_yaw=2.0,
        max_yaw_correction_radps=0.12,
        k_lateral=1.4,
        max_lateral_mps=0.035,
        yaw_min_confidence=0.55,
        lateral_min_confidence=0.55,
        reacquire_stable_samples=3,
        current_reacquire_samples=3,
        small_reacquire_yaw_rad=0.04,
        large_reacquire_yaw_rad=0.10,
        reacquire_speed_scale=0.65,
    )

    assert cmd.yaw_active
    assert not cmd.lateral_active
    assert cmd.angular_z_radps == pytest.approx(0.08)
    assert cmd.linear_y_mps == pytest.approx(0.0)
