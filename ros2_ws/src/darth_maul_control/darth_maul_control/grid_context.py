from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from darth_maul_control.geometry import clamp
from darth_maul_control.scan_geometry import (
    GridAlignmentEstimate,
    WallLineEstimate,
    expected_side_offset_m,
    is_adjacent_wall_line,
    wall_offset_error_m,
)


POS_X = 1
NEG_X = 2
POS_Y = 4
NEG_Y = 8
MISSING = 15

VALID_HEADINGS = {POS_X, NEG_X, POS_Y, NEG_Y}

OPPOSITE = {
    POS_X: NEG_X,
    NEG_X: POS_X,
    POS_Y: NEG_Y,
    NEG_Y: POS_Y,
}

LEFT_OF = {
    POS_X: POS_Y,
    POS_Y: NEG_X,
    NEG_X: NEG_Y,
    NEG_Y: POS_X,
}

RIGHT_OF = {
    POS_X: NEG_Y,
    NEG_Y: NEG_X,
    NEG_X: POS_Y,
    POS_Y: POS_X,
}

WALL_LOCK = 'wall_lock'
HEADING_COAST = 'heading_coast'
REACQUIRE = 'reacquire'
RECOVERY = 'recovery'
UNAVAILABLE = 'unavailable'


@dataclass(frozen=True)
class GridRunContext:
    valid: bool
    n: int
    m: int
    start_idx: int
    heading: int
    run_cells: int
    l: tuple[int, ...]
    reason: str = ''


@dataclass(frozen=True)
class VirtualCellEstimate:
    valid: bool
    cell_idx: int
    completed_cells: int
    progress_m: float
    distance_into_cell_m: float
    boundary_zone: bool
    reason: str


@dataclass(frozen=True)
class ExpectedWalls:
    valid: bool
    cell_idx: int
    cell_value: int
    front: bool
    rear: bool
    left: bool
    right: bool
    reason: str


@dataclass(frozen=True)
class GridObservation:
    context_valid: bool
    mode: str
    virtual_cell: VirtualCellEstimate
    expected: ExpectedWalls

    yaw_valid: bool
    yaw_error_rad: float
    lateral_valid: bool
    lateral_error_m: float

    source: str
    confidence: float
    matched_expected_wall_ids: tuple[str, ...]
    left_usable: bool
    right_usable: bool
    reason: str


@dataclass(frozen=True)
class LiveGridCommand:
    mode: str
    yaw_active: bool
    lateral_active: bool
    angular_z_radps: float
    linear_y_mps: float
    speed_scale: float
    reason: str


def idx_to_ij(index: int, n: int) -> tuple[int, int]:
    if n <= 0:
        raise ValueError('n must be positive')
    if index <= 0:
        raise ValueError('index must be 1-based and positive')
    return (int(index) - 1) % int(n), (int(index) - 1) // int(n)


def ij_to_idx(i: int, j: int, n: int, m: int) -> int:
    if i < 0 or i >= n or j < 0 or j >= m:
        return 0
    return int(j) * int(n) + int(i) + 1


def advance_cell(index: int, heading: int, n: int, m: int) -> int:
    i, j = idx_to_ij(index, n)
    if heading == POS_X:
        i += 1
    elif heading == NEG_X:
        i -= 1
    elif heading == POS_Y:
        j += 1
    elif heading == NEG_Y:
        j -= 1
    else:
        return 0
    return ij_to_idx(i, j, n, m)


def advance_cells(index: int, heading: int, steps: int, n: int, m: int) -> int:
    cell = int(index)
    for _ in range(max(0, int(steps))):
        cell = advance_cell(cell, heading, n, m)
        if cell == 0:
            return 0
    return cell


def cell_value(l: Sequence[int], n: int, m: int, index: int) -> int:
    if index <= 0 or index > n * m:
        return MISSING
    if len(l) != n * m:
        return MISSING
    return int(l[index - 1])


def grid_context_from_goal(goal) -> GridRunContext:
    n = int(getattr(goal, 'grid_n', 0))
    m = int(getattr(goal, 'grid_m', 0))
    start_idx = int(getattr(goal, 'grid_start_idx', 0))
    heading = int(getattr(goal, 'grid_heading', 0))
    run_cells = int(getattr(goal, 'grid_run_cells', 0))
    raw_l = tuple(int(v) for v in getattr(goal, 'grid_l', []))

    if n <= 0 or m <= 0:
        return GridRunContext(False, n, m, start_idx, heading, run_cells, raw_l, 'grid_n/grid_m absent')
    if len(raw_l) != n * m:
        return GridRunContext(False, n, m, start_idx, heading, run_cells, raw_l, 'grid_l length mismatch')
    if start_idx < 1 or start_idx > n * m:
        return GridRunContext(False, n, m, start_idx, heading, run_cells, raw_l, 'grid_start_idx out of range')
    if heading not in VALID_HEADINGS:
        return GridRunContext(False, n, m, start_idx, heading, run_cells, raw_l, 'grid_heading invalid')
    if run_cells <= 0:
        return GridRunContext(False, n, m, start_idx, heading, run_cells, raw_l, 'grid_run_cells absent')
    if cell_value(raw_l, n, m, start_idx) == MISSING:
        return GridRunContext(False, n, m, start_idx, heading, run_cells, raw_l, 'start cell missing')

    return GridRunContext(True, n, m, start_idx, heading, run_cells, raw_l, 'valid grid context')


def virtual_cell_for_progress(
    context: GridRunContext,
    progress_m: float,
    cell_length_m: float,
    boundary_margin_m: float,
) -> VirtualCellEstimate:
    if not context.valid:
        return VirtualCellEstimate(False, 0, 0, 0.0, 0.0, False, context.reason)

    progress = max(0.0, float(progress_m))
    cell_length = max(float(cell_length_m), 1.0e-6)

    completed = int(math.floor((progress + 1.0e-9) / cell_length))
    completed = min(max(0, completed), max(0, context.run_cells - 1))

    cell = advance_cells(
        context.start_idx,
        context.heading,
        completed,
        context.n,
        context.m,
    )
    if cell == 0:
        return VirtualCellEstimate(False, 0, completed, progress, 0.0, False, 'virtual cell outside maze')

    distance_into = progress - completed * cell_length
    distance_into = max(0.0, min(cell_length, distance_into))

    margin = max(0.0, min(float(boundary_margin_m), 0.45 * cell_length))
    boundary = distance_into <= margin or (cell_length - distance_into) <= margin

    return VirtualCellEstimate(
        True,
        cell,
        completed,
        progress,
        distance_into,
        boundary,
        'valid virtual cell',
    )


def expected_walls_for_cell(context: GridRunContext, cell_idx: int) -> ExpectedWalls:
    if not context.valid:
        return ExpectedWalls(False, 0, MISSING, False, False, False, False, context.reason)

    value = cell_value(context.l, context.n, context.m, cell_idx)
    if value == MISSING:
        return ExpectedWalls(False, cell_idx, value, False, False, False, False, 'virtual cell missing')

    heading = context.heading
    front_dir = heading
    rear_dir = OPPOSITE[heading]
    left_dir = LEFT_OF[heading]
    right_dir = RIGHT_OF[heading]

    return ExpectedWalls(
        True,
        cell_idx,
        value,
        front=bool(value & front_dir),
        rear=bool(value & rear_dir),
        left=bool(value & left_dir),
        right=bool(value & right_dir),
        reason='valid expected walls',
    )


def _wall_quality_ok(
    wall: WallLineEstimate,
    *,
    expected_half_width_m: float,
    adjacent_wall_tolerance_m: float,
    max_abs_yaw_error_rad: float,
    max_rms_error_m: float,
    min_span_x_m: float,
    min_support_count: int,
) -> bool:
    return bool(
        wall.valid
        and is_adjacent_wall_line(
            wall,
            expected_half_width_m=expected_half_width_m,
            adjacent_wall_tolerance_m=adjacent_wall_tolerance_m,
        )
        and abs(float(wall.yaw_error_rad)) <= float(max_abs_yaw_error_rad)
        and float(wall.rms_error_m) <= float(max_rms_error_m)
        and float(wall.span_x_m) >= float(min_span_x_m)
        and int(wall.support_count) >= int(min_support_count)
    )


def _wall_reject_reason(
    side: str,
    wall: WallLineEstimate,
    *,
    expected: bool,
    expected_half_width_m: float,
    adjacent_wall_tolerance_m: float,
    max_abs_yaw_error_rad: float,
    max_rms_error_m: float,
    min_span_x_m: float,
    min_support_count: int,
) -> str:
    if not expected:
        return f'{side} wall not expected by maze map'
    if not wall.valid:
        return f'{side} wall invalid: {wall.reason}'
    offset_error = wall_offset_error_m(wall, expected_half_width_m)
    if abs(offset_error) > adjacent_wall_tolerance_m:
        expected_offset = expected_side_offset_m(side, expected_half_width_m)
        return (
            f'{side} wall offset mismatch: offset={wall.offset_m:.3f}, '
            f'expected={expected_offset:.3f}, error={offset_error:.3f}'
        )
    if abs(wall.yaw_error_rad) > max_abs_yaw_error_rad:
        return f'{side} yaw too large: {wall.yaw_error_rad:.3f}'
    if wall.rms_error_m > max_rms_error_m:
        return f'{side} rms too high: {wall.rms_error_m:.3f}'
    if wall.span_x_m < min_span_x_m:
        return f'{side} span too small: {wall.span_x_m:.3f}'
    if wall.support_count < min_support_count:
        return f'{side} support too low: {wall.support_count}'
    return f'{side} wall accepted'


def observe_grid(
    *,
    context: GridRunContext,
    alignment: GridAlignmentEstimate,
    progress_m: float,
    cell_length_m: float,
    boundary_margin_m: float,
    expected_half_width_m: float,
    adjacent_wall_tolerance_m: float,
    pair_width_tolerance_m: float,
    max_abs_yaw_error_rad: float,
    max_rms_error_m: float,
    min_span_x_m: float,
    min_support_count: int,
) -> GridObservation:
    virtual = virtual_cell_for_progress(
        context,
        progress_m=progress_m,
        cell_length_m=cell_length_m,
        boundary_margin_m=boundary_margin_m,
    )
    expected = (
        expected_walls_for_cell(context, virtual.cell_idx)
        if virtual.valid
        else ExpectedWalls(False, 0, MISSING, False, False, False, False, virtual.reason)
    )

    if not context.valid or not virtual.valid or not expected.valid:
        return GridObservation(
            False,
            UNAVAILABLE,
            virtual,
            expected,
            False,
            0.0,
            False,
            0.0,
            'none',
            0.0,
            (),
            False,
            False,
            f'grid context unavailable: {context.reason}; {virtual.reason}; {expected.reason}',
        )

    left_ok = bool(
        expected.left
        and _wall_quality_ok(
            alignment.left,
            expected_half_width_m=expected_half_width_m,
            adjacent_wall_tolerance_m=adjacent_wall_tolerance_m,
            max_abs_yaw_error_rad=max_abs_yaw_error_rad,
            max_rms_error_m=max_rms_error_m,
            min_span_x_m=min_span_x_m,
            min_support_count=min_support_count,
        )
    )
    right_ok = bool(
        expected.right
        and _wall_quality_ok(
            alignment.right,
            expected_half_width_m=expected_half_width_m,
            adjacent_wall_tolerance_m=adjacent_wall_tolerance_m,
            max_abs_yaw_error_rad=max_abs_yaw_error_rad,
            max_rms_error_m=max_rms_error_m,
            min_span_x_m=min_span_x_m,
            min_support_count=min_support_count,
        )
    )

    left_reason = _wall_reject_reason(
        'left',
        alignment.left,
        expected=expected.left,
        expected_half_width_m=expected_half_width_m,
        adjacent_wall_tolerance_m=adjacent_wall_tolerance_m,
        max_abs_yaw_error_rad=max_abs_yaw_error_rad,
        max_rms_error_m=max_rms_error_m,
        min_span_x_m=min_span_x_m,
        min_support_count=min_support_count,
    )
    right_reason = _wall_reject_reason(
        'right',
        alignment.right,
        expected=expected.right,
        expected_half_width_m=expected_half_width_m,
        adjacent_wall_tolerance_m=adjacent_wall_tolerance_m,
        max_abs_yaw_error_rad=max_abs_yaw_error_rad,
        max_rms_error_m=max_rms_error_m,
        min_span_x_m=min_span_x_m,
        min_support_count=min_support_count,
    )

    boundary_scale = 0.80 if virtual.boundary_zone else 1.0

    if left_ok and right_ok:
        observed_width = float(alignment.left.offset_m) - float(alignment.right.offset_m)
        expected_width = 2.0 * float(expected_half_width_m)
        width_error = observed_width - expected_width
        if abs(width_error) <= float(pair_width_tolerance_m):
            yaw_error = (alignment.left.yaw_error_rad + alignment.right.yaw_error_rad) / 2.0
            lateral_error = (alignment.left.offset_m + alignment.right.offset_m) / 2.0
            return GridObservation(
                True,
                WALL_LOCK,
                virtual,
                expected,
                True,
                float(yaw_error),
                True,
                float(lateral_error),
                'left_right',
                1.0 * boundary_scale,
                ('left', 'right'),
                True,
                True,
                (
                    f'left/right expected side walls accepted; '
                    f'virtual_cell={virtual.cell_idx}; width_error={width_error:.3f}; '
                    f'boundary_zone={virtual.boundary_zone}'
                ),
            )

    if left_ok:
        return GridObservation(
            True,
            WALL_LOCK,
            virtual,
            expected,
            True,
            float(alignment.left.yaw_error_rad),
            True,
            float(wall_offset_error_m(alignment.left, expected_half_width_m)),
            'left',
            0.60 * boundary_scale,
            ('left',),
            True,
            False,
            f'expected left wall accepted; {right_reason}; boundary_zone={virtual.boundary_zone}',
        )

    if right_ok:
        return GridObservation(
            True,
            WALL_LOCK,
            virtual,
            expected,
            True,
            float(alignment.right.yaw_error_rad),
            True,
            float(wall_offset_error_m(alignment.right, expected_half_width_m)),
            'right',
            0.60 * boundary_scale,
            ('right',),
            False,
            True,
            f'expected right wall accepted; {left_reason}; boundary_zone={virtual.boundary_zone}',
        )

    return GridObservation(
        True,
        HEADING_COAST,
        virtual,
        expected,
        False,
        0.0,
        False,
        0.0,
        'none',
        0.0,
        (),
        False,
        False,
        (
            f'no usable expected side wall; virtual_cell={virtual.cell_idx}; '
            f'expected_left={expected.left}; expected_right={expected.right}; '
            f'{left_reason}; {right_reason}'
        ),
    )


def live_grid_command(
    *,
    observation: GridObservation,
    previous_mode: str,
    odom_heading_correction_radps: float,
    k_yaw: float,
    max_yaw_correction_radps: float,
    k_lateral: float,
    max_lateral_mps: float,
    min_confidence: float,
    reacquire_stable_samples: int,
    current_reacquire_samples: int,
    small_reacquire_yaw_rad: float,
    large_reacquire_yaw_rad: float,
    reacquire_speed_scale: float,
) -> LiveGridCommand:
    if (
        not observation.context_valid
        or not observation.yaw_valid
        or observation.confidence < min_confidence
    ):
        return LiveGridCommand(
            HEADING_COAST,
            False,
            False,
            float(odom_heading_correction_radps),
            0.0,
            1.0,
            observation.reason,
        )

    raw_yaw = clamp(
        float(k_yaw) * float(observation.yaw_error_rad),
        -abs(float(max_yaw_correction_radps)),
        abs(float(max_yaw_correction_radps)),
    )

    raw_lateral = 0.0
    lateral_active = bool(observation.lateral_valid)
    if lateral_active:
        raw_lateral = clamp(
            float(k_lateral) * float(observation.lateral_error_m),
            -abs(float(max_lateral_mps)),
            abs(float(max_lateral_mps)),
        )

    abs_yaw_error = abs(float(observation.yaw_error_rad))
    if previous_mode == HEADING_COAST:
        if abs_yaw_error >= float(large_reacquire_yaw_rad):
            return LiveGridCommand(
                RECOVERY,
                False,
                False,
                0.0,
                0.0,
                0.0,
                (
                    f'reacquire yaw disagreement too large: '
                    f'{abs_yaw_error:.3f} >= {float(large_reacquire_yaw_rad):.3f}'
                ),
            )

        if abs_yaw_error > float(small_reacquire_yaw_rad):
            return LiveGridCommand(
                REACQUIRE,
                True,
                lateral_active,
                raw_yaw,
                raw_lateral,
                float(reacquire_speed_scale),
                (
                    f'reacquire with reduced speed: yaw={observation.yaw_error_rad:.3f}; '
                    f'source={observation.source}; {observation.reason}'
                ),
            )

        weight = min(
            1.0,
            max(1, int(current_reacquire_samples)) / max(1, int(reacquire_stable_samples)),
        )
        blended_yaw = (1.0 - weight) * float(odom_heading_correction_radps) + weight * raw_yaw
        return LiveGridCommand(
            REACQUIRE if weight < 1.0 else WALL_LOCK,
            True,
            lateral_active,
            blended_yaw,
            weight * raw_lateral,
            1.0,
            (
                f'smooth reacquire weight={weight:.2f}; '
                f'source={observation.source}; {observation.reason}'
            ),
        )

    return LiveGridCommand(
        WALL_LOCK,
        True,
        lateral_active,
        raw_yaw,
        raw_lateral,
        1.0,
        f'wall lock: source={observation.source}; {observation.reason}',
    )
