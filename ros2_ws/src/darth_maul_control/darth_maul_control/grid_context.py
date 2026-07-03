from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from darth_maul_control.geometry import clamp
from darth_maul_control.grid_yaw import GridYawObservation
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

GRID_YAW_LOCK = 'grid_yaw_lock'
HEADING_COAST = 'heading_coast'
YAW_REACQUIRE = 'yaw_reacquire'
YAW_RECOVERY = 'yaw_recovery'
CENTER_LOCK = 'center_lock'
LATERAL_COAST = 'lateral_coast'

# Keep these old names only for compatibility with older tests/status.
WALL_LOCK = 'wall_lock'
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
class GridCenteringObservation:
    valid: bool
    lateral_error_m: float
    confidence: float
    source: str
    matched_expected_wall_ids: tuple[str, ...]
    left_usable: bool
    right_usable: bool
    reason: str


@dataclass(frozen=True)
class GridObservation:
    context_valid: bool
    virtual_cell: VirtualCellEstimate
    expected: ExpectedWalls

    yaw: GridYawObservation
    centering: GridCenteringObservation

    yaw_mode: str
    lateral_mode: str
    combined_mode: str
    reason: str


@dataclass(frozen=True)
class LiveGridCommand:
    mode: str
    yaw_mode: str
    lateral_mode: str
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


def invalid_centering(reason: str) -> GridCenteringObservation:
    return GridCenteringObservation(
        valid=False,
        lateral_error_m=0.0,
        confidence=0.0,
        source='none',
        matched_expected_wall_ids=(),
        left_usable=False,
        right_usable=False,
        reason=reason,
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


def observe_centering_from_expected_side_walls(
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
    min_confidence: float,
) -> tuple[VirtualCellEstimate, ExpectedWalls, GridCenteringObservation]:
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
        reason = f'grid context unavailable: {context.reason}; {virtual.reason}; {expected.reason}'
        return virtual, expected, invalid_centering(reason)

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
            lateral_error = (alignment.left.offset_m + alignment.right.offset_m) / 2.0
            confidence = 1.0 * boundary_scale
            if confidence >= float(min_confidence):
                return virtual, expected, GridCenteringObservation(
                    valid=True,
                    lateral_error_m=float(lateral_error),
                    confidence=float(confidence),
                    source='left_right',
                    matched_expected_wall_ids=('left', 'right'),
                    left_usable=True,
                    right_usable=True,
                    reason=(
                        f'left/right expected side walls accepted; '
                        f'virtual_cell={virtual.cell_idx}; width_error={width_error:.3f}; '
                        f'boundary_zone={virtual.boundary_zone}'
                    ),
                )

    if left_ok:
        confidence = 0.60 * boundary_scale
        if confidence >= float(min_confidence):
            return virtual, expected, GridCenteringObservation(
                valid=True,
                lateral_error_m=float(wall_offset_error_m(alignment.left, expected_half_width_m)),
                confidence=float(confidence),
                source='left',
                matched_expected_wall_ids=('left',),
                left_usable=True,
                right_usable=False,
                reason=(
                    f'expected left wall accepted; {right_reason}; '
                    f'boundary_zone={virtual.boundary_zone}'
                ),
            )

    if right_ok:
        confidence = 0.60 * boundary_scale
        if confidence >= float(min_confidence):
            return virtual, expected, GridCenteringObservation(
                valid=True,
                lateral_error_m=float(wall_offset_error_m(alignment.right, expected_half_width_m)),
                confidence=float(confidence),
                source='right',
                matched_expected_wall_ids=('right',),
                left_usable=False,
                right_usable=True,
                reason=(
                    f'expected right wall accepted; {left_reason}; '
                    f'boundary_zone={virtual.boundary_zone}'
                ),
            )

    return virtual, expected, invalid_centering(
        (
            f'no usable expected side wall for centering; virtual_cell={virtual.cell_idx}; '
            f'expected_left={expected.left}; expected_right={expected.right}; '
            f'{left_reason}; {right_reason}'
        )
    )


def make_grid_observation(
    *,
    context: GridRunContext,
    yaw: GridYawObservation,
    centering: GridCenteringObservation,
    virtual: VirtualCellEstimate,
    expected: ExpectedWalls,
) -> GridObservation:
    context_valid = bool(context.valid and virtual.valid and expected.valid)

    yaw_mode = GRID_YAW_LOCK if yaw.valid else HEADING_COAST
    lateral_mode = CENTER_LOCK if centering.valid else LATERAL_COAST

    if not context_valid:
        combined = UNAVAILABLE
    elif yaw.valid and centering.valid:
        combined = WALL_LOCK
    elif yaw.valid and not centering.valid:
        combined = GRID_YAW_LOCK
    elif not yaw.valid and centering.valid:
        combined = CENTER_LOCK
    else:
        combined = HEADING_COAST

    return GridObservation(
        context_valid=context_valid,
        virtual_cell=virtual,
        expected=expected,
        yaw=yaw,
        centering=centering,
        yaw_mode=yaw_mode,
        lateral_mode=lateral_mode,
        combined_mode=combined,
        reason=f'yaw={yaw.reason}; centering={centering.reason}',
    )


def live_grid_command(
    *,
    observation: GridObservation,
    previous_yaw_mode: str,
    odom_heading_correction_radps: float,
    k_yaw: float,
    max_yaw_correction_radps: float,
    k_lateral: float,
    max_lateral_mps: float,
    yaw_min_confidence: float,
    lateral_min_confidence: float,
    reacquire_stable_samples: int,
    current_reacquire_samples: int,
    small_reacquire_yaw_rad: float,
    large_reacquire_yaw_rad: float,
    reacquire_speed_scale: float,
) -> LiveGridCommand:
    yaw_valid = bool(
        observation.yaw.valid
        and observation.yaw.confidence >= float(yaw_min_confidence)
    )
    lateral_valid = bool(
        observation.centering.valid
        and observation.centering.confidence >= float(lateral_min_confidence)
    )

    raw_lateral = 0.0
    if lateral_valid:
        raw_lateral = clamp(
            float(k_lateral) * float(observation.centering.lateral_error_m),
            -abs(float(max_lateral_mps)),
            abs(float(max_lateral_mps)),
        )

    if not yaw_valid:
        return LiveGridCommand(
            mode=observation.combined_mode,
            yaw_mode=HEADING_COAST,
            lateral_mode=CENTER_LOCK if lateral_valid else LATERAL_COAST,
            yaw_active=False,
            lateral_active=lateral_valid,
            angular_z_radps=float(odom_heading_correction_radps),
            linear_y_mps=raw_lateral,
            speed_scale=1.0,
            reason=(
                f'heading coast: {observation.yaw.reason}; '
                f'centering={observation.centering.reason}'
            ),
        )

    raw_yaw = clamp(
        float(k_yaw) * float(observation.yaw.yaw_error_rad),
        -abs(float(max_yaw_correction_radps)),
        abs(float(max_yaw_correction_radps)),
    )

    abs_yaw_error = abs(float(observation.yaw.yaw_error_rad))
    yaw_mode = GRID_YAW_LOCK
    speed_scale = 1.0
    angular_z = raw_yaw
    reason_prefix = 'grid yaw lock'

    if previous_yaw_mode == HEADING_COAST:
        if abs_yaw_error >= float(large_reacquire_yaw_rad):
            return LiveGridCommand(
                mode=YAW_RECOVERY,
                yaw_mode=YAW_RECOVERY,
                lateral_mode=CENTER_LOCK if lateral_valid else LATERAL_COAST,
                yaw_active=False,
                lateral_active=False,
                angular_z_radps=0.0,
                linear_y_mps=0.0,
                speed_scale=0.0,
                reason=(
                    f'reacquire yaw disagreement too large: '
                    f'{abs_yaw_error:.3f} >= {float(large_reacquire_yaw_rad):.3f}; '
                    f'{observation.yaw.reason}'
                ),
            )

        if abs_yaw_error > float(small_reacquire_yaw_rad):
            yaw_mode = YAW_REACQUIRE
            speed_scale = float(reacquire_speed_scale)
            reason_prefix = 'yaw reacquire reduced speed'
        else:
            weight = min(
                1.0,
                max(1, int(current_reacquire_samples)) / max(1, int(reacquire_stable_samples)),
            )
            angular_z = (1.0 - weight) * float(odom_heading_correction_radps) + weight * raw_yaw
            raw_lateral *= weight
            yaw_mode = YAW_REACQUIRE if weight < 1.0 else GRID_YAW_LOCK
            reason_prefix = f'smooth yaw reacquire weight={weight:.2f}'

    lateral_mode = CENTER_LOCK if lateral_valid else LATERAL_COAST

    if yaw_mode == GRID_YAW_LOCK and lateral_mode == CENTER_LOCK:
        combined_mode = WALL_LOCK
    elif yaw_mode == GRID_YAW_LOCK:
        combined_mode = GRID_YAW_LOCK
    elif yaw_mode == YAW_REACQUIRE:
        combined_mode = YAW_REACQUIRE
    else:
        combined_mode = observation.combined_mode

    return LiveGridCommand(
        mode=combined_mode,
        yaw_mode=yaw_mode,
        lateral_mode=lateral_mode,
        yaw_active=True,
        lateral_active=lateral_valid,
        angular_z_radps=float(angular_z),
        linear_y_mps=float(raw_lateral),
        speed_scale=float(speed_scale),
        reason=(
            f'{reason_prefix}: yaw_source={observation.yaw.source}; '
            f'yaw={observation.yaw.yaw_error_rad:.3f}; '
            f'yaw_conf={observation.yaw.confidence:.2f}; '
            f'lateral_mode={lateral_mode}; lateral_source={observation.centering.source}; '
            f'{observation.reason}'
        ),
    )
