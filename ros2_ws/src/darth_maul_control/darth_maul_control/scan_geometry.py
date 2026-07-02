from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Dict, List, Sequence, Tuple

from darth_maul_control.geometry import normalize_angle


@dataclass(frozen=True)
class SectorRange:
    name: str
    center_angle_rad: float
    width_deg: float
    valid: bool
    count: int
    min_m: float
    median_m: float
    mean_m: float
    max_m: float


@dataclass(frozen=True)
class LidarProgressEstimate:
    valid: bool
    progress_m: float
    source: str
    disagreement_m: float
    reason: str


@dataclass(frozen=True)
class TranslationProgressSelection:
    valid: bool
    progress_m: float
    source: str
    reason: str


@dataclass(frozen=True)
class GridLateralDriftEstimate:
    start_valid: bool
    start_error_m: float
    end_valid: bool
    end_error_m: float
    drift_valid: bool
    drift_m: float
    drift_per_m: float
    reason: str


@dataclass(frozen=True)
class WallLineEstimate:
    valid: bool
    side: str
    offset_m: float
    yaw_error_rad: float
    slope: float
    intercept_m: float
    support_count: int
    span_x_m: float
    rms_error_m: float
    reason: str


@dataclass(frozen=True)
class GridAlignmentEstimate:
    valid: bool
    yaw_valid: bool
    yaw_error_rad: float
    lateral_valid: bool
    lateral_error_m: float
    source: str
    confidence: float
    left: WallLineEstimate
    right: WallLineEstimate
    reason: str


@dataclass(frozen=True)
class PostRotationGridYawRefineDecision:
    stable: bool
    should_correct: bool
    reason: str


@dataclass(frozen=True)
class SideWallObservation:
    valid: bool
    side: str
    offset_m: float
    yaw_error_rad: float
    rms_error_m: float
    span_x_m: float
    support_count: int
    confidence: float
    source: str
    reason: str


@dataclass(frozen=True)
class LidarParallelityEstimate:
    valid: bool
    active: bool
    side: str
    stable_samples: int
    progress_m: float
    start_offset_m: float
    current_offset_m: float
    offset_drift_m: float
    drift_per_m: float
    yaw_error_rad: float
    drift_yaw_error_rad: float
    fused_error_rad: float
    correction_radps: float
    reason: str


@dataclass(frozen=True)
class TemporalLidarProgressEstimate:
    valid: bool
    progress_m: float
    source: str
    degraded: bool
    degraded_samples: int
    disagreement_m: float
    reason: str


def invalid_wall_line(side: str, reason: str) -> WallLineEstimate:
    return WallLineEstimate(
        valid=False,
        side=side,
        offset_m=0.0,
        yaw_error_rad=0.0,
        slope=0.0,
        intercept_m=0.0,
        support_count=0,
        span_x_m=0.0,
        rms_error_m=0.0,
        reason=reason,
    )


def invalid_grid_alignment(reason: str) -> GridAlignmentEstimate:
    left = invalid_wall_line('left', 'not evaluated')
    right = invalid_wall_line('right', 'not evaluated')
    return GridAlignmentEstimate(
        valid=False,
        yaw_valid=False,
        yaw_error_rad=0.0,
        lateral_valid=False,
        lateral_error_m=0.0,
        source='none',
        confidence=0.0,
        left=left,
        right=right,
        reason=reason,
    )


def invalid_side_wall_observation(reason: str) -> SideWallObservation:
    return SideWallObservation(
        valid=False,
        side='none',
        offset_m=0.0,
        yaw_error_rad=0.0,
        rms_error_m=0.0,
        span_x_m=0.0,
        support_count=0,
        confidence=0.0,
        source='none',
        reason=reason,
    )


def invalid_lidar_parallelity(reason: str) -> LidarParallelityEstimate:
    return LidarParallelityEstimate(
        valid=False,
        active=False,
        side='none',
        stable_samples=0,
        progress_m=0.0,
        start_offset_m=0.0,
        current_offset_m=0.0,
        offset_drift_m=0.0,
        drift_per_m=0.0,
        yaw_error_rad=0.0,
        drift_yaw_error_rad=0.0,
        fused_error_rad=0.0,
        correction_radps=0.0,
        reason=reason,
    )


def valid_scan_points_xy(scan) -> List[Tuple[float, float]]:
    if scan is None:
        return []

    points: List[Tuple[float, float]] = []
    angle = float(scan.angle_min)

    for raw in scan.ranges:
        value = float(raw)
        if math.isfinite(value) and scan.range_min <= value <= scan.range_max:
            points.append((value * math.cos(angle), value * math.sin(angle)))
        angle += float(scan.angle_increment)

    return points


def side_wall_candidate_points(
    scan,
    side: str,
    min_x_m: float,
    max_x_m: float,
    min_side_distance_m: float,
    max_side_distance_m: float,
) -> List[Tuple[float, float]]:
    if side not in ('left', 'right'):
        raise ValueError(f'Unsupported side {side!r}')

    sign = 1.0 if side == 'left' else -1.0
    points = []

    for x, y in valid_scan_points_xy(scan):
        side_distance = sign * y
        if (
            min_x_m <= x <= max_x_m
            and min_side_distance_m <= side_distance <= max_side_distance_m
        ):
            points.append((x, y))

    return points


def fit_side_wall_line(
    scan,
    side: str,
    min_x_m: float,
    max_x_m: float,
    min_side_distance_m: float,
    max_side_distance_m: float,
    min_points: int,
    min_span_x_m: float,
    max_rms_error_m: float,
    max_abs_yaw_error_rad: float,
) -> WallLineEstimate:
    points: Sequence[Tuple[float, float]] = side_wall_candidate_points(
        scan,
        side=side,
        min_x_m=float(min_x_m),
        max_x_m=float(max_x_m),
        min_side_distance_m=float(min_side_distance_m),
        max_side_distance_m=float(max_side_distance_m),
    )

    if len(points) < int(min_points):
        return invalid_wall_line(
            side,
            f'not enough candidate points: {len(points)} < {int(min_points)}',
        )

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    span_x = max(xs) - min(xs)
    if span_x < float(min_span_x_m):
        return invalid_wall_line(
            side,
            f'x-span too small: {span_x:.3f} m < {float(min_span_x_m):.3f} m',
        )

    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)

    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom <= 1e-9:
        return invalid_wall_line(side, 'degenerate line fit')

    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denom
    intercept = mean_y - slope * mean_x

    residuals = [(y - (slope * x + intercept)) for x, y in points]
    rms = math.sqrt(sum(r * r for r in residuals) / len(residuals))
    yaw_error = math.atan(slope)

    if rms > float(max_rms_error_m):
        return invalid_wall_line(
            side,
            f'line rms too high: {rms:.3f} m > {float(max_rms_error_m):.3f} m',
        )

    if abs(yaw_error) > float(max_abs_yaw_error_rad):
        return invalid_wall_line(
            side,
            (
                f'wall yaw too large: {yaw_error:.3f} rad '
                f'> {float(max_abs_yaw_error_rad):.3f} rad'
            ),
        )

    if side == 'left' and intercept <= 0.0:
        return invalid_wall_line(side, f'left wall intercept not positive: {intercept:.3f}')
    if side == 'right' and intercept >= 0.0:
        return invalid_wall_line(side, f'right wall intercept not negative: {intercept:.3f}')

    return WallLineEstimate(
        valid=True,
        side=side,
        offset_m=float(intercept),
        yaw_error_rad=float(yaw_error),
        slope=float(slope),
        intercept_m=float(intercept),
        support_count=len(points),
        span_x_m=float(span_x),
        rms_error_m=float(rms),
        reason='valid side wall line',
    )


def grid_yaw_correction_radps(
    yaw_error_rad: float,
    k_yaw: float,
    max_correction_radps: float,
) -> float:
    correction = float(k_yaw) * float(yaw_error_rad)
    limit = abs(float(max_correction_radps))
    return float(max(-limit, min(limit, correction)))


def compose_angular_command(
    heading_correction_radps: float,
    grid_yaw_correction_radps: float,
    grid_yaw_active: bool,
    grid_yaw_active_heading_hold_scale: float,
) -> float:
    scale = 1.0
    if grid_yaw_active:
        scale = max(0.0, min(1.0, float(grid_yaw_active_heading_hold_scale)))
    return (
        scale * float(heading_correction_radps)
        + float(grid_yaw_correction_radps)
    )


def rotation_timeout_accepts_heading_error(
    final_heading_error_rad: float,
    accept_threshold_rad: float,
) -> bool:
    return abs(float(final_heading_error_rad)) <= max(0.0, float(accept_threshold_rad))


def grid_yaw_control_evidence_decision(
    *,
    source: str,
    valid: bool,
    yaw_valid: bool,
    confidence: float,
    yaw_error_rad: float,
    wall_valid: bool,
    wall_rms_error_m: float,
    wall_span_x_m: float,
    wall_support_count: int,
    require_strong_evidence: bool,
    allow_single_wall: bool,
    min_confidence_for_left_right: float,
    single_wall_min_confidence: float,
    single_wall_max_abs_yaw_error_rad: float,
    single_wall_max_rms_error_m: float,
    single_wall_min_span_x_m: float,
    single_wall_min_support_count: int,
) -> tuple[bool, str]:
    if not require_strong_evidence:
        return True, 'strong-evidence gate disabled'

    if not valid:
        return False, 'grid alignment invalid'

    if not yaw_valid:
        return False, 'grid yaw invalid'

    if source == 'left_right':
        if confidence < min_confidence_for_left_right:
            return (
                False,
                (
                    'left_right confidence too low: '
                    f'{confidence:.2f} < {min_confidence_for_left_right:.2f}'
                ),
            )
        return True, 'left_right adjacent-wall evidence accepted for active control'

    if source in ('left', 'right'):
        if not allow_single_wall:
            return (
                False,
                (
                    'single-wall active control disabled by default: '
                    f'source={source}, confidence={confidence:.2f}'
                ),
            )

        if not wall_valid:
            return False, f'{source} wall invalid'

        if confidence < single_wall_min_confidence:
            return (
                False,
                (
                    'single-wall confidence too low: '
                    f'{confidence:.2f} < {single_wall_min_confidence:.2f}'
                ),
            )

        abs_yaw = abs(float(yaw_error_rad))
        if abs_yaw > single_wall_max_abs_yaw_error_rad:
            return (
                False,
                (
                    'single-wall yaw error too large for active control: '
                    f'{abs_yaw:.3f} rad > '
                    f'{single_wall_max_abs_yaw_error_rad:.3f} rad'
                ),
            )

        if wall_rms_error_m > single_wall_max_rms_error_m:
            return (
                False,
                (
                    'single-wall rms too high for active control: '
                    f'{wall_rms_error_m:.3f} m > {single_wall_max_rms_error_m:.3f} m'
                ),
            )

        if wall_span_x_m < single_wall_min_span_x_m:
            return (
                False,
                (
                    'single-wall span too small for active control: '
                    f'{wall_span_x_m:.3f} m < {single_wall_min_span_x_m:.3f} m'
                ),
            )

        if wall_support_count < single_wall_min_support_count:
            return (
                False,
                (
                    'single-wall support too low for active control: '
                    f'{wall_support_count} < {single_wall_min_support_count}'
                ),
            )

        return (
            True,
            (
                'single-wall evidence accepted for active control: '
                f'source={source}, confidence={confidence:.2f}, '
                f'yaw={yaw_error_rad:.3f} rad, '
                f'rms={wall_rms_error_m:.3f} m, '
                f'span={wall_span_x_m:.3f} m, '
                f'count={wall_support_count}'
            ),
        )

    return False, f'unsupported grid yaw source for active control: {source}'


def side_wall_observation_from_alignment(
    alignment: GridAlignmentEstimate,
    *,
    preferred_side: str = 'none',
    min_confidence: float = 0.60,
    max_abs_yaw_error_rad: float = 0.100,
    max_rms_error_m: float = 0.020,
    min_span_x_m: float = 0.220,
    min_support_count: int = 80,
) -> SideWallObservation:
    if not alignment.valid or not alignment.yaw_valid:
        return invalid_side_wall_observation(
            f'grid alignment invalid for side-wall tracking: {alignment.reason}'
        )

    candidates: list[tuple[str, WallLineEstimate, float]] = []

    if alignment.source == 'left_right':
        if alignment.left.valid:
            candidates.append(('left', alignment.left, alignment.confidence))
        if alignment.right.valid:
            candidates.append(('right', alignment.right, alignment.confidence))
    elif alignment.source == 'left' and alignment.left.valid:
        candidates.append(('left', alignment.left, alignment.confidence))
    elif alignment.source == 'right' and alignment.right.valid:
        candidates.append(('right', alignment.right, alignment.confidence))

    if not candidates:
        return invalid_side_wall_observation(
            f'no adjacent side-wall candidate from source={alignment.source}: '
            f'{alignment.reason}'
        )

    if preferred_side in ('left', 'right'):
        preferred = [candidate for candidate in candidates if candidate[0] == preferred_side]
        if preferred:
            candidates = preferred

    best_side, best_wall, confidence = max(
        candidates,
        key=lambda item: (
            item[2],
            item[1].span_x_m,
            item[1].support_count,
            -item[1].rms_error_m,
        ),
    )

    if confidence < min_confidence:
        return invalid_side_wall_observation(
            f'side-wall confidence too low: {confidence:.2f} < {min_confidence:.2f}'
        )

    abs_yaw = abs(best_wall.yaw_error_rad)
    if abs_yaw > max_abs_yaw_error_rad:
        return invalid_side_wall_observation(
            f'side-wall yaw too large: {abs_yaw:.3f} rad > '
            f'{max_abs_yaw_error_rad:.3f} rad'
        )

    if best_wall.rms_error_m > max_rms_error_m:
        return invalid_side_wall_observation(
            f'side-wall rms too high: {best_wall.rms_error_m:.3f} m > '
            f'{max_rms_error_m:.3f} m'
        )

    if best_wall.span_x_m < min_span_x_m:
        return invalid_side_wall_observation(
            f'side-wall span too small: {best_wall.span_x_m:.3f} m < '
            f'{min_span_x_m:.3f} m'
        )

    if best_wall.support_count < min_support_count:
        return invalid_side_wall_observation(
            f'side-wall support too low: {best_wall.support_count} < '
            f'{min_support_count}'
        )

    return SideWallObservation(
        valid=True,
        side=best_side,
        offset_m=best_wall.offset_m,
        yaw_error_rad=best_wall.yaw_error_rad,
        rms_error_m=best_wall.rms_error_m,
        span_x_m=best_wall.span_x_m,
        support_count=best_wall.support_count,
        confidence=confidence,
        source=alignment.source,
        reason=(
            f'adjacent {best_side} wall accepted: confidence={confidence:.2f}, '
            f'yaw={best_wall.yaw_error_rad:.3f} rad, '
            f'offset={best_wall.offset_m:.3f} m, '
            f'rms={best_wall.rms_error_m:.3f} m, '
            f'span={best_wall.span_x_m:.3f} m, '
            f'count={best_wall.support_count}'
        ),
    )


def lidar_parallelity_correction_radps(
    *,
    yaw_error_rad: float,
    drift_per_m: float,
    k_yaw: float,
    k_drift: float,
    max_correction_radps: float,
    require_consistency: bool,
    max_yaw_drift_disagreement_rad: float,
) -> tuple[float, float, str]:
    if not math.isfinite(yaw_error_rad):
        yaw_error_rad = 0.0
    if not math.isfinite(drift_per_m):
        drift_per_m = 0.0

    drift_yaw_error_rad = math.atan(drift_per_m)

    if require_consistency:
        yaw_sign = 0 if abs(yaw_error_rad) < 1e-6 else (1 if yaw_error_rad > 0.0 else -1)
        drift_sign = (
            0
            if abs(drift_yaw_error_rad) < 1e-6
            else (1 if drift_yaw_error_rad > 0.0 else -1)
        )
        disagreement = abs(yaw_error_rad - drift_yaw_error_rad)

        if yaw_sign != 0 and drift_sign != 0 and yaw_sign != drift_sign:
            return (
                0.0,
                drift_yaw_error_rad,
                (
                    'yaw/drift sign disagreement: '
                    f'yaw={yaw_error_rad:.3f} rad, '
                    f'drift_yaw={drift_yaw_error_rad:.3f} rad'
                ),
            )

        if disagreement > max_yaw_drift_disagreement_rad:
            return (
                0.0,
                drift_yaw_error_rad,
                (
                    'yaw/drift magnitude disagreement: '
                    f'{disagreement:.3f} rad > '
                    f'{max_yaw_drift_disagreement_rad:.3f} rad'
                ),
            )

    fused_error = (float(k_yaw) * yaw_error_rad) + (float(k_drift) * drift_yaw_error_rad)
    limit = abs(float(max_correction_radps))
    correction = max(-limit, min(limit, fused_error))

    return (
        float(correction),
        float(drift_yaw_error_rad),
        (
            f'fused LiDAR parallelity correction: yaw={yaw_error_rad:.3f} rad, '
            f'drift_yaw={drift_yaw_error_rad:.3f} rad, '
            f'correction={correction:.3f} rad/s'
        ),
    )


def should_pre_align_grid_yaw(
    *,
    enabled: bool,
    direction: float,
    alignment_valid: bool,
    yaw_valid: bool,
    yaw_error_rad: float,
    confidence: float,
    min_confidence: float,
    start_threshold_rad: float,
    max_control_error_rad: float,
) -> tuple[bool, str]:
    if not enabled:
        return False, 'pre-translation grid-yaw alignment disabled'

    if float(direction) <= 0.0:
        return False, 'reverse translation does not use pre-alignment'

    if not alignment_valid or not yaw_valid:
        return False, 'no valid grid yaw'

    if float(confidence) < float(min_confidence):
        return (
            False,
            (
                f'grid confidence too low: {float(confidence):.2f} '
                f'< {float(min_confidence):.2f}'
            ),
        )

    abs_error = abs(float(yaw_error_rad))
    if abs_error > float(max_control_error_rad):
        return (
            False,
            (
                f'grid yaw error too large for pre-align control: '
                f'{float(yaw_error_rad):.3f} rad > '
                f'{float(max_control_error_rad):.3f} rad'
            ),
        )

    if abs_error <= float(start_threshold_rad):
        return (
            False,
            (
                f'start grid yaw within threshold: {float(yaw_error_rad):.3f} rad '
                f'<= {float(start_threshold_rad):.3f} rad'
            ),
        )

    return True, 'pre-translation grid-yaw alignment required'


def grid_yaw_pre_align_complete(
    *,
    alignment_valid: bool,
    yaw_valid: bool,
    yaw_error_rad: float,
    confidence: float,
    min_confidence: float,
    target_rad: float,
) -> tuple[bool, str]:
    if not alignment_valid or not yaw_valid:
        return False, 'no valid grid yaw'

    if float(confidence) < float(min_confidence):
        return (
            False,
            (
                f'grid confidence too low: {float(confidence):.2f} '
                f'< {float(min_confidence):.2f}'
            ),
        )

    abs_error = abs(float(yaw_error_rad))
    if abs_error > float(target_rad):
        return (
            False,
            (
                f'grid yaw outside target: {float(yaw_error_rad):.3f} rad '
                f'> {float(target_rad):.3f} rad'
            ),
        )

    return True, (
        f'grid yaw within target: {float(yaw_error_rad):.3f} rad '
        f'<= {float(target_rad):.3f} rad'
    )


def post_rotation_grid_yaw_refine_decision(
    *,
    yaw_error_rad: float,
    target_rad: float,
    start_threshold_rad: float,
) -> PostRotationGridYawRefineDecision:
    abs_error = abs(float(yaw_error_rad))
    target = float(target_rad)
    start = float(start_threshold_rad)

    if abs_error <= target:
        return PostRotationGridYawRefineDecision(
            stable=True,
            should_correct=False,
            reason=(
                f'post-rotation grid yaw within target: {float(yaw_error_rad):.3f} rad '
                f'<= {target:.3f} rad'
            ),
        )

    if abs_error <= start:
        return PostRotationGridYawRefineDecision(
            stable=True,
            should_correct=False,
            reason=(
                'post-rotation grid yaw within start threshold; '
                f'{float(yaw_error_rad):.3f} rad <= {start:.3f} rad; '
                'no correction needed'
            ),
        )

    return PostRotationGridYawRefineDecision(
        stable=False,
        should_correct=True,
        reason=(
            f'post-rotation grid yaw correction needed: {float(yaw_error_rad):.3f} rad '
            f'> {start:.3f} rad'
        ),
    )


def choose_heading_validation_error(
    odom_heading_error_rad: float,
    grid_yaw_error_rad: float,
    grid_yaw_valid: bool,
    grid_yaw_correction_used: bool,
    grid_alignment_confidence: float,
    min_grid_confidence: float,
    max_grid_yaw_abs_error_rad: float,
    grid_alignment_source: str = 'grid',
) -> Tuple[float, str]:
    grid_error = abs(float(grid_yaw_error_rad))
    if (
        grid_yaw_correction_used
        and grid_yaw_valid
        and float(grid_alignment_confidence) >= float(min_grid_confidence)
        and grid_error <= float(max_grid_yaw_abs_error_rad)
    ):
        source = str(grid_alignment_source) if grid_alignment_source else 'grid'
        return grid_error, f'grid_yaw/{source}'

    return abs(float(odom_heading_error_rad)), 'odom'


def grid_lateral_drift(
    start_valid: bool,
    start_error_m: float,
    end_valid: bool,
    end_error_m: float,
    progress_m: float,
) -> GridLateralDriftEstimate:
    if not start_valid:
        return GridLateralDriftEstimate(
            start_valid=False,
            start_error_m=0.0,
            end_valid=bool(end_valid),
            end_error_m=float(end_error_m) if end_valid else 0.0,
            drift_valid=False,
            drift_m=0.0,
            drift_per_m=0.0,
            reason='start lateral estimate invalid',
        )

    if not end_valid:
        return GridLateralDriftEstimate(
            start_valid=True,
            start_error_m=float(start_error_m),
            end_valid=False,
            end_error_m=0.0,
            drift_valid=False,
            drift_m=0.0,
            drift_per_m=0.0,
            reason='end lateral estimate invalid',
        )

    progress = max(float(progress_m), 0.05)
    drift = float(end_error_m) - float(start_error_m)
    return GridLateralDriftEstimate(
        start_valid=True,
        start_error_m=float(start_error_m),
        end_valid=True,
        end_error_m=float(end_error_m),
        drift_valid=True,
        drift_m=float(drift),
        drift_per_m=float(drift / progress),
        reason='valid grid lateral drift',
    )


def expected_side_offset_m(side: str, expected_half_width_m: float) -> float:
    if side == 'left':
        return float(expected_half_width_m)
    if side == 'right':
        return -float(expected_half_width_m)
    raise ValueError(f'Unsupported side {side!r}')


def wall_offset_error_m(
    wall: WallLineEstimate,
    expected_half_width_m: float,
) -> float:
    if not wall.valid:
        return 0.0
    return float(wall.offset_m) - expected_side_offset_m(
        wall.side,
        expected_half_width_m,
    )


def is_adjacent_wall_line(
    wall: WallLineEstimate,
    expected_half_width_m: float,
    adjacent_wall_tolerance_m: float,
) -> bool:
    return bool(
        wall.valid
        and abs(wall_offset_error_m(wall, expected_half_width_m))
        <= float(adjacent_wall_tolerance_m)
    )


def wall_quality_key(wall: WallLineEstimate) -> Tuple[float, int, float]:
    """Sort key for choosing the more reliable single-wall estimate.

    Lower RMS is best. If tied, more support and more x-span are better.
    """
    return (
        float(wall.rms_error_m),
        -int(wall.support_count),
        -float(wall.span_x_m),
    )


def choose_better_wall(
    left: WallLineEstimate,
    right: WallLineEstimate,
) -> WallLineEstimate:
    candidates = [wall for wall in (left, right) if wall.valid]
    if not candidates:
        return invalid_wall_line('none', 'no valid walls to choose from')
    return min(candidates, key=wall_quality_key)


def wall_adjacency_reason(
    wall: WallLineEstimate,
    expected_half_width_m: float,
    adjacent_wall_tolerance_m: float,
) -> str:
    if not wall.valid:
        return f'{wall.side}=invalid({wall.reason})'
    error = wall_offset_error_m(wall, expected_half_width_m)
    expected = expected_side_offset_m(wall.side, expected_half_width_m)
    return (
        f'{wall.side}=offset {wall.offset_m:.3f} m, '
        f'expected {expected:.3f} m, '
        f'error {error:.3f} m, '
        f'tolerance {float(adjacent_wall_tolerance_m):.3f} m'
    )


def estimate_grid_alignment(
    scan,
    expected_half_width_m: float,
    min_x_m: float,
    max_x_m: float,
    min_side_distance_m: float,
    max_side_distance_m: float,
    min_points: int,
    min_span_x_m: float,
    max_rms_error_m: float,
    max_abs_yaw_error_rad: float,
    max_reported_error_m: float,
    max_reported_yaw_rad: float,
    adjacent_wall_tolerance_m: float = 0.080,
    pair_width_tolerance_m: float = 0.080,
) -> GridAlignmentEstimate:
    if scan is None:
        return invalid_grid_alignment('no scan')

    left = fit_side_wall_line(
        scan,
        side='left',
        min_x_m=min_x_m,
        max_x_m=max_x_m,
        min_side_distance_m=min_side_distance_m,
        max_side_distance_m=max_side_distance_m,
        min_points=min_points,
        min_span_x_m=min_span_x_m,
        max_rms_error_m=max_rms_error_m,
        max_abs_yaw_error_rad=max_abs_yaw_error_rad,
    )
    right = fit_side_wall_line(
        scan,
        side='right',
        min_x_m=min_x_m,
        max_x_m=max_x_m,
        min_side_distance_m=min_side_distance_m,
        max_side_distance_m=max_side_distance_m,
        min_points=min_points,
        min_span_x_m=min_span_x_m,
        max_rms_error_m=max_rms_error_m,
        max_abs_yaw_error_rad=max_abs_yaw_error_rad,
    )

    valid_walls = [wall for wall in (left, right) if wall.valid]
    if not valid_walls:
        return GridAlignmentEstimate(
            valid=False,
            yaw_valid=False,
            yaw_error_rad=0.0,
            lateral_valid=False,
            lateral_error_m=0.0,
            source='none',
            confidence=0.0,
            left=left,
            right=right,
            reason=f'no valid wall lines: left={left.reason}; right={right.reason}',
        )

    left_adjacent = is_adjacent_wall_line(
        left,
        expected_half_width_m=expected_half_width_m,
        adjacent_wall_tolerance_m=adjacent_wall_tolerance_m,
    )
    right_adjacent = is_adjacent_wall_line(
        right,
        expected_half_width_m=expected_half_width_m,
        adjacent_wall_tolerance_m=adjacent_wall_tolerance_m,
    )

    if left_adjacent and right_adjacent:
        observed_width = float(left.offset_m) - float(right.offset_m)
        expected_width = 2.0 * float(expected_half_width_m)
        width_error = observed_width - expected_width
        pair_width_valid = abs(width_error) <= float(pair_width_tolerance_m)

        if pair_width_valid:
            yaw_error = (float(left.yaw_error_rad) + float(right.yaw_error_rad)) / 2.0
            lateral_error = (float(left.offset_m) + float(right.offset_m)) / 2.0
            source = 'left_right'
            confidence = 1.0
            reason = (
                'left and right adjacent wall lines valid: '
                f'observed_width={observed_width:.3f} m, '
                f'expected_width={expected_width:.3f} m'
            )
        else:
            chosen = choose_better_wall(left, right)
            yaw_error = float(chosen.yaw_error_rad)
            lateral_error = wall_offset_error_m(chosen, expected_half_width_m)
            source = chosen.side
            confidence = 0.6
            reason = (
                'left/right pair rejected by width; '
                f'observed_width={observed_width:.3f} m, '
                f'expected_width={expected_width:.3f} m, '
                f'width_error={width_error:.3f} m, '
                f'tolerance={float(pair_width_tolerance_m):.3f} m; '
                f'using adjacent {chosen.side} wall only'
            )

    elif left_adjacent:
        yaw_error = float(left.yaw_error_rad)
        lateral_error = wall_offset_error_m(left, expected_half_width_m)
        source = 'left'
        confidence = 0.6
        reason = (
            'using adjacent left wall only; '
            f'right not adjacent or invalid: '
            f'{wall_adjacency_reason(right, expected_half_width_m, adjacent_wall_tolerance_m)}'
        )

    elif right_adjacent:
        yaw_error = float(right.yaw_error_rad)
        lateral_error = wall_offset_error_m(right, expected_half_width_m)
        source = 'right'
        confidence = 0.6
        reason = (
            'using adjacent right wall only; '
            f'left not adjacent or invalid: '
            f'{wall_adjacency_reason(left, expected_half_width_m, adjacent_wall_tolerance_m)}'
        )

    else:
        return GridAlignmentEstimate(
            valid=False,
            yaw_valid=False,
            yaw_error_rad=0.0,
            lateral_valid=False,
            lateral_error_m=0.0,
            source='none',
            confidence=0.0,
            left=left,
            right=right,
            reason=(
                'no adjacent side walls: '
                f'{wall_adjacency_reason(left, expected_half_width_m, adjacent_wall_tolerance_m)}; '
                f'{wall_adjacency_reason(right, expected_half_width_m, adjacent_wall_tolerance_m)}'
            ),
        )

    yaw_error = max(
        -float(max_reported_yaw_rad),
        min(float(max_reported_yaw_rad), float(yaw_error)),
    )
    lateral_error = max(
        -float(max_reported_error_m),
        min(float(max_reported_error_m), float(lateral_error)),
    )

    return GridAlignmentEstimate(
        valid=True,
        yaw_valid=True,
        yaw_error_rad=float(yaw_error),
        lateral_valid=True,
        lateral_error_m=float(lateral_error),
        source=source,
        confidence=float(confidence),
        left=left,
        right=right,
        reason=reason,
    )


def valid_ranges_in_sector(scan, center_angle_rad: float, width_deg: float) -> List[float]:
    if scan is None:
        return []

    half_width = math.radians(float(width_deg)) / 2.0
    center = normalize_angle(float(center_angle_rad))

    values: List[float] = []
    angle = float(scan.angle_min)

    for raw in scan.ranges:
        value = float(raw)
        delta = normalize_angle(angle - center)

        if abs(delta) <= half_width:
            if math.isfinite(value) and scan.range_min <= value <= scan.range_max:
                values.append(value)

        angle += float(scan.angle_increment)

    return values


def sector_range(
    scan,
    name: str,
    center_angle_rad: float,
    width_deg: float,
    min_samples: int = 3,
) -> SectorRange:
    values = valid_ranges_in_sector(scan, center_angle_rad, width_deg)

    if len(values) < int(min_samples):
        return SectorRange(
            name=name,
            center_angle_rad=float(center_angle_rad),
            width_deg=float(width_deg),
            valid=False,
            count=len(values),
            min_m=0.0,
            median_m=0.0,
            mean_m=0.0,
            max_m=0.0,
        )

    return SectorRange(
        name=name,
        center_angle_rad=float(center_angle_rad),
        width_deg=float(width_deg),
        valid=True,
        count=len(values),
        min_m=min(values),
        median_m=float(median(values)),
        mean_m=float(sum(values) / len(values)),
        max_m=max(values),
    )


def cardinal_sector_ranges(
    scan,
    width_deg: float,
    min_samples: int = 3,
) -> Dict[str, SectorRange]:
    return {
        'front': sector_range(scan, 'front', 0.0, width_deg, min_samples),
        'left': sector_range(scan, 'left', math.pi / 2.0, width_deg, min_samples),
        'right': sector_range(scan, 'right', -math.pi / 2.0, width_deg, min_samples),
        'rear': sector_range(scan, 'rear', math.pi, width_deg, min_samples),
    }


def finite_median_or_nan(measurement: SectorRange) -> float:
    return float(measurement.median_m) if measurement.valid else float('nan')


def choose_lidar_progress(
    front_valid: bool,
    front_progress_m: float,
    rear_valid: bool,
    rear_progress_m: float,
    max_disagreement_m: float,
    min_progress_m: float,
    allow_single_source: bool,
) -> LidarProgressEstimate:
    def rejection_reason(
        name: str,
        valid: bool,
        progress_m: float,
    ) -> str:
        if not valid:
            return f'{name} invalid'
        if not math.isfinite(progress_m):
            return f'{name} progress non-finite'
        if progress_m < min_progress_m:
            return (
                f'{name} progress {float(progress_m):.3f} m '
                f'< {float(min_progress_m):.3f} m'
            )
        return f'{name} progress valid'

    front_ok = (
        front_valid
        and math.isfinite(front_progress_m)
        and front_progress_m >= min_progress_m
    )
    rear_ok = (
        rear_valid
        and math.isfinite(rear_progress_m)
        and rear_progress_m >= min_progress_m
    )
    front_rejection = rejection_reason(
        'front',
        front_valid,
        front_progress_m,
    )
    rear_rejection = rejection_reason(
        'rear',
        rear_valid,
        rear_progress_m,
    )

    if front_ok and rear_ok:
        disagreement = abs(float(front_progress_m) - float(rear_progress_m))
        if disagreement > float(max_disagreement_m):
            return LidarProgressEstimate(
                valid=False,
                progress_m=0.0,
                source='front_rear_rejected',
                disagreement_m=float(disagreement),
                reason=(
                    f'front/rear progress disagreement {disagreement:.3f} m '
                    f'> {float(max_disagreement_m):.3f} m'
                ),
            )

        return LidarProgressEstimate(
            valid=True,
            progress_m=float((float(front_progress_m) + float(rear_progress_m)) / 2.0),
            source='front_rear',
            disagreement_m=float(disagreement),
            reason='front and rear progress consistent',
        )

    if not allow_single_source:
        return LidarProgressEstimate(
            valid=False,
            progress_m=0.0,
            source='none',
            disagreement_m=0.0,
            reason=(
                'single-source LiDAR progress disabled: '
                f'front={front_rejection}; rear={rear_rejection}'
            ),
        )

    if front_ok:
        return LidarProgressEstimate(
            valid=True,
            progress_m=float(front_progress_m),
            source='front',
            disagreement_m=0.0,
            reason=f'front progress valid; rear rejected: {rear_rejection}',
        )

    if rear_ok:
        return LidarProgressEstimate(
            valid=True,
            progress_m=float(rear_progress_m),
            source='rear',
            disagreement_m=0.0,
            reason=f'rear progress valid; front rejected: {front_rejection}',
        )

    return LidarProgressEstimate(
        valid=False,
        progress_m=0.0,
        source='none',
        disagreement_m=0.0,
        reason=(
            'no valid LiDAR progress source: '
            f'front={front_rejection}; rear={rear_rejection}'
        ),
    )


def choose_temporal_lidar_progress(
    *,
    raw: LidarProgressEstimate,
    previous_valid: bool,
    previous_progress_m: float,
    previous_source: str,
    front_valid: bool,
    front_progress_m: float,
    rear_valid: bool,
    rear_progress_m: float,
    degraded_samples: int,
    max_backtrack_m: float,
    max_jump_m: float,
    max_degraded_samples: int,
) -> TemporalLidarProgressEstimate:
    if raw.valid:
        return TemporalLidarProgressEstimate(
            valid=True,
            progress_m=raw.progress_m,
            source=raw.source,
            degraded=False,
            degraded_samples=0,
            disagreement_m=raw.disagreement_m,
            reason=f'raw LiDAR progress accepted: {raw.reason}',
        )

    if not previous_valid:
        return TemporalLidarProgressEstimate(
            valid=False,
            progress_m=0.0,
            source='none',
            degraded=False,
            degraded_samples=degraded_samples,
            disagreement_m=raw.disagreement_m,
            reason=f'no previous LiDAR progress for temporal recovery: {raw.reason}',
        )

    if degraded_samples >= max_degraded_samples:
        return TemporalLidarProgressEstimate(
            valid=False,
            progress_m=previous_progress_m,
            source='temporal_rejected',
            degraded=True,
            degraded_samples=degraded_samples,
            disagreement_m=raw.disagreement_m,
            reason=(
                f'temporal LiDAR recovery exhausted: {degraded_samples} >= '
                f'{max_degraded_samples}; raw={raw.reason}'
            ),
        )

    candidates: list[tuple[str, float]] = []
    if front_valid and math.isfinite(front_progress_m):
        candidates.append(('front_temporal', float(front_progress_m)))
    if rear_valid and math.isfinite(rear_progress_m):
        candidates.append(('rear_temporal', float(rear_progress_m)))

    accepted: list[tuple[str, float, float]] = []
    for source, progress in candidates:
        backtrack = float(previous_progress_m) - progress
        jump = abs(progress - float(previous_progress_m))

        if backtrack > float(max_backtrack_m):
            continue
        if jump > float(max_jump_m):
            continue

        source_bonus = 0.0
        if str(previous_source).startswith('front') and source.startswith('front'):
            source_bonus = -0.001
        if str(previous_source).startswith('rear') and source.startswith('rear'):
            source_bonus = -0.001

        accepted.append((source, progress, jump + source_bonus))

    if not accepted:
        return TemporalLidarProgressEstimate(
            valid=False,
            progress_m=previous_progress_m,
            source='temporal_unavailable',
            degraded=True,
            degraded_samples=degraded_samples + 1,
            disagreement_m=raw.disagreement_m,
            reason=(
                'no temporal LiDAR candidate passed monotonic/jump gates; '
                f'previous={previous_progress_m:.3f} m; raw={raw.reason}'
            ),
        )

    accepted.sort(key=lambda item: item[2])
    source, progress, _ = accepted[0]

    return TemporalLidarProgressEstimate(
        valid=True,
        progress_m=progress,
        source=source,
        degraded=True,
        degraded_samples=degraded_samples + 1,
        disagreement_m=raw.disagreement_m,
        reason=(
            f'temporal LiDAR recovery using {source}: progress={progress:.3f} m, '
            f'previous={previous_progress_m:.3f} m; raw={raw.reason}'
        ),
    )


def choose_translation_progress(
    odom_progress_m: float,
    lidar_estimate: LidarProgressEstimate,
    mode: str,
    max_lidar_ahead_of_odom_m: float,
) -> TranslationProgressSelection:
    odom_progress = max(0.0, float(odom_progress_m))

    if mode == 'odom_only':
        return TranslationProgressSelection(
            valid=True,
            progress_m=odom_progress,
            source='odom',
            reason='translation_progress_source=odom_only',
        )

    if lidar_estimate.valid:
        lidar_progress = max(0.0, float(lidar_estimate.progress_m))
        if lidar_progress > odom_progress + float(max_lidar_ahead_of_odom_m):
            reason = (
                f'LiDAR progress {lidar_progress:.3f} m is ahead of odom '
                f'{odom_progress:.3f} m by more than '
                f'{float(max_lidar_ahead_of_odom_m):.3f} m'
            )

            if mode == 'lidar_required':
                return TranslationProgressSelection(
                    valid=False,
                    progress_m=0.0,
                    source='lidar_required_unavailable',
                    reason=(
                        'LiDAR progress required but rejected: '
                        f'{reason}; odom progress {odom_progress:.3f} m ignored'
                    ),
                )

            return TranslationProgressSelection(
                valid=True,
                progress_m=odom_progress,
                source='odom',
                reason='LiDAR rejected by odom sanity bound; falling back to odom: '
                + reason,
            )

        return TranslationProgressSelection(
            valid=True,
            progress_m=lidar_progress,
            source='lidar',
            reason=f'using LiDAR progress source={lidar_estimate.source}',
        )

    if mode == 'lidar_required':
        return TranslationProgressSelection(
            valid=False,
            progress_m=0.0,
            source='lidar_required_unavailable',
            reason=(
                'LiDAR progress required but unavailable/inconsistent: '
                f'{lidar_estimate.reason}; '
                f'odom progress {odom_progress:.3f} m ignored'
            ),
        )

    return TranslationProgressSelection(
        valid=True,
        progress_m=odom_progress,
        source='odom',
        reason='LiDAR progress unavailable or inconsistent; falling back to odom: '
        + lidar_estimate.reason,
    )
