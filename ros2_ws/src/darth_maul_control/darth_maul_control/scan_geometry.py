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

    if left.valid and right.valid:
        yaw_error = (left.yaw_error_rad + right.yaw_error_rad) / 2.0
        lateral_error = (left.offset_m + right.offset_m) / 2.0
        source = 'left_right'
        confidence = 1.0
        reason = 'left and right wall lines valid'
    elif left.valid:
        yaw_error = left.yaw_error_rad
        lateral_error = left.offset_m - float(expected_half_width_m)
        source = 'left'
        confidence = 0.6
        reason = 'left wall line valid'
    else:
        yaw_error = right.yaw_error_rad
        lateral_error = right.offset_m + float(expected_half_width_m)
        source = 'right'
        confidence = 0.6
        reason = 'right wall line valid'

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
            reason='single-source LiDAR progress disabled',
        )

    if front_ok:
        return LidarProgressEstimate(
            valid=True,
            progress_m=float(front_progress_m),
            source='front',
            disagreement_m=0.0,
            reason='front progress valid',
        )

    if rear_ok:
        return LidarProgressEstimate(
            valid=True,
            progress_m=float(rear_progress_m),
            source='rear',
            disagreement_m=0.0,
            reason='rear progress valid',
        )

    return LidarProgressEstimate(
        valid=False,
        progress_m=0.0,
        source='none',
        disagreement_m=0.0,
        reason='no valid LiDAR progress source',
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
                    progress_m=odom_progress,
                    source='none',
                    reason=reason,
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
            progress_m=odom_progress,
            source='none',
            reason='LiDAR progress required but invalid: ' + lidar_estimate.reason,
        )

    return TranslationProgressSelection(
        valid=True,
        progress_m=odom_progress,
        source='odom',
        reason='LiDAR progress unavailable or inconsistent; falling back to odom: '
        + lidar_estimate.reason,
    )
