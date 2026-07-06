from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class ScanPoint:
    index: int
    x: float
    y: float
    range_m: float


@dataclass(frozen=True)
class LineSegmentEstimate:
    valid: bool
    angle_rad: float
    length_m: float
    support_count: int
    rms_error_m: float
    centroid_x_m: float
    centroid_y_m: float
    weight: float
    reason: str


@dataclass(frozen=True)
class GridYawObservation:
    valid: bool
    yaw_error_rad: float
    confidence: float
    source: str
    line_count: int
    dominant_axis_rad: float
    total_weight: float
    concentration: float
    reason: str


def normalize_angle(angle: float) -> float:
    value = math.fmod(float(angle) + math.pi, 2.0 * math.pi)
    if value < 0.0:
        value += 2.0 * math.pi
    return value - math.pi


def normalize_line_angle(angle: float) -> float:
    """Normalize undirected line orientation into [-pi/2, pi/2)."""
    value = normalize_angle(float(angle))
    if value >= math.pi / 2.0:
        value -= math.pi
    if value < -math.pi / 2.0:
        value += math.pi
    return value


def normalize_manhattan_axis_error(angle: float) -> float:
    """Return line angle error to nearest Manhattan axis, period pi/2."""
    value = math.fmod(float(angle) + math.pi / 4.0, math.pi / 2.0)
    if value < 0.0:
        value += math.pi / 2.0
    return value - math.pi / 4.0


def invalid_grid_yaw(reason: str) -> GridYawObservation:
    return GridYawObservation(
        valid=False,
        yaw_error_rad=0.0,
        confidence=0.0,
        source='none',
        line_count=0,
        dominant_axis_rad=0.0,
        total_weight=0.0,
        concentration=0.0,
        reason=reason,
    )


def scan_points(
    scan,
    *,
    min_range_m: float,
    max_range_m: float,
) -> list[ScanPoint]:
    if scan is None:
        return []

    points: list[ScanPoint] = []
    angle = float(scan.angle_min)

    range_min = max(float(min_range_m), float(getattr(scan, 'range_min', 0.0)))
    range_max = min(float(max_range_m), float(getattr(scan, 'range_max', max_range_m)))

    for index, raw in enumerate(scan.ranges):
        value = float(raw)
        if math.isfinite(value) and range_min <= value <= range_max:
            points.append(
                ScanPoint(
                    index=index,
                    x=value * math.cos(angle),
                    y=value * math.sin(angle),
                    range_m=value,
                )
            )
        angle += float(scan.angle_increment)

    return points


def split_scan_clusters(
    points: Sequence[ScanPoint],
    *,
    max_point_gap_m: float,
    max_range_jump_m: float,
    min_cluster_points: int,
) -> list[list[ScanPoint]]:
    clusters: list[list[ScanPoint]] = []
    current: list[ScanPoint] = []

    gap_limit = max(0.001, float(max_point_gap_m))
    jump_limit = max(0.001, float(max_range_jump_m))

    def finish() -> None:
        nonlocal current
        if len(current) >= int(min_cluster_points):
            clusters.append(current)
        current = []

    previous: ScanPoint | None = None

    for point in points:
        if previous is None:
            current = [point]
            previous = point
            continue

        index_gap = point.index - previous.index
        euclidean_gap = math.hypot(point.x - previous.x, point.y - previous.y)
        range_jump = abs(point.range_m - previous.range_m)

        if index_gap != 1 or euclidean_gap > gap_limit or range_jump > jump_limit:
            finish()
            current = [point]
        else:
            current.append(point)

        previous = point

    finish()
    return clusters


def fit_line_segment_from_xy(
    points_xy: Sequence[tuple[float, float]],
    *,
    min_segment_points: int,
    min_segment_length_m: float,
    max_line_rms_m: float,
) -> LineSegmentEstimate:
    n = len(points_xy)
    if n < int(min_segment_points):
        return LineSegmentEstimate(
            False,
            0.0,
            0.0,
            n,
            float('inf'),
            0.0,
            0.0,
            0.0,
            f'not enough points: {n}',
        )

    cx = sum(x for x, _ in points_xy) / n
    cy = sum(y for _, y in points_xy) / n

    sxx = sum((x - cx) * (x - cx) for x, _ in points_xy) / n
    syy = sum((y - cy) * (y - cy) for _, y in points_xy) / n
    sxy = sum((x - cx) * (y - cy) for x, y in points_xy) / n

    if not all(math.isfinite(value) for value in (sxx, syy, sxy)):
        return LineSegmentEstimate(
            False, 0.0, 0.0, n, float('inf'), cx, cy, 0.0, 'non-finite covariance'
        )

    if sxx + syy <= 1.0e-12:
        return LineSegmentEstimate(
            False, 0.0, 0.0, n, float('inf'), cx, cy, 0.0, 'degenerate cluster'
        )

    angle = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    angle = normalize_line_angle(angle)

    ux = math.cos(angle)
    uy = math.sin(angle)
    vx = -uy
    vy = ux

    projections = [(x - cx) * ux + (y - cy) * uy for x, y in points_xy]
    length = max(projections) - min(projections)

    residuals = [(x - cx) * vx + (y - cy) * vy for x, y in points_xy]
    rms = math.sqrt(sum(value * value for value in residuals) / n)

    if length < float(min_segment_length_m):
        return LineSegmentEstimate(
            False,
            angle,
            length,
            n,
            rms,
            cx,
            cy,
            0.0,
            f'segment too short: {length:.3f}',
        )

    if rms > float(max_line_rms_m):
        return LineSegmentEstimate(
            False,
            angle,
            length,
            n,
            rms,
            cx,
            cy,
            0.0,
            f'line rms too high: {rms:.3f}',
        )

    rms_floor = max(0.002, float(max_line_rms_m) * 0.25)
    clean_factor = max(0.1, min(1.0, float(max_line_rms_m) / max(rms, rms_floor)))
    weight = float(length) * math.sqrt(float(n)) * clean_factor

    return LineSegmentEstimate(
        True,
        angle,
        length,
        n,
        rms,
        cx,
        cy,
        weight,
        'valid line segment',
    )


def fit_line_segment(
    cluster: Sequence[ScanPoint],
    *,
    min_segment_points: int,
    min_segment_length_m: float,
    max_line_rms_m: float,
) -> LineSegmentEstimate:
    return fit_line_segment_from_xy(
        [(point.x, point.y) for point in cluster],
        min_segment_points=min_segment_points,
        min_segment_length_m=min_segment_length_m,
        max_line_rms_m=max_line_rms_m,
    )


def estimate_manhattan_grid_yaw_from_segments(
    segments: Sequence[LineSegmentEstimate],
    *,
    min_line_count: int,
    min_total_weight: float,
    min_concentration: float,
    max_abs_yaw_error_rad: float,
) -> GridYawObservation:
    valid_segments = [segment for segment in segments if segment.valid and segment.weight > 0.0]

    if len(valid_segments) < int(min_line_count):
        return invalid_grid_yaw(f'not enough valid lines: {len(valid_segments)}')

    total_weight = sum(segment.weight for segment in valid_segments)
    if total_weight < float(min_total_weight):
        return invalid_grid_yaw(f'total line weight too low: {total_weight:.3f}')

    sum_x = sum(segment.weight * math.cos(4.0 * segment.angle_rad) for segment in valid_segments)
    sum_y = sum(segment.weight * math.sin(4.0 * segment.angle_rad) for segment in valid_segments)

    resultant = math.hypot(sum_x, sum_y)
    concentration = resultant / max(total_weight, 1.0e-9)

    if concentration < float(min_concentration):
        return invalid_grid_yaw(
            f'manhattan orientation concentration too low: {concentration:.3f}'
        )

    dominant_axis = normalize_manhattan_axis_error(0.25 * math.atan2(sum_y, sum_x))
    yaw_error = dominant_axis

    if abs(yaw_error) > float(max_abs_yaw_error_rad):
        return invalid_grid_yaw(
            f'manhattan yaw error too large: {yaw_error:.3f}'
        )

    confidence = max(
        0.0,
        min(
            1.0,
            concentration
            * min(1.0, total_weight / max(float(min_total_weight), 1.0e-9)),
        ),
    )

    return GridYawObservation(
        valid=True,
        yaw_error_rad=float(yaw_error),
        confidence=float(confidence),
        source='manhattan_lines',
        line_count=len(valid_segments),
        dominant_axis_rad=float(dominant_axis),
        total_weight=float(total_weight),
        concentration=float(concentration),
        reason=(
            f'valid manhattan yaw from {len(valid_segments)} lines; '
            f'axis={dominant_axis:.3f}; confidence={confidence:.2f}; '
            f'weight={total_weight:.3f}; concentration={concentration:.3f}'
        ),
    )


def estimate_manhattan_grid_yaw(
    scan,
    *,
    min_range_m: float,
    max_range_m: float,
    max_point_gap_m: float,
    max_range_jump_m: float,
    min_cluster_points: int,
    min_segment_points: int,
    min_segment_length_m: float,
    max_line_rms_m: float,
    min_line_count: int,
    min_total_weight: float,
    min_concentration: float,
    max_abs_yaw_error_rad: float,
) -> GridYawObservation:
    points = scan_points(
        scan,
        min_range_m=min_range_m,
        max_range_m=max_range_m,
    )
    if not points:
        return invalid_grid_yaw('no finite scan points')

    clusters = split_scan_clusters(
        points,
        max_point_gap_m=max_point_gap_m,
        max_range_jump_m=max_range_jump_m,
        min_cluster_points=min_cluster_points,
    )
    if not clusters:
        return invalid_grid_yaw('no scan clusters')

    segments = [
        fit_line_segment(
            cluster,
            min_segment_points=min_segment_points,
            min_segment_length_m=min_segment_length_m,
            max_line_rms_m=max_line_rms_m,
        )
        for cluster in clusters
    ]

    return estimate_manhattan_grid_yaw_from_segments(
        segments,
        min_line_count=min_line_count,
        min_total_weight=min_total_weight,
        min_concentration=min_concentration,
        max_abs_yaw_error_rad=max_abs_yaw_error_rad,
    )
