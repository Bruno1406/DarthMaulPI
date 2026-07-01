from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Dict, List

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
