from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Dict, List, Tuple

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


def combine_lidar_progress_candidates(
    front_valid: bool,
    front_progress_m: float,
    rear_valid: bool,
    rear_progress_m: float,
) -> Tuple[bool, float]:
    candidates = []
    if front_valid and math.isfinite(front_progress_m):
        candidates.append(float(front_progress_m))
    if rear_valid and math.isfinite(rear_progress_m):
        candidates.append(float(rear_progress_m))

    if not candidates:
        return False, 0.0

    return True, float(sum(candidates) / len(candidates))
