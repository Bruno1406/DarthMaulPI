import math


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle_rad: float) -> float:
    if not math.isfinite(angle_rad):
        return 0.0

    normalized = (angle_rad + math.pi) % (2.0 * math.pi) - math.pi
    if normalized <= -math.pi:
        return normalized + 2.0 * math.pi
    return normalized


def clamp(value: float, low: float, high: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(low, min(high, value))


def sign(value: float) -> float:
    return 1.0 if value >= 0.0 else -1.0


def is_finite_pose_stamped(pose_stamped) -> bool:
    pose = pose_stamped.pose
    values = (
        pose.position.x,
        pose.position.y,
        pose.position.z,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    return all(math.isfinite(value) for value in values)
