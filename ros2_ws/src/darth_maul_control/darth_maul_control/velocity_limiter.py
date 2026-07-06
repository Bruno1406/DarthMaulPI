from dataclasses import dataclass
import math

from geometry_msgs.msg import Twist


@dataclass(frozen=True)
class VelocityLimits:
    max_linear_x_mps: float
    max_linear_y_mps: float
    max_angular_z_radps: float


class VelocityLimiter:
    def __init__(
        self,
        max_linear_x_mps: float,
        max_linear_y_mps: float,
        max_angular_z_radps: float,
    ):
        self.hard_limits = VelocityLimits(
            max_linear_x_mps=self._clean_limit(max_linear_x_mps),
            max_linear_y_mps=self._clean_limit(max_linear_y_mps),
            max_angular_z_radps=self._clean_limit(max_angular_z_radps),
        )

    def sanitize_goal_limits(
        self,
        max_linear_x_mps: float,
        max_linear_y_mps: float,
        max_angular_z_radps: float,
    ) -> VelocityLimits:
        return VelocityLimits(
            max_linear_x_mps=self._goal_limit(
                max_linear_x_mps,
                self.hard_limits.max_linear_x_mps,
            ),
            max_linear_y_mps=self._goal_limit(
                max_linear_y_mps,
                self.hard_limits.max_linear_y_mps,
            ),
            max_angular_z_radps=self._goal_limit(
                max_angular_z_radps,
                self.hard_limits.max_angular_z_radps,
            ),
        )

    def clamp(self, twist: Twist, limits: VelocityLimits | None = None) -> Twist:
        active = limits or self.hard_limits

        bounded = Twist()
        bounded.linear.x = self._clamp_axis(twist.linear.x, active.max_linear_x_mps)
        bounded.linear.y = self._clamp_axis(twist.linear.y, active.max_linear_y_mps)
        bounded.linear.z = 0.0
        bounded.angular.x = 0.0
        bounded.angular.y = 0.0
        bounded.angular.z = self._clamp_axis(twist.angular.z, active.max_angular_z_radps)
        return bounded

    @staticmethod
    def _clean_limit(value: float) -> float:
        if not math.isfinite(value):
            return 0.0
        return abs(float(value))

    @staticmethod
    def _goal_limit(value: float, hard_limit: float) -> float:
        if not math.isfinite(value) or value <= 0.0:
            return hard_limit
        return min(abs(float(value)), hard_limit)

    @staticmethod
    def _clamp_axis(value: float, limit: float) -> float:
        if not math.isfinite(value):
            return 0.0
        return max(-limit, min(limit, float(value)))
