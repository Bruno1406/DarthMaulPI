from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import threading
import time
from typing import Optional

from darth_maul_control.geometry import (
    clamp,
    is_finite_pose_stamped,
    normalize_angle,
    sign,
    yaw_from_quaternion,
)
from darth_maul_control.scan_geometry import (
    SectorRange,
    cardinal_sector_ranges,
    combine_lidar_progress_candidates,
    finite_median_or_nan,
)
from darth_maul_control.velocity_limiter import VelocityLimiter, VelocityLimits
from darth_maul_control_interfaces.action import ExecuteMotionPrimitive
from darth_maul_control_interfaces.msg import ControlStatus
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Trigger


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass
class MotionSnapshot:
    pose: Pose2D
    pose_msg: PoseStamped


@dataclass(frozen=True)
class LidarRangeSnapshot:
    front: SectorRange
    rear: SectorRange
    left: SectorRange
    right: SectorRange


@dataclass(frozen=True)
class TranslationDiagnostics:
    odom_progress_m: float = 0.0

    front_range_valid: bool = False
    front_range_start_m: float = 0.0
    front_range_end_m: float = 0.0
    front_progress_m: float = 0.0

    rear_range_valid: bool = False
    rear_range_start_m: float = 0.0
    rear_range_end_m: float = 0.0
    rear_progress_m: float = 0.0

    left_range_valid: bool = False
    left_range_start_m: float = 0.0
    left_range_end_m: float = 0.0

    right_range_valid: bool = False
    right_range_start_m: float = 0.0
    right_range_end_m: float = 0.0

    lidar_progress_valid: bool = False
    lidar_progress_m: float = 0.0
    lidar_minus_odom_m: float = 0.0


SUPPORTED_PRIMITIVES = {
    ExecuteMotionPrimitive.Goal.DRIVE_FORWARD,
    ExecuteMotionPrimitive.Goal.DRIVE_BACKWARD,
    ExecuteMotionPrimitive.Goal.ROTATE_RELATIVE,
    ExecuteMotionPrimitive.Goal.STOP,
    ExecuteMotionPrimitive.Goal.WAIT,
    ExecuteMotionPrimitive.Goal.ADVANCE_CELL,
}

PRIMITIVE_NAMES = {
    ExecuteMotionPrimitive.Goal.DRIVE_FORWARD: 'DRIVE_FORWARD',
    ExecuteMotionPrimitive.Goal.DRIVE_BACKWARD: 'DRIVE_BACKWARD',
    ExecuteMotionPrimitive.Goal.ROTATE_RELATIVE: 'ROTATE_RELATIVE',
    ExecuteMotionPrimitive.Goal.STOP: 'STOP',
    ExecuteMotionPrimitive.Goal.WAIT: 'WAIT',
    ExecuteMotionPrimitive.Goal.ADVANCE_CELL: 'ADVANCE_CELL',
}


class DarthMaulControlNode(Node):
    def __init__(self):
        super().__init__('darth_maul_control')

        self._callback_group = ReentrantCallbackGroup()

        self._declare_and_load_params()

        self._limiter = VelocityLimiter(
            self.max_linear_x_mps,
            self.max_linear_y_mps,
            self.max_angular_z_radps,
        )

        self._state_lock = threading.RLock()
        self._odom_lock = threading.RLock()
        self._scan_lock = threading.RLock()
        self._motion_lock = threading.RLock()

        self._state = ControlStatus.STATE_IDLE
        self._status = 'ready'
        self._command_enabled = True
        self._active_primitive_type = 0
        self._distance_remaining_m = 0.0
        self._distance_traveled_m = 0.0
        self._heading_error_rad = 0.0
        self._front_clearance_m = float('inf')
        self._front_range_m = float('nan')
        self._rear_range_m = float('nan')
        self._left_range_m = float('nan')
        self._right_range_m = float('nan')

        self._active_goal = False
        self._stop_requested = False
        self._cancel_requested = False

        self._current_odom: Optional[Odometry] = None
        self._last_odom_monotonic: Optional[float] = None

        self._latest_scan: Optional[LaserScan] = None
        self._last_scan_monotonic: Optional[float] = None

        self._last_commanded_twist = Twist()
        self._last_command_time = time.monotonic()

        self._cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self._status_pub = self.create_publisher(ControlStatus, self.status_topic, 10)

        self._odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self._odom_callback,
            10,
            callback_group=self._callback_group,
        )

        self._scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self._scan_callback,
            qos_profile_sensor_data,
            callback_group=self._callback_group,
        )

        self._stop_srv = self.create_service(
            Trigger,
            '/darth_maul_control/stop',
            self._handle_stop,
            callback_group=self._callback_group,
        )

        self._action_server = ActionServer(
            self,
            ExecuteMotionPrimitive,
            '/darth_maul_control/execute_motion_primitive',
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._callback_group,
        )

        self._status_timer = self.create_timer(
            0.5,
            self._publish_status,
            callback_group=self._callback_group,
        )

        self.get_logger().info('darth_maul_control direct primitive controller ready.')
        self.get_logger().warn(
            'Do not run joystick_control or teleop_key_control while this node is active.'
        )
        self._publish_zero_for_duration()

    def _declare_and_load_params(self) -> None:
        self.odom_topic = self._string_param('odom_topic', '/odom')
        self.scan_topic = self._string_param('scan_topic', '/ldlidar_node/scan')
        self.cmd_vel_topic = self._string_param('cmd_vel_topic', '/controller/cmd_vel')
        self.status_topic = self._string_param(
            'status_topic',
            '/darth_maul_control/status',
        )

        self.cell_length_m = self._positive_float_param('cell_length_m', 0.254)
        self.control_rate_hz = self._positive_float_param('control_rate_hz', 25.0)

        self.default_position_tolerance_m = self._positive_float_param(
            'default_position_tolerance_m',
            0.025,
        )
        self.default_heading_tolerance_rad = self._positive_float_param(
            'default_heading_tolerance_rad',
            0.06,
        )

        self.default_linear_speed_mps = self._positive_float_param(
            'default_linear_speed_mps',
            0.070,
        )
        self.default_reverse_speed_mps = self._positive_float_param(
            'default_reverse_speed_mps',
            0.050,
        )
        self.default_angular_speed_radps = self._positive_float_param(
            'default_angular_speed_radps',
            0.280,
        )

        self.max_linear_x_mps = self._positive_float_param('max_linear_x_mps', 0.090)
        self.max_linear_y_mps = self._nonnegative_float_param(
            'max_linear_y_mps',
            0.035,
        )
        self.max_angular_z_radps = self._positive_float_param(
            'max_angular_z_radps',
            0.350,
        )

        self.min_linear_x_mps = self._nonnegative_float_param(
            'min_linear_x_mps',
            0.025,
        )
        self.min_angular_z_radps = self._nonnegative_float_param(
            'min_angular_z_radps',
            0.080,
        )

        self.k_distance = self._nonnegative_float_param('k_distance', 0.80)
        self.k_heading = self._nonnegative_float_param('k_heading', 1.60)

        self.max_linear_accel_mps2 = self._positive_float_param(
            'max_linear_accel_mps2',
            0.18,
        )
        self.max_lateral_accel_mps2 = self._positive_float_param(
            'max_lateral_accel_mps2',
            0.12,
        )
        self.max_angular_accel_radps2 = self._positive_float_param(
            'max_angular_accel_radps2',
            0.75,
        )

        self.odom_timeout_sec = self._positive_float_param('odom_timeout_sec', 0.50)
        self.scan_timeout_sec = self._positive_float_param('scan_timeout_sec', 0.50)

        self.require_scan_for_forward = self._bool_param(
            'require_scan_for_forward',
            True,
        )
        self.front_sector_deg = self._positive_float_param('front_sector_deg', 35.0)
        self.front_stop_distance_m = self._positive_float_param(
            'front_stop_distance_m',
            0.13,
        )
        self.lidar_diagnostics_enabled = self._bool_param(
            'lidar_diagnostics_enabled',
            True,
        )
        self.lidar_diagnostic_sector_width_deg = self._positive_float_param(
            'lidar_diagnostic_sector_width_deg',
            10.0,
        )
        self.lidar_diagnostic_min_samples = int(
            self._positive_float_param(
                'lidar_diagnostic_min_samples',
                3.0,
            )
        )
        self.lidar_diagnostic_min_samples = max(1, self.lidar_diagnostic_min_samples)

        self.wall_centering_enabled = self._bool_param(
            'wall_centering_enabled',
            False,
        )
        self.side_sector_center_deg = self._positive_float_param(
            'side_sector_center_deg',
            90.0,
        )
        self.side_sector_width_deg = self._positive_float_param(
            'side_sector_width_deg',
            25.0,
        )
        self.side_wall_target_distance_m = self._positive_float_param(
            'side_wall_target_distance_m',
            0.125,
        )

        self.angular_wall_centering_enabled = self._bool_param(
            'angular_wall_centering_enabled',
            False,
        )
        self.k_wall_centering = self._nonnegative_float_param(
            'k_wall_centering',
            0.60,
        )
        self.wall_centering_max_correction_radps = self._nonnegative_float_param(
            'wall_centering_max_correction_radps',
            0.08,
        )

        self.lateral_correction_enabled = self._bool_param(
            'lateral_correction_enabled',
            False,
        )
        self.k_lateral_cross_track = self._nonnegative_float_param(
            'k_lateral_cross_track',
            0.50,
        )
        self.max_lateral_correction_mps = self._nonnegative_float_param(
            'max_lateral_correction_mps',
            0.025,
        )

        self.wall_lateral_correction_enabled = self._bool_param(
            'wall_lateral_correction_enabled',
            False,
        )
        self.k_lateral_wall = self._nonnegative_float_param(
            'k_lateral_wall',
            0.40,
        )
        self.max_wall_lateral_correction_mps = self._nonnegative_float_param(
            'max_wall_lateral_correction_mps',
            0.025,
        )

        self.timeout_margin_sec = self._nonnegative_float_param(
            'timeout_margin_sec',
            3.0,
        )
        self.wait_timeout_margin_sec = self._nonnegative_float_param(
            'wait_timeout_margin_sec',
            1.0,
        )

        self.stop_publish_duration_sec = self._positive_float_param(
            'stop_publish_duration_sec',
            0.40,
        )
        self.stop_publish_rate_hz = self._positive_float_param(
            'stop_publish_rate_hz',
            25.0,
        )
        self.settle_time_sec = self._nonnegative_float_param('settle_time_sec', 0.15)
        self.enforce_final_error = self._bool_param('enforce_final_error', True)

    def _string_param(self, name: str, default: str) -> str:
        value = self.declare_parameter(name, default).value
        if not isinstance(value, str) or not value:
            self.get_logger().warn(f'Parameter {name} invalid; using {default!r}')
            return default
        return value

    def _bool_param(self, name: str, default: bool) -> bool:
        value = self.declare_parameter(name, default).value
        if isinstance(value, bool):
            return value
        self.get_logger().warn(f'Parameter {name} invalid; using {default!r}')
        return default

    def _positive_float_param(self, name: str, default: float) -> float:
        value = self.declare_parameter(name, default).value
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = default
        if not math.isfinite(value) or value <= 0.0:
            self.get_logger().warn(f'Parameter {name} invalid; using {default}')
            return default
        return value

    def _nonnegative_float_param(self, name: str, default: float) -> float:
        value = self.declare_parameter(name, default).value
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = default
        if not math.isfinite(value) or value < 0.0:
            self.get_logger().warn(f'Parameter {name} invalid; using {default}')
            return default
        return value

    def _odom_callback(self, msg: Odometry) -> None:
        with self._odom_lock:
            self._current_odom = msg
            self._last_odom_monotonic = time.monotonic()

    def _scan_callback(self, msg: LaserScan) -> None:
        front_clearance = self._min_range_in_sector(msg, 0.0, self.front_sector_deg)
        cardinal = cardinal_sector_ranges(
            msg,
            self.lidar_diagnostic_sector_width_deg,
            self.lidar_diagnostic_min_samples,
        )

        with self._scan_lock:
            self._latest_scan = msg
            self._last_scan_monotonic = time.monotonic()

        with self._state_lock:
            self._front_clearance_m = front_clearance
            self._front_range_m = finite_median_or_nan(cardinal['front'])
            self._rear_range_m = finite_median_or_nan(cardinal['rear'])
            self._left_range_m = finite_median_or_nan(cardinal['left'])
            self._right_range_m = finite_median_or_nan(cardinal['right'])

    def _goal_callback(self, goal_request) -> GoalResponse:
        primitive = int(goal_request.primitive_type)

        with self._motion_lock:
            active = self._active_goal

        if active:
            self.get_logger().warn('Rejecting goal because another primitive is active.')
            return GoalResponse.REJECT

        if primitive not in SUPPORTED_PRIMITIVES:
            self.get_logger().warn(f'Rejecting unsupported primitive_type={primitive}')
            return GoalResponse.REJECT

        if primitive in (
            ExecuteMotionPrimitive.Goal.DRIVE_FORWARD,
            ExecuteMotionPrimitive.Goal.DRIVE_BACKWARD,
        ) and goal_request.value <= 0.0:
            self.get_logger().warn('Rejecting drive goal with non-positive value.')
            return GoalResponse.REJECT

        if primitive == ExecuteMotionPrimitive.Goal.WAIT and goal_request.value < 0.0:
            self.get_logger().warn('Rejecting WAIT goal with negative value.')
            return GoalResponse.REJECT

        if primitive == ExecuteMotionPrimitive.Goal.ROTATE_RELATIVE:
            if not math.isfinite(float(goal_request.value)):
                self.get_logger().warn('Rejecting ROTATE_RELATIVE with non-finite value.')
                return GoalResponse.REJECT

        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle) -> CancelResponse:
        self.get_logger().warn('Cancel requested.')
        with self._motion_lock:
            self._cancel_requested = True
            self._stop_requested = True
        return CancelResponse.ACCEPT

    def _handle_stop(self, request, response):
        del request
        with self._motion_lock:
            self._stop_requested = True
            self._cancel_requested = True

        self._set_state(
            ControlStatus.STATE_STOPPING,
            'external stop requested',
            primitive_type=0,
        )
        self._publish_zero_for_duration()

        response.success = True
        response.message = 'Stop requested and zero Twist published.'
        return response

    def _execute_callback(self, goal_handle):
        primitive = int(goal_handle.request.primitive_type)
        primitive_name = PRIMITIVE_NAMES.get(primitive, f'UNKNOWN({primitive})')

        with self._motion_lock:
            self._active_goal = True
            self._stop_requested = False
            self._cancel_requested = False

        self._set_state(
            ControlStatus.STATE_EXECUTING,
            f'executing {primitive_name}',
            primitive_type=primitive,
        )

        self.get_logger().info(f'Accepted primitive {primitive_name}')

        try:
            if primitive == ExecuteMotionPrimitive.Goal.STOP:
                result = self._execute_stop(goal_handle)
            elif primitive == ExecuteMotionPrimitive.Goal.WAIT:
                result = self._execute_wait(goal_handle)
            elif primitive == ExecuteMotionPrimitive.Goal.ROTATE_RELATIVE:
                result = self._execute_rotate(goal_handle)
            elif primitive in (
                ExecuteMotionPrimitive.Goal.DRIVE_FORWARD,
                ExecuteMotionPrimitive.Goal.DRIVE_BACKWARD,
                ExecuteMotionPrimitive.Goal.ADVANCE_CELL,
            ):
                result = self._execute_translate(goal_handle)
            else:
                goal_handle.abort()
                result = self._make_result(
                    False,
                    ExecuteMotionPrimitive.Result.REJECTED_INVALID_GOAL,
                    f'unsupported primitive_type={primitive}',
                )

            return result

        except Exception as exc:
            self.get_logger().exception(f'Primitive execution failed: {exc}')
            self._publish_zero_for_duration()
            self._set_state(ControlStatus.STATE_FAILED, 'internal error', primitive_type=0)
            goal_handle.abort()
            return self._make_result(
                False,
                ExecuteMotionPrimitive.Result.INTERNAL_ERROR,
                f'internal error: {exc}',
            )

        finally:
            self._publish_zero_for_duration()
            with self._motion_lock:
                self._active_goal = False
                self._stop_requested = False
                self._cancel_requested = False
            with self._state_lock:
                if self._state not in (
                    ControlStatus.STATE_SUCCEEDED,
                    ControlStatus.STATE_CANCELED,
                    ControlStatus.STATE_FAILED,
                ):
                    self._state = ControlStatus.STATE_IDLE
                    self._status = 'ready'
                    self._active_primitive_type = 0

    def _execute_stop(self, goal_handle):
        self._set_state(ControlStatus.STATE_STOPPING, 'stopping', primitive_type=0)
        self._publish_zero_for_duration()
        self._set_state(ControlStatus.STATE_SUCCEEDED, 'STOP succeeded', primitive_type=0)
        goal_handle.succeed()
        return self._make_result(
            True,
            ExecuteMotionPrimitive.Result.SUCCESS,
            'STOP succeeded',
        )

    def _execute_wait(self, goal_handle):
        wait_s = max(0.0, float(goal_handle.request.value))
        timeout_s = self._goal_timeout(
            goal_handle.request.timeout_s,
            wait_s + self.wait_timeout_margin_sec,
        )
        start = time.monotonic()

        while time.monotonic() - start < wait_s:
            if self._cancel_or_stop_requested(goal_handle):
                return self._cancel_result(goal_handle)

            if time.monotonic() - start > timeout_s:
                return self._timeout_result(goal_handle, 'WAIT timed out')

            self.publish_zero_twist()
            self._publish_feedback(
                goal_handle,
                progress=(time.monotonic() - start) / max(wait_s, 1e-6),
                distance_remaining=0.0,
                heading_remaining=0.0,
                state='waiting',
            )
            time.sleep(1.0 / self.control_rate_hz)

        self._set_state(ControlStatus.STATE_SUCCEEDED, 'WAIT succeeded', primitive_type=0)
        goal_handle.succeed()
        return self._make_result(
            True,
            ExecuteMotionPrimitive.Result.SUCCESS,
            'WAIT succeeded',
        )

    def _execute_translate(self, goal_handle):
        request = goal_handle.request
        primitive = int(request.primitive_type)

        if primitive == ExecuteMotionPrimitive.Goal.ADVANCE_CELL:
            target_distance = self.cell_length_m
            direction = 1.0
            name = 'ADVANCE_CELL'
        elif primitive == ExecuteMotionPrimitive.Goal.DRIVE_FORWARD:
            target_distance = abs(float(request.value))
            direction = 1.0
            name = 'DRIVE_FORWARD'
        elif primitive == ExecuteMotionPrimitive.Goal.DRIVE_BACKWARD:
            target_distance = abs(float(request.value))
            direction = -1.0
            name = 'DRIVE_BACKWARD'
        else:
            return self._fail_goal(
                goal_handle,
                ExecuteMotionPrimitive.Result.REJECTED_INVALID_GOAL,
                'invalid translation primitive',
            )

        position_tol = self._goal_tolerance(
            request.position_tolerance_m,
            self.default_position_tolerance_m,
        )
        heading_tol = self._goal_tolerance(
            request.heading_tolerance_rad,
            self.default_heading_tolerance_rad,
        )

        limits = self._goal_limits(request)

        max_speed = min(
            limits.max_linear_x_mps,
            self.default_reverse_speed_mps if direction < 0.0 else self.default_linear_speed_mps,
        )
        max_speed = max(max_speed, self.min_linear_x_mps)

        timeout_s = self._translation_timeout(request.timeout_s, target_distance, max_speed)

        if direction > 0.0 and bool(request.collision_check_enabled):
            if self.require_scan_for_forward and not self._is_scan_fresh():
                return self._fail_goal(
                    goal_handle,
                    ExecuteMotionPrimitive.Result.OBSTACLE_TOO_CLOSE,
                    f'{name} requires fresh LiDAR scan',
                )

            clearance = self._front_clearance()
            if clearance < self.front_stop_distance_m:
                return self._fail_goal(
                    goal_handle,
                    ExecuteMotionPrimitive.Result.OBSTACLE_TOO_CLOSE,
                    f'{name} blocked before start: front_clearance={clearance:.3f} m',
                )

        start_snapshot = self._get_motion_snapshot()
        if start_snapshot is None:
            return self._odom_unavailable_result(goal_handle)

        start = start_snapshot.pose
        start_ranges = self._cardinal_range_snapshot()
        start_time = time.monotonic()

        self._set_state(ControlStatus.STATE_EXECUTING, f'{name} executing', primitive_type=primitive)

        result_success = False
        result_code = ExecuteMotionPrimitive.Result.INTERNAL_ERROR
        result_message = 'unknown translation result'
        final_position_error = target_distance
        final_heading_error = 0.0
        final_odom_progress = 0.0
        translation_diagnostics = TranslationDiagnostics()

        while True:
            if self._cancel_or_stop_requested(goal_handle):
                return self._cancel_result(goal_handle)

            elapsed = time.monotonic() - start_time
            if elapsed > timeout_s:
                result_code = ExecuteMotionPrimitive.Result.TIMEOUT
                result_message = f'{name} timed out after {elapsed:.1f}s'
                break

            snapshot = self._get_motion_snapshot()
            if snapshot is None:
                result_code = ExecuteMotionPrimitive.Result.ODOM_UNAVAILABLE
                result_message = f'{name} lost odom'
                break

            current = snapshot.pose

            progress_signed, cross_track = self._translation_errors(start, current, direction)
            final_odom_progress = max(0.0, progress_signed)
            remaining = target_distance - progress_signed
            heading_error = normalize_angle(start.yaw - current.yaw)

            final_position_error = abs(remaining)
            final_heading_error = abs(heading_error)

            if direction > 0.0 and bool(request.collision_check_enabled):
                if self.require_scan_for_forward and not self._is_scan_fresh():
                    result_code = ExecuteMotionPrimitive.Result.OBSTACLE_TOO_CLOSE
                    result_message = f'{name} stopped: LiDAR scan became stale'
                    break

                clearance = self._front_clearance()
                if not math.isfinite(clearance):
                    clearance = float('inf')
                if clearance < self.front_stop_distance_m:
                    result_code = ExecuteMotionPrimitive.Result.OBSTACLE_TOO_CLOSE
                    result_message = (
                        f'{name} stopped: front_clearance={clearance:.3f} m '
                        f'< {self.front_stop_distance_m:.3f} m'
                    )
                    break

            if remaining <= position_tol:
                self._publish_zero_for_duration()
                time.sleep(self.settle_time_sec)

                final_snapshot = self._get_motion_snapshot()
                if final_snapshot is not None:
                    progress_signed, cross_track = self._translation_errors(
                        start,
                        final_snapshot.pose,
                        direction,
                    )
                    remaining = target_distance - progress_signed
                    heading_error = normalize_angle(start.yaw - final_snapshot.pose.yaw)
                    final_position_error = abs(remaining)
                    final_heading_error = abs(heading_error)
                    final_odom_progress = max(0.0, progress_signed)

                if (
                    self.enforce_final_error
                    and (
                        final_position_error > position_tol * 1.5
                        or final_heading_error > heading_tol * 1.5
                    )
                ):
                    result_code = ExecuteMotionPrimitive.Result.FINAL_ERROR_TOO_LARGE
                    result_message = (
                        f'{name} final error too large: '
                        f'pos={final_position_error:.3f} m, '
                        f'heading={final_heading_error:.3f} rad'
                    )
                    break

                result_success = True
                result_code = ExecuteMotionPrimitive.Result.SUCCESS
                result_message = (
                    f'{name} succeeded: '
                    f'pos_error={final_position_error:.3f} m, '
                    f'heading_error={final_heading_error:.3f} rad'
                )
                break

            cmd = Twist()
            speed_mag = clamp(
                self.k_distance * max(remaining, 0.0),
                self.min_linear_x_mps,
                max_speed,
            )
            cmd.linear.x = direction * speed_mag

            heading_correction = self.k_heading * heading_error

            lateral_correction = 0.0

            if direction > 0.0 and self.lateral_correction_enabled:
                lateral_correction += clamp(
                    -self.k_lateral_cross_track * cross_track,
                    -self.max_lateral_correction_mps,
                    self.max_lateral_correction_mps,
                )

            if direction > 0.0:
                lateral_correction += self._wall_lateral_correction()

            cmd.linear.y = clamp(
                lateral_correction,
                -self.max_linear_y_mps,
                self.max_linear_y_mps,
            )

            cmd.angular.z = heading_correction

            if direction > 0.0:
                cmd.angular.z += self._wall_angular_correction()

            cmd = self._limiter.clamp(cmd, limits)
            cmd = self._apply_acceleration_limits(cmd)
            self._cmd_vel_pub.publish(cmd)

            self._update_motion_state(
                distance_remaining=max(0.0, remaining),
                distance_traveled=max(0.0, progress_signed),
                heading_error=heading_error,
                status=(
                    f'{name}: remaining={remaining:.3f} m, '
                    f'heading_error={heading_error:.3f} rad, '
                    f'linear_y={cmd.linear.y:.3f} m/s'
                ),
            )

            self._publish_feedback(
                goal_handle,
                progress=clamp(progress_signed / max(target_distance, 1e-6), 0.0, 1.0),
                distance_remaining=max(0.0, remaining),
                heading_remaining=heading_error,
                state=name,
            )

            time.sleep(1.0 / self.control_rate_hz)

        end_ranges = self._cardinal_range_snapshot()
        translation_diagnostics = self._translation_diagnostics(
            direction=direction,
            odom_progress_m=final_odom_progress,
            start_ranges=start_ranges,
            end_ranges=end_ranges,
        )

        result_message = self._append_translation_diagnostics(
            result_message,
            translation_diagnostics,
        )

        self.get_logger().info(
            f'{name} diagnostics: '
            f'odom_progress={translation_diagnostics.odom_progress_m:.3f} m, '
            f'front_valid={translation_diagnostics.front_range_valid}, '
            f'front_progress={translation_diagnostics.front_progress_m:.3f} m, '
            f'rear_valid={translation_diagnostics.rear_range_valid}, '
            f'rear_progress={translation_diagnostics.rear_progress_m:.3f} m, '
            f'lidar_valid={translation_diagnostics.lidar_progress_valid}, '
            f'lidar_progress={translation_diagnostics.lidar_progress_m:.3f} m, '
            f'lidar_minus_odom={translation_diagnostics.lidar_minus_odom_m:.3f} m'
        )

        return self._finish_motion_result(
            goal_handle,
            result_success,
            result_code,
            result_message,
            final_position_error,
            final_heading_error,
            translation_diagnostics=translation_diagnostics,
        )

    def _execute_rotate(self, goal_handle):
        request = goal_handle.request
        target_angle = float(request.value)

        heading_tol = self._goal_tolerance(
            request.heading_tolerance_rad,
            self.default_heading_tolerance_rad,
        )
        limits = self._goal_limits(request)

        max_speed = min(limits.max_angular_z_radps, self.default_angular_speed_radps)
        max_speed = max(max_speed, self.min_angular_z_radps)

        timeout_s = self._rotation_timeout(request.timeout_s, target_angle, max_speed)

        start_snapshot = self._get_motion_snapshot()
        if start_snapshot is None:
            return self._odom_unavailable_result(goal_handle)

        start = start_snapshot.pose
        target_yaw = normalize_angle(start.yaw + target_angle)
        start_time = time.monotonic()

        self._set_state(
            ControlStatus.STATE_EXECUTING,
            'ROTATE_RELATIVE executing',
            primitive_type=ExecuteMotionPrimitive.Goal.ROTATE_RELATIVE,
        )

        result_success = False
        result_code = ExecuteMotionPrimitive.Result.INTERNAL_ERROR
        result_message = 'unknown rotation result'
        final_heading_error = abs(target_angle)

        while True:
            if self._cancel_or_stop_requested(goal_handle):
                return self._cancel_result(goal_handle)

            elapsed = time.monotonic() - start_time
            if elapsed > timeout_s:
                result_code = ExecuteMotionPrimitive.Result.TIMEOUT
                result_message = f'ROTATE_RELATIVE timed out after {elapsed:.1f}s'
                break

            snapshot = self._get_motion_snapshot()
            if snapshot is None:
                result_code = ExecuteMotionPrimitive.Result.ODOM_UNAVAILABLE
                result_message = 'ROTATE_RELATIVE lost odom'
                break

            current = snapshot.pose
            remaining = normalize_angle(target_yaw - current.yaw)
            final_heading_error = abs(remaining)

            rotated = abs(normalize_angle(current.yaw - start.yaw))
            target_abs = max(abs(target_angle), 1e-6)

            if final_heading_error <= heading_tol:
                self._publish_zero_for_duration()
                time.sleep(self.settle_time_sec)

                final_snapshot = self._get_motion_snapshot()
                if final_snapshot is not None:
                    remaining = normalize_angle(target_yaw - final_snapshot.pose.yaw)
                    final_heading_error = abs(remaining)

                if self.enforce_final_error and final_heading_error > heading_tol * 1.5:
                    result_code = ExecuteMotionPrimitive.Result.FINAL_ERROR_TOO_LARGE
                    result_message = (
                        f'ROTATE_RELATIVE final heading error too large: '
                        f'{final_heading_error:.3f} rad'
                    )
                    break

                result_success = True
                result_code = ExecuteMotionPrimitive.Result.SUCCESS
                result_message = (
                    f'ROTATE_RELATIVE succeeded: '
                    f'heading_error={final_heading_error:.3f} rad'
                )
                break

            speed_mag = clamp(
                self.k_heading * final_heading_error,
                self.min_angular_z_radps,
                max_speed,
            )

            cmd = Twist()
            cmd.angular.z = sign(remaining) * speed_mag
            cmd = self._limiter.clamp(cmd, limits)
            cmd = self._apply_acceleration_limits(cmd)
            self._cmd_vel_pub.publish(cmd)

            self._update_motion_state(
                distance_remaining=0.0,
                distance_traveled=0.0,
                heading_error=remaining,
                status=f'ROTATE_RELATIVE: remaining={remaining:.3f} rad',
            )

            self._publish_feedback(
                goal_handle,
                progress=clamp(rotated / target_abs, 0.0, 1.0),
                distance_remaining=0.0,
                heading_remaining=remaining,
                state='ROTATE_RELATIVE',
            )

            time.sleep(1.0 / self.control_rate_hz)

        return self._finish_motion_result(
            goal_handle,
            result_success,
            result_code,
            result_message,
            0.0,
            final_heading_error,
        )

    def _translation_errors(self, start: Pose2D, current: Pose2D, direction: float):
        dx = current.x - start.x
        dy = current.y - start.y

        forward_x = math.cos(start.yaw)
        forward_y = math.sin(start.yaw)

        left_x = -math.sin(start.yaw)
        left_y = math.cos(start.yaw)

        forward_progress = dx * forward_x + dy * forward_y
        cross_track = dx * left_x + dy * left_y

        commanded_progress = direction * forward_progress
        return commanded_progress, cross_track

    def _side_wall_error(self) -> Optional[float]:
        scan = self._fresh_scan_copy()
        if scan is None:
            return None

        left = self._min_range_in_sector(
            scan,
            math.radians(self.side_sector_center_deg),
            self.side_sector_width_deg,
        )
        right = self._min_range_in_sector(
            scan,
            -math.radians(self.side_sector_center_deg),
            self.side_sector_width_deg,
        )

        left_ok = math.isfinite(left)
        right_ok = math.isfinite(right)

        if left_ok and right_ok:
            # Positive means robot is closer to left wall than right wall,
            # so it should move right in body frame if y-left is positive.
            return right - left

        if left_ok:
            # Positive if too close to left wall.
            return self.side_wall_target_distance_m - left

        if right_ok:
            # Negative if too close to right wall.
            return right - self.side_wall_target_distance_m

        return None

    def _wall_angular_correction(self) -> float:
        if not self.wall_centering_enabled or not self.angular_wall_centering_enabled:
            return 0.0

        error = self._side_wall_error()
        if error is None:
            return 0.0

        correction = self.k_wall_centering * error
        return clamp(
            correction,
            -self.wall_centering_max_correction_radps,
            self.wall_centering_max_correction_radps,
        )

    def _wall_lateral_correction(self) -> float:
        if not self.wall_centering_enabled or not self.wall_lateral_correction_enabled:
            return 0.0

        error = self._side_wall_error()
        if error is None:
            return 0.0

        # ROS body-frame convention: +linear.y is left.
        # Positive error means too close to left or left side needs correction,
        # so command negative y to move right.
        correction = -self.k_lateral_wall * error
        return clamp(
            correction,
            -self.max_wall_lateral_correction_mps,
            self.max_wall_lateral_correction_mps,
        )

    def _min_range_in_sector(
        self,
        scan: LaserScan,
        center_angle_rad: float,
        width_deg: float,
    ) -> float:
        if scan is None:
            return float('inf')

        half_width = math.radians(width_deg) / 2.0
        center = normalize_angle(center_angle_rad)

        ranges = []
        angle = scan.angle_min

        for value in scan.ranges:
            delta = normalize_angle(angle - center)
            if abs(delta) <= half_width:
                if math.isfinite(value) and scan.range_min <= value <= scan.range_max:
                    ranges.append(float(value))
            angle += scan.angle_increment

        if not ranges:
            return float('inf')

        return min(ranges)

    def _front_clearance(self) -> float:
        scan = self._fresh_scan_copy()
        if scan is None:
            return float('inf')
        return self._min_range_in_sector(scan, 0.0, self.front_sector_deg)

    def _cardinal_range_snapshot(self) -> Optional[LidarRangeSnapshot]:
        if not self.lidar_diagnostics_enabled:
            return None

        scan = self._fresh_scan_copy()
        if scan is None:
            return None

        measurements = cardinal_sector_ranges(
            scan,
            self.lidar_diagnostic_sector_width_deg,
            self.lidar_diagnostic_min_samples,
        )

        return LidarRangeSnapshot(
            front=measurements['front'],
            rear=measurements['rear'],
            left=measurements['left'],
            right=measurements['right'],
        )

    @staticmethod
    def _range_value(measurement: SectorRange) -> float:
        return float(measurement.median_m) if measurement.valid else 0.0

    def _translation_diagnostics(
        self,
        direction: float,
        odom_progress_m: float,
        start_ranges: Optional[LidarRangeSnapshot],
        end_ranges: Optional[LidarRangeSnapshot],
    ) -> TranslationDiagnostics:
        if start_ranges is None or end_ranges is None:
            return TranslationDiagnostics(odom_progress_m=float(max(0.0, odom_progress_m)))

        front_valid = start_ranges.front.valid and end_ranges.front.valid
        rear_valid = start_ranges.rear.valid and end_ranges.rear.valid
        left_valid = start_ranges.left.valid and end_ranges.left.valid
        right_valid = start_ranges.right.valid and end_ranges.right.valid

        front_start = self._range_value(start_ranges.front)
        front_end = self._range_value(end_ranges.front)
        rear_start = self._range_value(start_ranges.rear)
        rear_end = self._range_value(end_ranges.rear)
        left_start = self._range_value(start_ranges.left)
        left_end = self._range_value(end_ranges.left)
        right_start = self._range_value(start_ranges.right)
        right_end = self._range_value(end_ranges.right)

        if direction >= 0.0:
            front_progress = front_start - front_end if front_valid else 0.0
            rear_progress = rear_end - rear_start if rear_valid else 0.0
        else:
            front_progress = front_end - front_start if front_valid else 0.0
            rear_progress = rear_start - rear_end if rear_valid else 0.0

        lidar_progress_valid, lidar_progress = combine_lidar_progress_candidates(
            front_valid,
            front_progress,
            rear_valid,
            rear_progress,
        )

        odom_progress = float(max(0.0, odom_progress_m))
        lidar_minus_odom = lidar_progress - odom_progress if lidar_progress_valid else 0.0

        return TranslationDiagnostics(
            odom_progress_m=odom_progress,
            front_range_valid=front_valid,
            front_range_start_m=front_start,
            front_range_end_m=front_end,
            front_progress_m=front_progress,
            rear_range_valid=rear_valid,
            rear_range_start_m=rear_start,
            rear_range_end_m=rear_end,
            rear_progress_m=rear_progress,
            left_range_valid=left_valid,
            left_range_start_m=left_start,
            left_range_end_m=left_end,
            right_range_valid=right_valid,
            right_range_start_m=right_start,
            right_range_end_m=right_end,
            lidar_progress_valid=lidar_progress_valid,
            lidar_progress_m=lidar_progress,
            lidar_minus_odom_m=lidar_minus_odom,
        )

    @staticmethod
    def _append_translation_diagnostics(
        message: str,
        diagnostics: TranslationDiagnostics,
    ) -> str:
        return (
            f'{message}; '
            f'odom_progress={diagnostics.odom_progress_m:.3f} m; '
            f'front_valid={diagnostics.front_range_valid}; '
            f'front_start={diagnostics.front_range_start_m:.3f} m; '
            f'front_end={diagnostics.front_range_end_m:.3f} m; '
            f'front_progress={diagnostics.front_progress_m:.3f} m; '
            f'rear_valid={diagnostics.rear_range_valid}; '
            f'rear_start={diagnostics.rear_range_start_m:.3f} m; '
            f'rear_end={diagnostics.rear_range_end_m:.3f} m; '
            f'rear_progress={diagnostics.rear_progress_m:.3f} m; '
            f'left_valid={diagnostics.left_range_valid}; '
            f'left_start={diagnostics.left_range_start_m:.3f} m; '
            f'left_end={diagnostics.left_range_end_m:.3f} m; '
            f'right_valid={diagnostics.right_range_valid}; '
            f'right_start={diagnostics.right_range_start_m:.3f} m; '
            f'right_end={diagnostics.right_range_end_m:.3f} m; '
            f'lidar_valid={diagnostics.lidar_progress_valid}; '
            f'lidar_progress={diagnostics.lidar_progress_m:.3f} m; '
            f'lidar_minus_odom={diagnostics.lidar_minus_odom_m:.3f} m'
        )

    def _fresh_scan_copy(self) -> Optional[LaserScan]:
        with self._scan_lock:
            if self._latest_scan is None or self._last_scan_monotonic is None:
                return None
            if time.monotonic() - self._last_scan_monotonic > self.scan_timeout_sec:
                return None
            return deepcopy(self._latest_scan)

    def _get_motion_snapshot(self) -> Optional[MotionSnapshot]:
        with self._odom_lock:
            if self._current_odom is None or self._last_odom_monotonic is None:
                return None
            if time.monotonic() - self._last_odom_monotonic > self.odom_timeout_sec:
                return None
            odom = deepcopy(self._current_odom)

        pose_msg = PoseStamped()
        pose_msg.header = odom.header
        pose_msg.pose = odom.pose.pose

        if not is_finite_pose_stamped(pose_msg):
            return None

        pose = Pose2D(
            x=float(pose_msg.pose.position.x),
            y=float(pose_msg.pose.position.y),
            yaw=yaw_from_quaternion(pose_msg.pose.orientation),
        )

        return MotionSnapshot(pose=pose, pose_msg=pose_msg)

    def _goal_limits(self, request) -> VelocityLimits:
        return self._limiter.sanitize_goal_limits(
            float(request.max_linear_x_mps),
            float(request.max_linear_y_mps),
            float(request.max_angular_z_radps),
        )

    def _goal_tolerance(self, requested: float, default: float) -> float:
        if math.isfinite(float(requested)) and float(requested) > 0.0:
            return float(requested)
        return default

    def _goal_timeout(self, requested: float, fallback: float) -> float:
        if math.isfinite(float(requested)) and float(requested) > 0.0:
            return float(requested)
        return fallback

    def _translation_timeout(self, requested: float, distance: float, speed: float) -> float:
        fallback = abs(distance) / max(abs(speed), 1e-3) + self.timeout_margin_sec
        fallback = max(fallback, 3.0)
        return self._goal_timeout(requested, fallback)

    def _rotation_timeout(self, requested: float, angle: float, speed: float) -> float:
        fallback = abs(angle) / max(abs(speed), 1e-3) + self.timeout_margin_sec
        fallback = max(fallback, 3.0)
        return self._goal_timeout(requested, fallback)

    def _apply_acceleration_limits(self, desired: Twist) -> Twist:
        now = time.monotonic()
        dt = max(now - self._last_command_time, 1.0 / self.control_rate_hz)

        limited = Twist()

        max_dx = self.max_linear_accel_mps2 * dt
        max_dy = self.max_lateral_accel_mps2 * dt
        max_dz = self.max_angular_accel_radps2 * dt

        limited.linear.x = self._step_toward(
            self._last_commanded_twist.linear.x,
            desired.linear.x,
            max_dx,
        )
        limited.linear.y = self._step_toward(
            self._last_commanded_twist.linear.y,
            desired.linear.y,
            max_dy,
        )
        limited.angular.z = self._step_toward(
            self._last_commanded_twist.angular.z,
            desired.angular.z,
            max_dz,
        )

        self._last_commanded_twist = limited
        self._last_command_time = now

        return limited

    @staticmethod
    def _step_toward(current: float, target: float, max_step: float) -> float:
        delta = target - current
        if abs(delta) <= max_step:
            return target
        return current + sign(delta) * max_step

    def _cancel_or_stop_requested(self, goal_handle) -> bool:
        if goal_handle.is_cancel_requested:
            with self._motion_lock:
                self._cancel_requested = True
                self._stop_requested = True
            return True

        with self._motion_lock:
            return self._stop_requested or self._cancel_requested

    def _cancel_result(self, goal_handle):
        self._publish_zero_for_duration()
        goal_handle.canceled()
        self._set_state(ControlStatus.STATE_CANCELED, 'canceled', primitive_type=0)
        return self._make_result(
            False,
            ExecuteMotionPrimitive.Result.CANCELED,
            'primitive canceled',
        )

    def _timeout_result(self, goal_handle, message: str):
        self._publish_zero_for_duration()
        goal_handle.abort()
        self._set_state(ControlStatus.STATE_FAILED, message, primitive_type=0)
        return self._make_result(
            False,
            ExecuteMotionPrimitive.Result.TIMEOUT,
            message,
        )

    def _odom_unavailable_result(self, goal_handle):
        self._publish_zero_for_duration()
        goal_handle.abort()
        self._set_state(ControlStatus.STATE_FAILED, 'odom unavailable', primitive_type=0)
        return self._make_result(
            False,
            ExecuteMotionPrimitive.Result.ODOM_UNAVAILABLE,
            'fresh odom unavailable',
        )

    def _fail_goal(self, goal_handle, result_code: int, message: str):
        self._publish_zero_for_duration()
        goal_handle.abort()
        self._set_state(ControlStatus.STATE_FAILED, message, primitive_type=0)
        return self._make_result(False, result_code, message)

    def _finish_motion_result(
        self,
        goal_handle,
        success: bool,
        code: int,
        message: str,
        final_position_error: float,
        final_heading_error: float,
        translation_diagnostics: Optional[TranslationDiagnostics] = None,
    ):
        self._publish_zero_for_duration()

        if success:
            goal_handle.succeed()
            self._set_state(ControlStatus.STATE_SUCCEEDED, message, primitive_type=0)
        else:
            goal_handle.abort()
            self._set_state(ControlStatus.STATE_FAILED, message, primitive_type=0)

        return self._make_result(
            success,
            code,
            message,
            final_position_error=final_position_error,
            final_heading_error=final_heading_error,
            translation_diagnostics=translation_diagnostics,
        )

    def _make_result(
        self,
        success: bool,
        code: int,
        message: str,
        final_position_error: float = 0.0,
        final_heading_error: float = 0.0,
        translation_diagnostics: Optional[TranslationDiagnostics] = None,
    ):
        result = ExecuteMotionPrimitive.Result()
        result.success = bool(success)
        result.result_code = int(code)
        result.message = str(message)
        result.final_position_error_m = float(final_position_error)
        result.final_heading_error_rad = float(final_heading_error)
        diagnostics = translation_diagnostics or TranslationDiagnostics()

        result.final_odom_progress_m = float(diagnostics.odom_progress_m)

        result.front_range_valid = bool(diagnostics.front_range_valid)
        result.front_range_start_m = float(diagnostics.front_range_start_m)
        result.front_range_end_m = float(diagnostics.front_range_end_m)
        result.front_progress_m = float(diagnostics.front_progress_m)

        result.rear_range_valid = bool(diagnostics.rear_range_valid)
        result.rear_range_start_m = float(diagnostics.rear_range_start_m)
        result.rear_range_end_m = float(diagnostics.rear_range_end_m)
        result.rear_progress_m = float(diagnostics.rear_progress_m)

        result.left_range_valid = bool(diagnostics.left_range_valid)
        result.left_range_start_m = float(diagnostics.left_range_start_m)
        result.left_range_end_m = float(diagnostics.left_range_end_m)

        result.right_range_valid = bool(diagnostics.right_range_valid)
        result.right_range_start_m = float(diagnostics.right_range_start_m)
        result.right_range_end_m = float(diagnostics.right_range_end_m)

        result.lidar_progress_valid = bool(diagnostics.lidar_progress_valid)
        result.lidar_progress_m = float(diagnostics.lidar_progress_m)
        result.lidar_minus_odom_m = float(diagnostics.lidar_minus_odom_m)
        return result

    def _publish_feedback(
        self,
        goal_handle,
        progress: float,
        distance_remaining: float,
        heading_remaining: float,
        state: str,
    ) -> None:
        feedback = ExecuteMotionPrimitive.Feedback()
        feedback.progress = float(clamp(progress, 0.0, 1.0))
        feedback.distance_remaining_m = float(distance_remaining)
        feedback.heading_remaining_rad = float(heading_remaining)
        feedback.front_clearance_m = float(self._front_clearance())
        feedback.state = state
        goal_handle.publish_feedback(feedback)

    def _update_motion_state(
        self,
        distance_remaining: float,
        distance_traveled: float,
        heading_error: float,
        status: str,
    ) -> None:
        with self._state_lock:
            self._distance_remaining_m = float(distance_remaining)
            self._distance_traveled_m = float(distance_traveled)
            self._heading_error_rad = float(heading_error)
            self._status = status

    def _set_state(
        self,
        state: int,
        status: str,
        primitive_type: int = 0,
    ) -> None:
        with self._state_lock:
            self._state = int(state)
            self._status = str(status)
            self._active_primitive_type = int(primitive_type)
            if primitive_type == 0:
                self._distance_remaining_m = 0.0
                self._distance_traveled_m = 0.0
                self._heading_error_rad = 0.0

    def _is_odom_fresh(self) -> bool:
        with self._odom_lock:
            if self._last_odom_monotonic is None:
                return False
            return time.monotonic() - self._last_odom_monotonic <= self.odom_timeout_sec

    def _is_scan_fresh(self) -> bool:
        with self._scan_lock:
            if self._last_scan_monotonic is None:
                return False
            return time.monotonic() - self._last_scan_monotonic <= self.scan_timeout_sec

    def publish_zero_twist(self) -> None:
        zero = Twist()
        self._cmd_vel_pub.publish(zero)
        self._last_commanded_twist = zero
        self._last_command_time = time.monotonic()

    def _publish_zero_for_duration(self) -> None:
        end = time.monotonic() + self.stop_publish_duration_sec
        period = 1.0 / max(self.stop_publish_rate_hz, 1.0)
        while time.monotonic() < end:
            self.publish_zero_twist()
            time.sleep(period)

    def _publish_status(self) -> None:
        with self._state_lock:
            state = self._state
            status = self._status
            command_enabled = self._command_enabled
            active_primitive_type = self._active_primitive_type
            distance_remaining = self._distance_remaining_m
            distance_traveled = self._distance_traveled_m
            heading_error = self._heading_error_rad
            front_clearance = self._front_clearance_m
            front_range = self._front_range_m
            rear_range = self._rear_range_m
            left_range = self._left_range_m
            right_range = self._right_range_m

        msg = ControlStatus()
        msg.stamp = self.get_clock().now().to_msg()
        msg.state = state
        msg.status = status
        msg.odom_available = self._is_odom_fresh()
        msg.scan_available = self._is_scan_fresh()
        msg.command_enabled = command_enabled
        msg.active_primitive_type = active_primitive_type
        msg.distance_remaining_m = distance_remaining
        msg.distance_traveled_m = distance_traveled
        msg.heading_error_rad = heading_error
        msg.front_clearance_m = front_clearance
        msg.front_range_m = front_range
        msg.rear_range_m = rear_range
        msg.left_range_m = left_range
        msg.right_range_m = right_range
        self._status_pub.publish(msg)

    def destroy_node(self):
        self._publish_zero_for_duration()
        if hasattr(self, '_action_server'):
            self._action_server.destroy()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DarthMaulControlNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.publish_zero_twist()
        try:
            node.destroy_node()
        finally:
            executor.shutdown()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    main()
