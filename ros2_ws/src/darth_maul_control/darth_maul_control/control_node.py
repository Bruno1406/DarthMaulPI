from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
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
    GridAlignmentEstimate,
    LidarProgressEstimate,
    SectorRange,
    WallLineEstimate,
    cardinal_sector_ranges,
    choose_lidar_progress,
    choose_translation_progress,
    estimate_grid_alignment,
    finite_median_or_nan,
    grid_lateral_drift,
    grid_yaw_correction_radps,
    invalid_grid_alignment,
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

    final_control_progress_m: float = 0.0
    progress_source_used: str = 'odom'

    lidar_progress_source: str = 'none'
    lidar_progress_disagreement_m: float = 0.0
    lidar_progress_reason: str = ''
    control_progress_reason: str = ''

    grid_yaw_correction_used: bool = False
    final_grid_yaw_correction_radps: float = 0.0
    grid_yaw_control_reason: str = ''

    grid_lateral_start_valid: bool = False
    grid_lateral_start_error_m: float = 0.0
    grid_lateral_end_valid: bool = False
    grid_lateral_end_error_m: float = 0.0
    grid_lateral_drift_valid: bool = False
    grid_lateral_drift_m: float = 0.0
    grid_lateral_drift_per_m: float = 0.0
    grid_lateral_drift_reason: str = ''


@dataclass(frozen=True)
class GridYawCorrection:
    active: bool
    correction_radps: float
    reason: str
    source: str
    using_memory: bool


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
        self._grid_alignment = invalid_grid_alignment('not initialized')
        self._last_valid_grid_control_alignment = invalid_grid_alignment(
            'no valid grid control alignment yet'
        )
        self._last_valid_grid_control_alignment_monotonic: Optional[float] = None
        self._grid_yaw_control_memory_accepting = False
        self._grid_yaw_correction_active = False
        self._grid_yaw_correction_radps = 0.0
        self._grid_yaw_control_reason = ''

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
        self.lidar_diagnostic_min_samples = self._positive_int_param(
            'lidar_diagnostic_min_samples',
            3,
        )
        self.translation_progress_source = self._string_param(
            'translation_progress_source',
            'lidar_when_consistent',
        )
        if self.translation_progress_source not in (
            'odom_only',
            'lidar_when_consistent',
            'lidar_required',
        ):
            self.get_logger().warning(
                f'Invalid translation_progress_source={self.translation_progress_source!r}; '
                'falling back to odom_only'
            )
            self.translation_progress_source = 'odom_only'

        self.lidar_progress_max_disagreement_m = self._positive_float_param(
            'lidar_progress_max_disagreement_m',
            0.025,
        )
        self.lidar_progress_min_m = self._float_param(
            'lidar_progress_min_m',
            -0.010,
        )
        self.lidar_progress_allow_single_source = self._bool_param(
            'lidar_progress_allow_single_source',
            True,
        )
        self.lidar_progress_max_ahead_of_odom_m = self._positive_float_param(
            'lidar_progress_max_ahead_of_odom_m',
            0.060,
        )
        self.lidar_odom_warning_threshold_m = self._positive_float_param(
            'lidar_odom_warning_threshold_m',
            0.020,
        )

        self.grid_alignment_diagnostics_enabled = self._bool_param(
            'grid_alignment_diagnostics_enabled',
            True,
        )
        self.grid_alignment_expected_half_width_m = self._positive_float_param(
            'grid_alignment_expected_half_width_m',
            0.125,
        )
        self.grid_alignment_min_x_m = self._float_param(
            'grid_alignment_min_x_m',
            -0.18,
        )
        self.grid_alignment_max_x_m = self._float_param(
            'grid_alignment_max_x_m',
            0.45,
        )
        if self.grid_alignment_max_x_m <= self.grid_alignment_min_x_m:
            self.get_logger().warning(
                'Invalid grid alignment x-window; using [-0.18, 0.45]'
            )
            self.grid_alignment_min_x_m = -0.18
            self.grid_alignment_max_x_m = 0.45

        self.grid_alignment_min_side_distance_m = self._positive_float_param(
            'grid_alignment_min_side_distance_m',
            0.06,
        )
        self.grid_alignment_max_side_distance_m = self._positive_float_param(
            'grid_alignment_max_side_distance_m',
            0.45,
        )
        if (
            self.grid_alignment_max_side_distance_m
            <= self.grid_alignment_min_side_distance_m
        ):
            self.get_logger().warning(
                'Invalid grid alignment side-distance window; using [0.06, 0.45]'
            )
            self.grid_alignment_min_side_distance_m = 0.06
            self.grid_alignment_max_side_distance_m = 0.45

        self.grid_alignment_min_points = self._positive_int_param(
            'grid_alignment_min_points',
            8,
        )
        self.grid_alignment_min_span_x_m = self._positive_float_param(
            'grid_alignment_min_span_x_m',
            0.12,
        )
        self.grid_alignment_max_rms_error_m = self._positive_float_param(
            'grid_alignment_max_rms_error_m',
            0.025,
        )
        self.grid_alignment_max_abs_yaw_error_rad = self._positive_float_param(
            'grid_alignment_max_abs_yaw_error_rad',
            0.35,
        )
        self.grid_alignment_max_reported_error_m = self._positive_float_param(
            'grid_alignment_max_reported_error_m',
            0.30,
        )
        self.grid_alignment_max_reported_yaw_rad = self._positive_float_param(
            'grid_alignment_max_reported_yaw_rad',
            0.50,
        )

        self.grid_alignment_control_enabled = self._bool_param(
            'grid_alignment_control_enabled',
            False,
        )
        self.grid_yaw_correction_enabled = self._bool_param(
            'grid_yaw_correction_enabled',
            False,
        )
        self.k_grid_yaw = self._nonnegative_float_param('k_grid_yaw', 0.70)
        self.max_grid_yaw_correction_radps = self._nonnegative_float_param(
            'max_grid_yaw_correction_radps',
            0.045,
        )
        self.grid_yaw_min_confidence_for_control = self._nonnegative_float_param(
            'grid_yaw_min_confidence_for_control',
            0.60,
        )
        self.grid_yaw_max_abs_error_for_control_rad = self._positive_float_param(
            'grid_yaw_max_abs_error_for_control_rad',
            0.20,
        )
        self.grid_yaw_memory_timeout_s = self._nonnegative_float_param(
            'grid_yaw_memory_timeout_s',
            0.35,
        )

        self.grid_lateral_drift_diagnostics_enabled = self._bool_param(
            'grid_lateral_drift_diagnostics_enabled',
            True,
        )
        self.grid_lateral_drift_require_left_right = self._bool_param(
            'grid_lateral_drift_require_left_right',
            True,
        )
        self.grid_lateral_drift_min_confidence = self._nonnegative_float_param(
            'grid_lateral_drift_min_confidence',
            0.90,
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

    def _float_param(self, name: str, default: float) -> float:
        self.declare_parameter(name, default)
        value = float(self.get_parameter(name).value)
        if not math.isfinite(value):
            self.get_logger().warning(
                f'Invalid parameter {name}={value}; using {default}'
            )
            return float(default)
        return value

    def _positive_int_param(self, name: str, default: int) -> int:
        value = self.declare_parameter(name, int(default)).value
        try:
            value = int(value)
        except (TypeError, ValueError):
            raise ValueError(f'Parameter {name} must be an integer')
        if value <= 0:
            raise ValueError(f'Parameter {name} must be positive, got {value}')
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
        now = time.monotonic()
        front_clearance = self._min_range_in_sector(msg, 0.0, self.front_sector_deg)
        cardinal = cardinal_sector_ranges(
            msg,
            self.lidar_diagnostic_sector_width_deg,
            self.lidar_diagnostic_min_samples,
        )
        grid_alignment = self._grid_alignment_from_scan(msg)

        with self._scan_lock:
            self._latest_scan = msg
            self._last_scan_monotonic = now

        with self._state_lock:
            self._front_clearance_m = front_clearance
            self._front_range_m = finite_median_or_nan(cardinal['front'])
            self._rear_range_m = finite_median_or_nan(cardinal['rear'])
            self._left_range_m = finite_median_or_nan(cardinal['left'])
            self._right_range_m = finite_median_or_nan(cardinal['right'])
            self._grid_alignment = grid_alignment
            if (
                self._grid_yaw_control_memory_accepting
                and grid_alignment.valid
                and grid_alignment.yaw_valid
            ):
                self._last_valid_grid_control_alignment = grid_alignment
                self._last_valid_grid_control_alignment_monotonic = now

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
        start_alignment = self._grid_alignment_snapshot()
        start_time = time.monotonic()
        self._begin_grid_yaw_control_memory(
            start_alignment,
            accept_updates=direction > 0.0,
        )

        self.get_logger().info(
            f'{name} start grid diagnostics: '
            f'valid={start_alignment.valid}, '
            f'yaw={start_alignment.yaw_error_rad:.3f} rad, '
            f'lateral={start_alignment.lateral_error_m:.3f} m, '
            f'source={start_alignment.source}'
        )

        self._set_state(
            ControlStatus.STATE_EXECUTING,
            f'{name} executing',
            primitive_type=primitive,
        )

        result_success = False
        result_code = ExecuteMotionPrimitive.Result.INTERNAL_ERROR
        result_message = 'unknown translation result'
        final_position_error = target_distance
        final_heading_error = 0.0
        final_odom_progress = 0.0
        final_control_progress = 0.0
        final_progress_source_used = 'odom'
        final_control_progress_reason = ''
        translation_diagnostics = TranslationDiagnostics()
        final_grid_yaw_correction = GridYawCorrection(
            active=False,
            correction_radps=0.0,
            reason='not evaluated',
            source='none',
            using_memory=False,
        )
        grid_yaw_correction_ever_used = False

        while True:
            if self._cancel_or_stop_requested(goal_handle):
                self._end_grid_yaw_control_memory()
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

            progress_signed, _cross_track = self._translation_errors(
                start,
                current,
                direction,
            )
            odom_progress = max(0.0, progress_signed)
            final_odom_progress = odom_progress

            current_ranges = self._cardinal_range_snapshot()
            current_alignment = self._grid_alignment_snapshot()
            current_diagnostics = self._translation_diagnostics(
                direction=direction,
                odom_progress_m=odom_progress,
                start_ranges=start_ranges,
                end_ranges=current_ranges,
            )
            progress_selection = self._select_translation_progress(
                odom_progress_m=odom_progress,
                diagnostics=current_diagnostics,
            )

            if not progress_selection.valid:
                result_code = ExecuteMotionPrimitive.Result.INTERNAL_ERROR
                result_message = (
                    f'{name} aborted: invalid translation progress source: '
                    f'{progress_selection.reason}'
                )
                translation_diagnostics = replace(
                    current_diagnostics,
                    final_control_progress_m=progress_selection.progress_m,
                    progress_source_used=progress_selection.source,
                    control_progress_reason=progress_selection.reason,
                )
                break

            control_progress = max(0.0, progress_selection.progress_m)
            final_control_progress = control_progress
            final_progress_source_used = progress_selection.source
            final_control_progress_reason = progress_selection.reason

            remaining = target_distance - control_progress
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
                    progress_signed, _cross_track = self._translation_errors(
                        start,
                        final_snapshot.pose,
                        direction,
                    )
                    odom_progress = max(0.0, progress_signed)
                    final_odom_progress = odom_progress

                    final_ranges = self._cardinal_range_snapshot()
                    final_diagnostics = self._translation_diagnostics(
                        direction=direction,
                        odom_progress_m=odom_progress,
                        start_ranges=start_ranges,
                        end_ranges=final_ranges,
                    )
                    final_selection = self._select_translation_progress(
                        odom_progress_m=odom_progress,
                        diagnostics=final_diagnostics,
                    )

                    if not final_selection.valid:
                        result_code = ExecuteMotionPrimitive.Result.INTERNAL_ERROR
                        result_message = (
                            f'{name} final progress invalid: '
                            f'{final_selection.reason}'
                        )
                        translation_diagnostics = replace(
                            final_diagnostics,
                            final_control_progress_m=final_selection.progress_m,
                            progress_source_used=final_selection.source,
                            control_progress_reason=final_selection.reason,
                        )
                        break

                    final_control_progress = max(0.0, final_selection.progress_m)
                    final_progress_source_used = final_selection.source
                    final_control_progress_reason = final_selection.reason

                    remaining = target_distance - final_control_progress
                    heading_error = normalize_angle(start.yaw - final_snapshot.pose.yaw)
                    final_position_error = abs(remaining)
                    final_heading_error = abs(heading_error)

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

            grid_yaw = GridYawCorrection(
                active=False,
                correction_radps=0.0,
                reason='grid yaw correction only applies to forward translation',
                source='none',
                using_memory=False,
            )

            if direction > 0.0:
                grid_yaw = self._grid_yaw_correction(current_alignment)

            final_grid_yaw_correction = grid_yaw
            grid_yaw_correction_ever_used = (
                grid_yaw_correction_ever_used or grid_yaw.active
            )

            cmd.linear.y = 0.0
            cmd.angular.z = heading_correction + grid_yaw.correction_radps

            cmd = self._limiter.clamp(cmd, limits)
            cmd = self._apply_acceleration_limits(cmd)
            self._cmd_vel_pub.publish(cmd)
            self._set_grid_yaw_control_status(
                grid_yaw.active,
                grid_yaw.correction_radps,
                grid_yaw.reason,
            )

            self._update_motion_state(
                distance_remaining=max(0.0, remaining),
                distance_traveled=max(0.0, control_progress),
                heading_error=heading_error,
                status=(
                    f'{name}: remaining={remaining:.3f} m, '
                    f'progress_source={progress_selection.source}, '
                    f'odom_progress={odom_progress:.3f} m, '
                    f'control_progress={control_progress:.3f} m, '
                    f'grid_valid={current_alignment.valid}, '
                    f'grid_yaw={current_alignment.yaw_error_rad:.3f} rad, '
                    f'grid_lat={current_alignment.lateral_error_m:.3f} m, '
                    f'grid_source={current_alignment.source}, '
                    f'grid_yaw_corr={grid_yaw.correction_radps:.3f} rad/s, '
                    f'grid_yaw_corr_active={grid_yaw.active}, '
                    f'heading_error={heading_error:.3f} rad, '
                    f'angular_z_cmd={cmd.angular.z:.3f} rad/s, '
                    f'linear_y={cmd.linear.y:.3f} m/s'
                ),
            )

            self._publish_feedback(
                goal_handle,
                progress=clamp(control_progress / max(target_distance, 1e-6), 0.0, 1.0),
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

        final_selection = self._select_translation_progress(
            odom_progress_m=final_odom_progress,
            diagnostics=translation_diagnostics,
        )

        if final_selection.valid:
            final_control_progress = max(0.0, final_selection.progress_m)
            final_progress_source_used = final_selection.source
            final_control_progress_reason = final_selection.reason

        translation_diagnostics = replace(
            translation_diagnostics,
            final_control_progress_m=final_control_progress,
            progress_source_used=final_progress_source_used,
            control_progress_reason=final_control_progress_reason,
        )

        final_alignment = self._grid_alignment_snapshot()
        drift = self._lateral_drift_diagnostics(
            start_alignment,
            final_alignment,
            final_control_progress,
        )

        translation_diagnostics = replace(
            translation_diagnostics,
            grid_yaw_correction_used=grid_yaw_correction_ever_used,
            final_grid_yaw_correction_radps=final_grid_yaw_correction.correction_radps,
            grid_yaw_control_reason=final_grid_yaw_correction.reason,
            grid_lateral_start_valid=drift['start_valid'],
            grid_lateral_start_error_m=drift['start_error'],
            grid_lateral_end_valid=drift['end_valid'],
            grid_lateral_end_error_m=drift['end_error'],
            grid_lateral_drift_valid=drift['drift_valid'],
            grid_lateral_drift_m=drift['drift'],
            grid_lateral_drift_per_m=drift['drift_per_m'],
            grid_lateral_drift_reason=drift['reason'],
        )

        result_message = self._append_translation_diagnostics(
            result_message,
            translation_diagnostics,
        )
        result_message = self._append_grid_alignment_diagnostics(
            result_message,
            final_alignment,
        )

        if (
            translation_diagnostics.lidar_progress_valid
            and abs(translation_diagnostics.lidar_minus_odom_m)
            > self.lidar_odom_warning_threshold_m
        ):
            self.get_logger().warning(
                f'{name}: odom/LiDAR progress mismatch: '
                f'odom={translation_diagnostics.odom_progress_m:.3f} m, '
                f'lidar={translation_diagnostics.lidar_progress_m:.3f} m, '
                f'delta={translation_diagnostics.lidar_minus_odom_m:.3f} m'
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
            f'lidar_minus_odom={translation_diagnostics.lidar_minus_odom_m:.3f} m, '
            f'control_progress={translation_diagnostics.final_control_progress_m:.3f} m, '
            f'progress_source={translation_diagnostics.progress_source_used}, '
            f'lidar_source={translation_diagnostics.lidar_progress_source}, '
            f'lidar_disagreement={translation_diagnostics.lidar_progress_disagreement_m:.3f} m'
        )

        self._end_grid_yaw_control_memory()

        return self._finish_motion_result(
            goal_handle,
            result_success,
            result_code,
            result_message,
            final_position_error,
            final_heading_error,
            translation_diagnostics=translation_diagnostics,
            grid_alignment=final_alignment,
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
        start_alignment = self._grid_alignment_snapshot()
        start_time = time.monotonic()

        self.get_logger().info(
            'ROTATE_RELATIVE start grid diagnostics: '
            f'valid={start_alignment.valid}, '
            f'yaw={start_alignment.yaw_error_rad:.3f} rad, '
            f'source={start_alignment.source}'
        )

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
            current_alignment = self._grid_alignment_snapshot()
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
                status=(
                    f'ROTATE_RELATIVE: remaining={remaining:.3f} rad, '
                    f'grid_valid={current_alignment.valid}, '
                    f'grid_yaw={current_alignment.yaw_error_rad:.3f} rad, '
                    f'grid_source={current_alignment.source}'
                ),
            )

            self._publish_feedback(
                goal_handle,
                progress=clamp(rotated / target_abs, 0.0, 1.0),
                distance_remaining=0.0,
                heading_remaining=remaining,
                state='ROTATE_RELATIVE',
            )

            time.sleep(1.0 / self.control_rate_hz)

        final_alignment = self._grid_alignment_snapshot()
        # Deliberately diagnostic only. The current grid estimator is side-wall based and
        # is not reliable enough to refine all turns at junctions/corners/openings.
        result_message = self._append_grid_alignment_diagnostics(
            result_message,
            final_alignment,
        )

        return self._finish_motion_result(
            goal_handle,
            result_success,
            result_code,
            result_message,
            0.0,
            final_heading_error,
            grid_alignment=final_alignment,
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

    def _grid_alignment_from_scan(
        self,
        scan: Optional[LaserScan],
    ) -> GridAlignmentEstimate:
        if not self.grid_alignment_diagnostics_enabled:
            return invalid_grid_alignment('grid alignment diagnostics disabled')
        if scan is None:
            return invalid_grid_alignment('no scan')

        return estimate_grid_alignment(
            scan,
            expected_half_width_m=self.grid_alignment_expected_half_width_m,
            min_x_m=self.grid_alignment_min_x_m,
            max_x_m=self.grid_alignment_max_x_m,
            min_side_distance_m=self.grid_alignment_min_side_distance_m,
            max_side_distance_m=self.grid_alignment_max_side_distance_m,
            min_points=self.grid_alignment_min_points,
            min_span_x_m=self.grid_alignment_min_span_x_m,
            max_rms_error_m=self.grid_alignment_max_rms_error_m,
            max_abs_yaw_error_rad=self.grid_alignment_max_abs_yaw_error_rad,
            max_reported_error_m=self.grid_alignment_max_reported_error_m,
            max_reported_yaw_rad=self.grid_alignment_max_reported_yaw_rad,
        )

    def _grid_alignment_snapshot(self) -> GridAlignmentEstimate:
        scan = self._fresh_scan_copy()
        return self._grid_alignment_from_scan(scan)

    def _clear_grid_yaw_control_memory_locked(self) -> None:
        self._last_valid_grid_control_alignment = invalid_grid_alignment(
            'no valid grid control alignment for current translation'
        )
        self._last_valid_grid_control_alignment_monotonic = None
        self._grid_yaw_control_memory_accepting = False

    def _begin_grid_yaw_control_memory(
        self,
        seed: GridAlignmentEstimate,
        accept_updates: bool,
    ) -> None:
        now = time.monotonic()
        with self._state_lock:
            self._grid_yaw_control_memory_accepting = bool(accept_updates)
            if accept_updates and seed.valid and seed.yaw_valid:
                self._last_valid_grid_control_alignment = seed
                self._last_valid_grid_control_alignment_monotonic = now
            else:
                self._last_valid_grid_control_alignment = invalid_grid_alignment(
                    'no valid grid control alignment for current translation'
                )
                self._last_valid_grid_control_alignment_monotonic = None

    def _end_grid_yaw_control_memory(self) -> None:
        with self._state_lock:
            self._clear_grid_yaw_control_memory_locked()

    def _grid_alignment_for_yaw_control(
        self,
        current: GridAlignmentEstimate,
    ) -> tuple[GridAlignmentEstimate, bool, str]:
        """Return current or short-memory grid alignment for yaw control."""
        if current.valid and current.yaw_valid:
            return current, False, 'current valid grid alignment'

        if self.grid_yaw_memory_timeout_s <= 0.0:
            return current, False, f'grid yaw invalid and memory disabled: {current.reason}'

        with self._state_lock:
            memory = self._last_valid_grid_control_alignment
            memory_time = self._last_valid_grid_control_alignment_monotonic

        if memory_time is None or not memory.valid or not memory.yaw_valid:
            return current, False, f'grid yaw invalid and no valid memory: {current.reason}'

        age = time.monotonic() - memory_time
        if age > self.grid_yaw_memory_timeout_s:
            return current, False, (
                f'grid yaw invalid and memory stale: age={age:.3f}s '
                f'> {self.grid_yaw_memory_timeout_s:.3f}s; current={current.reason}'
            )

        return memory, True, (
            f'using grid yaw memory age={age:.3f}s because current invalid: {current.reason}'
        )

    def _grid_yaw_correction(
        self,
        current: GridAlignmentEstimate,
    ) -> GridYawCorrection:
        if not self.grid_alignment_control_enabled:
            return GridYawCorrection(False, 0.0, 'grid alignment control disabled', 'none', False)

        if not self.grid_yaw_correction_enabled:
            return GridYawCorrection(False, 0.0, 'grid yaw correction disabled', 'none', False)

        alignment, using_memory, memory_reason = self._grid_alignment_for_yaw_control(current)

        if not alignment.valid or not alignment.yaw_valid:
            return GridYawCorrection(False, 0.0, memory_reason, alignment.source, using_memory)

        if alignment.confidence < self.grid_yaw_min_confidence_for_control:
            return GridYawCorrection(
                False,
                0.0,
                (
                    f'grid confidence too low: {alignment.confidence:.2f} '
                    f'< {self.grid_yaw_min_confidence_for_control:.2f}'
                ),
                alignment.source,
                using_memory,
            )

        abs_error = abs(alignment.yaw_error_rad)
        if abs_error > self.grid_yaw_max_abs_error_for_control_rad:
            return GridYawCorrection(
                False,
                0.0,
                (
                    f'grid yaw error too large for control: {alignment.yaw_error_rad:.3f} rad '
                    f'> {self.grid_yaw_max_abs_error_for_control_rad:.3f} rad'
                ),
                alignment.source,
                using_memory,
            )

        correction = grid_yaw_correction_radps(
            alignment.yaw_error_rad,
            self.k_grid_yaw,
            self.max_grid_yaw_correction_radps,
        )

        reason = (
            f'grid yaw correction from {alignment.source}: '
            f'yaw_error={alignment.yaw_error_rad:.3f} rad, '
            f'confidence={alignment.confidence:.2f}, '
            f'correction={correction:.3f} rad/s'
        )
        if using_memory:
            reason += '; ' + memory_reason

        return GridYawCorrection(
            active=True,
            correction_radps=float(correction),
            reason=reason,
            source=alignment.source,
            using_memory=using_memory,
        )

    def _lateral_drift_diagnostics(
        self,
        start_alignment: GridAlignmentEstimate,
        end_alignment: GridAlignmentEstimate,
        control_progress_m: float,
    ) -> dict:
        if not self.grid_lateral_drift_diagnostics_enabled:
            return {
                'start_valid': False,
                'start_error': 0.0,
                'end_valid': False,
                'end_error': 0.0,
                'drift_valid': False,
                'drift': 0.0,
                'drift_per_m': 0.0,
                'reason': 'grid lateral drift diagnostics disabled',
            }

        def usable(alignment: GridAlignmentEstimate) -> tuple[bool, str]:
            if not alignment.valid or not alignment.lateral_valid:
                return False, alignment.reason
            if alignment.confidence < self.grid_lateral_drift_min_confidence:
                return (
                    False,
                    f'confidence too low: {alignment.confidence:.2f} '
                    f'< {self.grid_lateral_drift_min_confidence:.2f}',
                )
            if self.grid_lateral_drift_require_left_right and alignment.source != 'left_right':
                return False, f'source is {alignment.source}, left_right required'
            return True, 'usable'

        start_ok, start_reason = usable(start_alignment)
        end_ok, end_reason = usable(end_alignment)

        start_error = float(start_alignment.lateral_error_m) if start_ok else 0.0
        end_error = float(end_alignment.lateral_error_m) if end_ok else 0.0

        if not start_ok or not end_ok:
            return {
                'start_valid': start_ok,
                'start_error': start_error,
                'end_valid': end_ok,
                'end_error': end_error,
                'drift_valid': False,
                'drift': 0.0,
                'drift_per_m': 0.0,
                'reason': f'lateral drift unavailable: start={start_reason}; end={end_reason}',
            }

        drift = grid_lateral_drift(
            start_ok,
            start_error,
            end_ok,
            end_error,
            control_progress_m,
        )
        return {
            'start_valid': drift.start_valid,
            'start_error': drift.start_error_m,
            'end_valid': drift.end_valid,
            'end_error': drift.end_error_m,
            'drift_valid': drift.drift_valid,
            'drift': drift.drift_m,
            'drift_per_m': drift.drift_per_m,
            'reason': 'valid left_right grid lateral drift',
        }

    def _set_grid_yaw_control_status(
        self,
        active: bool,
        correction_radps: float,
        reason: str,
    ) -> None:
        with self._state_lock:
            self._grid_yaw_correction_active = bool(active)
            self._grid_yaw_correction_radps = float(correction_radps)
            self._grid_yaw_control_reason = str(reason)

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

        lidar_estimate = choose_lidar_progress(
            front_valid=front_valid,
            front_progress_m=front_progress,
            rear_valid=rear_valid,
            rear_progress_m=rear_progress,
            max_disagreement_m=self.lidar_progress_max_disagreement_m,
            min_progress_m=self.lidar_progress_min_m,
            allow_single_source=self.lidar_progress_allow_single_source,
        )

        lidar_progress_valid = lidar_estimate.valid
        lidar_progress = lidar_estimate.progress_m

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
            lidar_progress_source=lidar_estimate.source,
            lidar_progress_disagreement_m=lidar_estimate.disagreement_m,
            lidar_progress_reason=lidar_estimate.reason,
        )

    def _select_translation_progress(
        self,
        odom_progress_m: float,
        diagnostics: TranslationDiagnostics,
    ):
        lidar_estimate = LidarProgressEstimate(
            valid=diagnostics.lidar_progress_valid,
            progress_m=diagnostics.lidar_progress_m,
            source=diagnostics.lidar_progress_source,
            disagreement_m=diagnostics.lidar_progress_disagreement_m,
            reason=diagnostics.lidar_progress_reason,
        )

        return choose_translation_progress(
            odom_progress_m=odom_progress_m,
            lidar_estimate=lidar_estimate,
            mode=self.translation_progress_source,
            max_lidar_ahead_of_odom_m=self.lidar_progress_max_ahead_of_odom_m,
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
            f'lidar_minus_odom={diagnostics.lidar_minus_odom_m:.3f} m; '
            f'control_progress={diagnostics.final_control_progress_m:.3f} m; '
            f'progress_source={diagnostics.progress_source_used}; '
            f'lidar_source={diagnostics.lidar_progress_source}; '
            f'lidar_disagreement={diagnostics.lidar_progress_disagreement_m:.3f} m; '
            f'lidar_reason={diagnostics.lidar_progress_reason}; '
            f'control_reason={diagnostics.control_progress_reason}; '
            f'grid_yaw_correction_used={diagnostics.grid_yaw_correction_used}; '
            f'final_grid_yaw_correction={diagnostics.final_grid_yaw_correction_radps:.3f} rad/s; '
            f'grid_yaw_control_reason={diagnostics.grid_yaw_control_reason}; '
            f'grid_lateral_start_valid={diagnostics.grid_lateral_start_valid}; '
            f'grid_lateral_start_error={diagnostics.grid_lateral_start_error_m:.3f} m; '
            f'grid_lateral_end_valid={diagnostics.grid_lateral_end_valid}; '
            f'grid_lateral_end_error={diagnostics.grid_lateral_end_error_m:.3f} m; '
            f'grid_lateral_drift_valid={diagnostics.grid_lateral_drift_valid}; '
            f'grid_lateral_drift={diagnostics.grid_lateral_drift_m:.3f} m; '
            f'grid_lateral_drift_per_m={diagnostics.grid_lateral_drift_per_m:.3f}; '
            f'grid_lateral_drift_reason={diagnostics.grid_lateral_drift_reason}'
        )

    @staticmethod
    def _append_grid_alignment_diagnostics(
        message: str,
        alignment: GridAlignmentEstimate,
    ) -> str:
        return (
            f'{message}; '
            f'grid_valid={alignment.valid}; '
            f'grid_yaw_valid={alignment.yaw_valid}; '
            f'grid_yaw_error={alignment.yaw_error_rad:.3f} rad; '
            f'grid_lateral_valid={alignment.lateral_valid}; '
            f'grid_lateral_error={alignment.lateral_error_m:.3f} m; '
            f'grid_source={alignment.source}; '
            f'grid_confidence={alignment.confidence:.2f}; '
            f'grid_reason={alignment.reason}; '
            f'left_wall_valid={alignment.left.valid}; '
            f'left_wall_offset={alignment.left.offset_m:.3f} m; '
            f'left_wall_yaw={alignment.left.yaw_error_rad:.3f} rad; '
            f'left_wall_rms={alignment.left.rms_error_m:.3f} m; '
            f'left_wall_span={alignment.left.span_x_m:.3f} m; '
            f'left_wall_count={alignment.left.support_count}; '
            f'left_wall_reason={alignment.left.reason}; '
            f'right_wall_valid={alignment.right.valid}; '
            f'right_wall_offset={alignment.right.offset_m:.3f} m; '
            f'right_wall_yaw={alignment.right.yaw_error_rad:.3f} rad; '
            f'right_wall_rms={alignment.right.rms_error_m:.3f} m; '
            f'right_wall_span={alignment.right.span_x_m:.3f} m; '
            f'right_wall_count={alignment.right.support_count}; '
            f'right_wall_reason={alignment.right.reason}'
        )

    @staticmethod
    def _wall_support_count(estimate: WallLineEstimate) -> int:
        return int(max(0, min(65535, estimate.support_count)))

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
        grid_alignment: Optional[GridAlignmentEstimate] = None,
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
            grid_alignment=grid_alignment,
        )

    def _make_result(
        self,
        success: bool,
        code: int,
        message: str,
        final_position_error: float = 0.0,
        final_heading_error: float = 0.0,
        translation_diagnostics: Optional[TranslationDiagnostics] = None,
        grid_alignment: Optional[GridAlignmentEstimate] = None,
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
        result.final_control_progress_m = float(diagnostics.final_control_progress_m)
        result.progress_source_used = str(diagnostics.progress_source_used)
        result.lidar_progress_source = str(diagnostics.lidar_progress_source)
        result.lidar_progress_disagreement_m = float(
            diagnostics.lidar_progress_disagreement_m
        )
        result.lidar_progress_reason = str(diagnostics.lidar_progress_reason)
        result.control_progress_reason = str(diagnostics.control_progress_reason)

        result.grid_yaw_correction_used = bool(diagnostics.grid_yaw_correction_used)
        result.final_grid_yaw_correction_radps = float(
            diagnostics.final_grid_yaw_correction_radps
        )
        result.grid_yaw_control_reason = str(diagnostics.grid_yaw_control_reason)

        result.grid_lateral_start_valid = bool(diagnostics.grid_lateral_start_valid)
        result.grid_lateral_start_error_m = float(diagnostics.grid_lateral_start_error_m)
        result.grid_lateral_end_valid = bool(diagnostics.grid_lateral_end_valid)
        result.grid_lateral_end_error_m = float(diagnostics.grid_lateral_end_error_m)
        result.grid_lateral_drift_valid = bool(diagnostics.grid_lateral_drift_valid)
        result.grid_lateral_drift_m = float(diagnostics.grid_lateral_drift_m)
        result.grid_lateral_drift_per_m = float(diagnostics.grid_lateral_drift_per_m)
        result.grid_lateral_drift_reason = str(diagnostics.grid_lateral_drift_reason)

        alignment = grid_alignment or invalid_grid_alignment('not available')

        result.grid_alignment_valid = bool(alignment.valid)
        result.grid_yaw_valid = bool(alignment.yaw_valid)
        result.grid_yaw_error_rad = float(alignment.yaw_error_rad)
        result.grid_lateral_valid = bool(alignment.lateral_valid)
        result.grid_lateral_error_m = float(alignment.lateral_error_m)
        result.grid_alignment_source = str(alignment.source)
        result.grid_alignment_confidence = float(alignment.confidence)
        result.grid_alignment_reason = str(alignment.reason)

        result.left_wall_line_valid = bool(alignment.left.valid)
        result.left_wall_offset_m = float(alignment.left.offset_m)
        result.left_wall_yaw_error_rad = float(alignment.left.yaw_error_rad)
        result.left_wall_rms_error_m = float(alignment.left.rms_error_m)
        result.left_wall_span_x_m = float(alignment.left.span_x_m)
        result.left_wall_support_count = self._wall_support_count(alignment.left)
        result.left_wall_reason = str(alignment.left.reason)

        result.right_wall_line_valid = bool(alignment.right.valid)
        result.right_wall_offset_m = float(alignment.right.offset_m)
        result.right_wall_yaw_error_rad = float(alignment.right.yaw_error_rad)
        result.right_wall_rms_error_m = float(alignment.right.rms_error_m)
        result.right_wall_span_x_m = float(alignment.right.span_x_m)
        result.right_wall_support_count = self._wall_support_count(alignment.right)
        result.right_wall_reason = str(alignment.right.reason)
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
                self._grid_yaw_correction_active = False
                self._grid_yaw_correction_radps = 0.0
                self._grid_yaw_control_reason = ''
                self._clear_grid_yaw_control_memory_locked()

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
        self._set_grid_yaw_control_status(False, 0.0, '')

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
            grid_alignment = self._grid_alignment
            grid_yaw_correction_active = self._grid_yaw_correction_active
            grid_yaw_correction_radps = self._grid_yaw_correction_radps
            grid_yaw_control_reason = self._grid_yaw_control_reason

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
        msg.grid_alignment_valid = bool(grid_alignment.valid)
        msg.grid_yaw_valid = bool(grid_alignment.yaw_valid)
        msg.grid_yaw_error_rad = float(grid_alignment.yaw_error_rad)
        msg.grid_lateral_valid = bool(grid_alignment.lateral_valid)
        msg.grid_lateral_error_m = float(grid_alignment.lateral_error_m)
        msg.grid_alignment_source = str(grid_alignment.source)
        msg.grid_alignment_confidence = float(grid_alignment.confidence)
        msg.grid_yaw_correction_active = bool(grid_yaw_correction_active)
        msg.grid_yaw_correction_radps = float(grid_yaw_correction_radps)
        msg.grid_yaw_control_reason = str(grid_yaw_control_reason)

        msg.left_wall_line_valid = bool(grid_alignment.left.valid)
        msg.left_wall_offset_m = float(grid_alignment.left.offset_m)
        msg.left_wall_yaw_error_rad = float(grid_alignment.left.yaw_error_rad)
        msg.left_wall_rms_error_m = float(grid_alignment.left.rms_error_m)
        msg.left_wall_span_x_m = float(grid_alignment.left.span_x_m)
        msg.left_wall_support_count = self._wall_support_count(grid_alignment.left)

        msg.right_wall_line_valid = bool(grid_alignment.right.valid)
        msg.right_wall_offset_m = float(grid_alignment.right.offset_m)
        msg.right_wall_yaw_error_rad = float(grid_alignment.right.yaw_error_rad)
        msg.right_wall_rms_error_m = float(grid_alignment.right.rms_error_m)
        msg.right_wall_span_x_m = float(grid_alignment.right.span_x_m)
        msg.right_wall_support_count = self._wall_support_count(grid_alignment.right)
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
