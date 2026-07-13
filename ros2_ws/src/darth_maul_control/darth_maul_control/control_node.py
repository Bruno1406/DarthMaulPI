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
    AxialCellCenterEstimate,
    GridAlignmentEstimate,
    LidarProgressEstimate,
    SectorRange,
    TemporalLidarProgressEstimate,
    WallLineEstimate,
    cardinal_sector_ranges,
    choose_axial_cell_centering,
    choose_lidar_progress,
    choose_heading_validation_error,
    choose_temporal_lidar_progress,
    choose_translation_progress,
    estimate_grid_alignment,
    finite_median_or_nan,
    grid_lateral_drift,
    rotation_timeout_accepts_heading_error,
    invalid_grid_alignment,
)
from darth_maul_control.grid_context import (
    HEADING_COAST,
    LATERAL_COAST,
    UNAVAILABLE,
    YAW_RECOVERY,
    AxialWallReference,
    ExpectedWalls,
    GridCenteringObservation,
    GridObservation,
    GridRunContext,
    LiveGridCommand,
    VirtualCellEstimate,
    advance_cells,
    grid_context_from_goal,
    invalid_centering,
    live_grid_command,
    make_grid_observation,
    nearest_axial_wall_reference,
    observe_centering_from_expected_side_walls,
)
from darth_maul_control.grid_yaw import (
    GridYawObservation,
    estimate_manhattan_grid_yaw,
    invalid_grid_yaw,
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

    live_grid_mode: str = UNAVAILABLE
    live_grid_context_valid: bool = False
    live_grid_virtual_cell: int = 0
    live_grid_boundary_zone: bool = False
    live_grid_expected_front: bool = False
    live_grid_expected_rear: bool = False
    live_grid_expected_left: bool = False
    live_grid_expected_right: bool = False
    live_grid_yaw_active: bool = False
    live_grid_lateral_active: bool = False
    live_grid_yaw_cmd_radps: float = 0.0
    live_grid_lateral_cmd_mps: float = 0.0
    live_grid_reason: str = ''

    grid_lateral_start_valid: bool = False
    grid_lateral_start_error_m: float = 0.0
    grid_lateral_end_valid: bool = False
    grid_lateral_end_error_m: float = 0.0
    grid_lateral_drift_valid: bool = False
    grid_lateral_drift_m: float = 0.0
    grid_lateral_drift_per_m: float = 0.0
    grid_lateral_drift_reason: str = ''

    lidar_progress_temporal_degraded: bool = False
    lidar_progress_temporal_reason: str = ''


@dataclass(frozen=True)
class LiveGridDiagnostics:
    mode: str = UNAVAILABLE
    context_valid: bool = False
    virtual_cell: int = 0
    virtual_cell_progress_m: float = 0.0
    boundary_zone: bool = False
    expected_front: bool = False
    expected_rear: bool = False
    expected_left: bool = False
    expected_right: bool = False
    yaw_active: bool = False
    lateral_active: bool = False
    yaw_cmd_radps: float = 0.0
    lateral_cmd_mps: float = 0.0
    reason: str = ''


@dataclass(frozen=True)
class GridCellSettleResult:
    canceled: bool
    success: bool
    result_code: int
    message: str
    position_error_m: float
    heading_error_rad: float
    heading_source: str
    progress_m: float
    odom_progress_m: float
    progress_source: str
    progress_reason: str
    yaw_correction_used: bool


@dataclass(frozen=True)
class PostRotateManhattanSnapResult:
    canceled: bool
    correction_applied: bool
    converged: bool
    measured_abs_error_rad: Optional[float]
    reason: str


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


def _zero_twist() -> Twist:
    msg = Twist()
    msg.linear.x = 0.0
    msg.linear.y = 0.0
    msg.linear.z = 0.0
    msg.angular.x = 0.0
    msg.angular.y = 0.0
    msg.angular.z = 0.0
    return msg


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
        self._rear_clearance_m = float('inf')
        self._front_range_m = float('nan')
        self._rear_range_m = float('nan')
        self._left_range_m = float('nan')
        self._right_range_m = float('nan')
        self._grid_alignment = invalid_grid_alignment('not initialized')
        self._grid_yaw_correction_active = False
        self._grid_yaw_correction_radps = 0.0
        self._grid_yaw_control_reason = ''
        self._grid_live_mode = UNAVAILABLE
        self._grid_yaw_mode = UNAVAILABLE
        self._grid_lateral_mode = UNAVAILABLE
        self._grid_live_reacquire_samples = 0
        self._grid_live_last_observation = None
        self._grid_live_last_yaw_observation = invalid_grid_yaw('not initialized')
        self._grid_live_last_centering_observation = invalid_centering('not initialized')
        self._grid_live_last_command = LiveGridCommand(
            mode=UNAVAILABLE,
            yaw_mode=UNAVAILABLE,
            lateral_mode=UNAVAILABLE,
            yaw_active=False,
            lateral_active=False,
            angular_z_radps=0.0,
            linear_y_mps=0.0,
            speed_scale=1.0,
            reason='not initialized',
        )
        self._grid_live_last_context_valid = False
        self._grid_live_last_virtual_cell = 0
        self._grid_live_last_virtual_cell_progress_m = 0.0
        self._grid_live_last_boundary_zone = False
        self._grid_live_last_expected_front = False
        self._grid_live_last_expected_rear = False
        self._grid_live_last_expected_left = False
        self._grid_live_last_expected_right = False
        self._grid_live_last_reason = ''
        self._temporal_lidar_progress_valid = False
        self._temporal_lidar_progress_m = 0.0
        self._temporal_lidar_progress_source = 'none'
        self._temporal_lidar_degraded_samples = 0

        self._active_goal = False
        self._stop_requested = False
        self._cancel_requested = False

        self._current_odom: Optional[Odometry] = None
        self._last_odom_monotonic: Optional[float] = None

        self._latest_scan: Optional[LaserScan] = None
        self._last_scan_monotonic: Optional[float] = None

        self._last_commanded_twist = Twist()
        self._last_command_time = time.monotonic()
        self._settle_consumed_for_transition = False

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
            0.015,
        )
        self.default_heading_tolerance_rad = self._positive_float_param(
            'default_heading_tolerance_rad',
            0.060,
        )

        self.default_linear_speed_mps = self._positive_float_param(
            'default_linear_speed_mps',
            0.135,
        )
        self.default_reverse_speed_mps = self._positive_float_param(
            'default_reverse_speed_mps',
            0.055,
        )
        self.default_angular_speed_radps = self._positive_float_param(
            'default_angular_speed_radps',
            0.420,
        )

        self.max_linear_x_mps = self._positive_float_param('max_linear_x_mps', 0.150)
        self.max_linear_y_mps = self._nonnegative_float_param(
            'max_linear_y_mps',
            0.050,
        )
        self.max_angular_z_radps = self._positive_float_param(
            'max_angular_z_radps',
            0.550,
        )

        self.min_linear_x_mps = self._nonnegative_float_param(
            'min_linear_x_mps',
            0.025,
        )
        self.min_angular_z_radps = self._nonnegative_float_param(
            'min_angular_z_radps',
            0.080,
        )

        self.k_distance = self._nonnegative_float_param('k_distance', 0.90)
        self.k_heading = self._nonnegative_float_param('k_heading', 1.70)

        self.max_linear_accel_mps2 = self._positive_float_param(
            'max_linear_accel_mps2',
            0.360,
        )
        self.max_lateral_accel_mps2 = self._positive_float_param(
            'max_lateral_accel_mps2',
            0.180,
        )
        self.max_angular_accel_radps2 = self._positive_float_param(
            'max_angular_accel_radps2',
            1.300,
        )

        self.odom_timeout_sec = self._positive_float_param('odom_timeout_sec', 0.50)
        self.scan_timeout_sec = self._positive_float_param('scan_timeout_sec', 0.50)

        self.require_scan_for_collision_check = self._bool_param(
            'require_scan_for_collision_check',
            True,
        )
        self.collision_sector_deg = self._positive_float_param(
            'collision_sector_deg',
            35.0,
        )
        self.front_stop_distance_m = self._positive_float_param(
            'front_stop_distance_m',
            0.09,
        )
        self.rear_stop_distance_m = self._positive_float_param(
            'rear_stop_distance_m',
            0.09,
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
            'lidar_required',
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

        self.translation_lidar_required_invalid_max_consecutive_samples = (
            self._nonnegative_int_param(
                'translation_lidar_required_invalid_max_consecutive_samples',
                0,
            )
        )
        self.translation_lidar_recovery_enabled = self._bool_param(
            'translation_lidar_recovery_enabled',
            True,
        )
        self.translation_lidar_recovery_linear_x_mps = self._positive_float_param(
            'translation_lidar_recovery_linear_x_mps',
            0.045,
        )
        self.translation_stall_detection_enabled = (
            self._bool_param(
                'translation_stall_detection_enabled',
                True,
            )
        )
        self.translation_stall_grace_sec = (
            self._nonnegative_float_param(
                'translation_stall_grace_sec',
                0.75,
            )
        )
        self.translation_stall_window_sec = (
            self._positive_float_param(
                'translation_stall_window_sec',
                0.80,
            )
        )
        self.translation_stall_min_command_mps = (
            self._positive_float_param(
                'translation_stall_min_command_mps',
                0.080,
            )
        )
        self.translation_stall_min_progress_m = (
            self._positive_float_param(
                'translation_stall_min_progress_m',
                0.008,
            )
        )
        self.translation_stall_min_remaining_m = (
            self._positive_float_param(
                'translation_stall_min_remaining_m',
                0.050,
            )
        )

        self.lidar_progress_max_disagreement_m = self._positive_float_param(
            'lidar_progress_max_disagreement_m',
            0.035,
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
            0.075,
        )
        self.lidar_progress_odom_arbitration_tolerance_m = (
            self._positive_float_param(
                'lidar_progress_odom_arbitration_tolerance_m',
                0.060,
            )
        )
        self.lidar_progress_odom_arbitration_min_margin_m = (
            self._nonnegative_float_param(
                'lidar_progress_odom_arbitration_min_margin_m',
                0.010,
            )
        )
        self.lidar_odom_warning_threshold_m = self._positive_float_param(
            'lidar_odom_warning_threshold_m',
            0.030,
        )

        self.grid_alignment_diagnostics_enabled = self._bool_param(
            'grid_alignment_diagnostics_enabled',
            True,
        )
        self.grid_alignment_expected_half_width_m = self._positive_float_param(
            'grid_alignment_expected_half_width_m',
            0.125,
        )
        self.grid_alignment_adjacent_wall_tolerance_m = self._positive_float_param(
            'grid_alignment_adjacent_wall_tolerance_m',
            0.080,
        )
        self.grid_alignment_pair_width_tolerance_m = self._positive_float_param(
            'grid_alignment_pair_width_tolerance_m',
            0.080,
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

        self.grid_live_control_enabled = self._bool_param(
            'grid_live_control_enabled',
            True,
        )
        self.k_grid_lateral = self._nonnegative_float_param('k_grid_lateral', 1.60)
        self.max_grid_lateral_mps = self._nonnegative_float_param(
            'max_grid_lateral_mps',
            0.045,
        )
        self.grid_lateral_min_confidence = self._nonnegative_float_param(
            'grid_lateral_min_confidence',
            0.55,
        )
        self.k_grid_live_yaw = self._nonnegative_float_param('k_grid_live_yaw', 2.40)
        self.max_grid_live_yaw_correction_radps = self._nonnegative_float_param(
            'max_grid_live_yaw_correction_radps',
            0.140,
        )
        self.grid_heading_coast_max_odom_yaw_radps = self._nonnegative_float_param(
            'grid_heading_coast_max_odom_yaw_radps',
            0.120,
        )
        self.grid_heading_coast_speed_scale = self._nonnegative_float_param(
            'grid_heading_coast_speed_scale',
            0.35,
        )
        self.grid_heading_coast_stop_odom_yaw_radps = self._nonnegative_float_param(
            'grid_heading_coast_stop_odom_yaw_radps',
            0.450,
        )
        self.grid_manhattan_yaw_enabled = self._bool_param(
            'grid_manhattan_yaw_enabled',
            True,
        )
        self.grid_manhattan_yaw_min_range_m = self._positive_float_param(
            'grid_manhattan_yaw_min_range_m',
            0.08,
        )
        self.grid_manhattan_yaw_max_range_m = self._positive_float_param(
            'grid_manhattan_yaw_max_range_m',
            2.50,
        )
        self.grid_manhattan_yaw_max_point_gap_m = self._positive_float_param(
            'grid_manhattan_yaw_max_point_gap_m',
            0.055,
        )
        self.grid_manhattan_yaw_max_range_jump_m = self._positive_float_param(
            'grid_manhattan_yaw_max_range_jump_m',
            0.080,
        )
        self.grid_manhattan_yaw_min_cluster_points = self._positive_int_param(
            'grid_manhattan_yaw_min_cluster_points',
            8,
        )
        self.grid_manhattan_yaw_min_segment_points = self._positive_int_param(
            'grid_manhattan_yaw_min_segment_points',
            8,
        )
        self.grid_manhattan_yaw_min_segment_length_m = self._positive_float_param(
            'grid_manhattan_yaw_min_segment_length_m',
            0.120,
        )
        self.grid_manhattan_yaw_max_line_rms_m = self._positive_float_param(
            'grid_manhattan_yaw_max_line_rms_m',
            0.020,
        )
        self.grid_manhattan_yaw_min_line_count = self._positive_int_param(
            'grid_manhattan_yaw_min_line_count',
            1,
        )
        self.grid_manhattan_yaw_min_total_weight = self._positive_float_param(
            'grid_manhattan_yaw_min_total_weight',
            0.20,
        )
        self.grid_manhattan_yaw_min_concentration = self._nonnegative_float_param(
            'grid_manhattan_yaw_min_concentration',
            0.70,
        )
        self.grid_manhattan_yaw_min_confidence = self._nonnegative_float_param(
            'grid_manhattan_yaw_min_confidence',
            0.55,
        )
        self.grid_manhattan_yaw_max_abs_error_rad = self._positive_float_param(
            'grid_manhattan_yaw_max_abs_error_rad',
            0.140,
        )

        if self.grid_manhattan_yaw_max_range_m <= self.grid_manhattan_yaw_min_range_m:
            self.get_logger().warning(
                'Invalid grid Manhattan yaw range window; using [0.08, 2.50]'
            )
            self.grid_manhattan_yaw_min_range_m = 0.08
            self.grid_manhattan_yaw_max_range_m = 2.50

        self.grid_manhattan_yaw_min_concentration = max(
            0.0,
            min(1.0, self.grid_manhattan_yaw_min_concentration),
        )
        self.grid_manhattan_yaw_min_confidence = max(
            0.0,
            min(1.0, self.grid_manhattan_yaw_min_confidence),
        )
        self.grid_live_max_abs_yaw_error_rad = self._positive_float_param(
            'grid_live_max_abs_yaw_error_rad',
            0.120,
        )
        self.grid_live_min_span_x_m = self._positive_float_param(
            'grid_live_min_span_x_m',
            0.120,
        )
        self.grid_live_min_support_count = self._positive_int_param(
            'grid_live_min_support_count',
            8,
        )
        self.grid_live_max_rms_error_m = self._positive_float_param(
            'grid_live_max_rms_error_m',
            0.025,
        )
        self.grid_virtual_cell_boundary_margin_m = self._nonnegative_float_param(
            'grid_virtual_cell_boundary_margin_m',
            0.025,
        )
        self.grid_reacquire_stable_samples = self._positive_int_param(
            'grid_reacquire_stable_samples',
            3,
        )
        self.grid_reacquire_small_yaw_rad = self._positive_float_param(
            'grid_reacquire_small_yaw_rad',
            0.055,
        )
        self.grid_reacquire_large_yaw_rad = self._positive_float_param(
            'grid_reacquire_large_yaw_rad',
            0.200,
        )
        self.grid_reacquire_speed_scale = self._float_param(
            'grid_reacquire_speed_scale',
            0.65,
        )
        self.grid_reacquire_speed_scale = max(
            0.0,
            min(1.0, self.grid_reacquire_speed_scale),
        )

        self.grid_cell_settle_enabled = self._bool_param(
            'grid_cell_settle_enabled',
            True,
        )
        self.grid_cell_settle_timeout_sec = self._positive_float_param(
            'grid_cell_settle_timeout_sec',
            2.5,
        )
        self.grid_cell_settle_max_linear_x_mps = self._positive_float_param(
            'grid_cell_settle_max_linear_x_mps',
            0.035,
        )
        self.grid_cell_settle_min_linear_x_mps = self._nonnegative_float_param(
            'grid_cell_settle_min_linear_x_mps',
            0.010,
        )
        self.grid_cell_settle_max_linear_y_mps = self._nonnegative_float_param(
            'grid_cell_settle_max_linear_y_mps',
            0.030,
        )
        self.grid_cell_settle_max_yaw_radps = self._nonnegative_float_param(
            'grid_cell_settle_max_yaw_radps',
            0.100,
        )
        self.grid_cell_settle_position_tolerance_m = self._positive_float_param(
            'grid_cell_settle_position_tolerance_m',
            0.010,
        )
        self.grid_cell_settle_latch_immediate_axial_once_seen = self._bool_param(
            'grid_cell_settle_latch_immediate_axial_once_seen',
            True,
        )
        self.grid_cell_settle_disable_progress_after_axial_seen = self._bool_param(
            'grid_cell_settle_disable_progress_after_axial_seen',
            True,
        )
        self.grid_cell_settle_compact_transit_accept_enabled = self._bool_param(
            'grid_cell_settle_compact_transit_accept_enabled',
            True,
        )
        self.grid_cell_settle_compact_transit_min_run_cells = self._positive_int_param(
            'grid_cell_settle_compact_transit_min_run_cells',
            2,
        )
        self.grid_cell_settle_compact_transit_position_tolerance_m = (
            self._positive_float_param(
                'grid_cell_settle_compact_transit_position_tolerance_m',
                0.030,
            )
        )
        self.grid_cell_settle_compact_transit_lateral_tolerance_m = (
            self._positive_float_param(
                'grid_cell_settle_compact_transit_lateral_tolerance_m',
                0.045,
            )
        )
        self.grid_cell_settle_compact_transit_heading_tolerance_rad = (
            self._positive_float_param(
                'grid_cell_settle_compact_transit_heading_tolerance_rad',
                0.080,
            )
        )
        self.grid_cell_settle_compact_transit_lidar_progress_strict = (
            self._bool_param(
                'grid_cell_settle_compact_transit_lidar_progress_strict',
                True,
            )
        )
        self.grid_cell_settle_compact_transit_mask_returned_progress_error = (
            self._bool_param(
                'grid_cell_settle_compact_transit_mask_returned_progress_error',
                False,
            )
        )
        self.grid_cell_settle_attempt_after_final_progress_invalid = (
            self._bool_param(
                'grid_cell_settle_attempt_after_final_progress_invalid',
                True,
            )
        )
        self.grid_cell_settle_final_invalid_max_odom_error_m = (
            self._positive_float_param(
                'grid_cell_settle_final_invalid_max_odom_error_m',
                0.200,
            )
        )
        self.grid_cell_settle_rejected_axial_single_wall_enabled = (
            self._bool_param(
                'grid_cell_settle_rejected_axial_single_wall_enabled',
                True,
            )
        )
        self.grid_cell_settle_rejected_axial_single_wall_max_error_m = (
            self._positive_float_param(
                'grid_cell_settle_rejected_axial_single_wall_max_error_m',
                0.080,
            )
        )
        self.grid_cell_settle_projected_axial_enabled = self._bool_param(
            'grid_cell_settle_projected_axial_enabled',
            True,
        )
        self.grid_cell_settle_projected_axial_min_run_cells = (
            self._positive_int_param(
                'grid_cell_settle_projected_axial_min_run_cells',
                2,
            )
        )
        self.grid_cell_settle_projected_axial_max_open_cells = (
            self._positive_int_param(
                'grid_cell_settle_projected_axial_max_open_cells',
                4,
            )
        )
        self.grid_cell_settle_projected_axial_max_distance_m = (
            self._positive_float_param(
                'grid_cell_settle_projected_axial_max_distance_m',
                1.200,
            )
        )
        self.grid_cell_settle_projected_axial_max_error_m = (
            self._positive_float_param(
                'grid_cell_settle_projected_axial_max_error_m',
                0.160,
            )
        )
        self.grid_cell_settle_projected_axial_front_rear_agreement_m = (
            self._positive_float_param(
                'grid_cell_settle_projected_axial_front_rear_agreement_m',
                0.120,
            )
        )
        self.grid_cell_settle_open_corridor_lattice_enabled = self._bool_param(
            'grid_cell_settle_open_corridor_lattice_enabled',
            True,
        )
        self.grid_cell_settle_open_corridor_lattice_max_distance_m = (
            self._positive_float_param(
                'grid_cell_settle_open_corridor_lattice_max_distance_m',
                1.800,
            )
        )
        self.grid_cell_settle_open_corridor_lattice_max_error_m = (
            self._positive_float_param(
                'grid_cell_settle_open_corridor_lattice_max_error_m',
                0.080,
            )
        )
        self.grid_cell_settle_open_corridor_lattice_agreement_m = (
            self._positive_float_param(
                'grid_cell_settle_open_corridor_lattice_agreement_m',
                0.080,
            )
        )
        self.grid_cell_settle_lateral_tolerance_m = self._positive_float_param(
            'grid_cell_settle_lateral_tolerance_m',
            0.020,
        )
        self.grid_cell_settle_heading_tolerance_rad = self._positive_float_param(
            'grid_cell_settle_heading_tolerance_rad',
            0.060,
        )
        self.grid_cell_settle_required_stable_samples = self._positive_int_param(
            'grid_cell_settle_required_stable_samples',
            4,
        )
        self.grid_cell_settle_min_duration_sec = self._nonnegative_float_param(
            'grid_cell_settle_min_duration_sec',
            0.30,
        )
        self.grid_cell_settle_require_longitudinal_reference = self._bool_param(
            'grid_cell_settle_require_longitudinal_reference',
            False,
        )
        self.grid_cell_settle_abort_position_error_m = self._positive_float_param(
            'grid_cell_settle_abort_position_error_m',
            0.120,
        )
        self.grid_cell_settle_abort_heading_error_rad = self._positive_float_param(
            'grid_cell_settle_abort_heading_error_rad',
            0.220,
        )
        self.grid_cell_settle_travel_guard_enabled = self._bool_param(
            'grid_cell_settle_travel_guard_enabled',
            True,
        )
        self.grid_cell_settle_min_front_distance_m = self._positive_float_param(
            'grid_cell_settle_min_front_distance_m',
            0.120,
        )
        self.grid_cell_settle_min_rear_distance_m = self._positive_float_param(
            'grid_cell_settle_min_rear_distance_m',
            0.105,
        )
        self.grid_center_expected_front_distance_m = self._positive_float_param(
            'grid_center_expected_front_distance_m',
            0.125,
        )
        self.grid_center_expected_rear_distance_m = self._positive_float_param(
            'grid_center_expected_rear_distance_m',
            0.125,
        )
        self.grid_center_front_rear_agreement_tolerance_m = self._positive_float_param(
            'grid_center_front_rear_agreement_tolerance_m',
            0.035,
        )
        self.lidar_progress_temporal_filter_enabled = self._bool_param(
            'lidar_progress_temporal_filter_enabled',
            True,
        )
        self.lidar_progress_temporal_max_backtrack_m = self._positive_float_param(
            'lidar_progress_temporal_max_backtrack_m',
            0.020,
        )
        self.lidar_progress_temporal_max_jump_m = self._positive_float_param(
            'lidar_progress_temporal_max_jump_m',
            0.080,
        )
        self.lidar_progress_temporal_max_degraded_samples = self._positive_int_param(
            'lidar_progress_temporal_max_degraded_samples',
            6,
        )
        self.rotate_timeout_accept_heading_error_rad = self._nonnegative_float_param(
            'rotate_timeout_accept_heading_error_rad',
            0.070,
        )
        self.rotate_timeout_margin_sec = self._nonnegative_float_param(
            'rotate_timeout_margin_sec',
            8.0,
        )
        self.rotate_timeout_recovery_extra_sec = self._nonnegative_float_param(
            'rotate_timeout_recovery_extra_sec',
            6.0,
        )
        self.rotate_pre_clearance_settle_enabled = self._bool_param(
            'rotate_pre_clearance_settle_enabled',
            True,
        )
        self.rotate_pre_clearance_front_min_m = self._positive_float_param(
            'rotate_pre_clearance_front_min_m',
            0.155,
        )
        self.rotate_pre_clearance_rear_min_m = self._positive_float_param(
            'rotate_pre_clearance_rear_min_m',
            0.155,
        )
        self.rotate_pre_clearance_target_front_m = self._positive_float_param(
            'rotate_pre_clearance_target_front_m',
            0.145,
        )
        self.rotate_pre_clearance_target_rear_m = self._positive_float_param(
            'rotate_pre_clearance_target_rear_m',
            0.135,
        )
        self.rotate_pre_clearance_timeout_sec = self._positive_float_param(
            'rotate_pre_clearance_timeout_sec',
            1.0,
        )
        self.rotate_pre_clearance_max_linear_x_mps = self._positive_float_param(
            'rotate_pre_clearance_max_linear_x_mps',
            0.045,
        )
        self.rotate_post_manhattan_snap_enabled = self._bool_param(
            'rotate_post_manhattan_snap_enabled',
            True,
        )
        self.rotate_post_manhattan_snap_acquire_timeout_sec = (
            self._positive_float_param(
                'rotate_post_manhattan_snap_acquire_timeout_sec',
                0.60,
            )
        )
        self.rotate_post_manhattan_snap_timeout_sec = self._positive_float_param(
            'rotate_post_manhattan_snap_timeout_sec',
            2.0,
        )
        self.rotate_post_manhattan_snap_tolerance_rad = self._positive_float_param(
            'rotate_post_manhattan_snap_tolerance_rad',
            0.040,
        )
        self.rotate_post_manhattan_snap_max_abs_error_rad = (
            self._positive_float_param(
                'rotate_post_manhattan_snap_max_abs_error_rad',
                0.200,
            )
        )
        self.rotate_post_manhattan_snap_max_yaw_radps = (
            self._positive_float_param(
                'rotate_post_manhattan_snap_max_yaw_radps',
                0.140,
            )
        )
        self.rotate_post_manhattan_snap_min_confidence = (
            self._nonnegative_float_param(
                'rotate_post_manhattan_snap_min_confidence',
                0.70,
            )
        )
        self.rotate_post_manhattan_snap_confirm_samples = self._positive_int_param(
            'rotate_post_manhattan_snap_confirm_samples',
            2,
        )
        self.rotate_post_manhattan_snap_stable_samples = self._positive_int_param(
            'rotate_post_manhattan_snap_stable_samples',
            2,
        )
        self.rotate_post_manhattan_snap_max_sample_delta_rad = (
            self._positive_float_param(
                'rotate_post_manhattan_snap_max_sample_delta_rad',
                0.035,
            )
        )
        self.rotate_post_manhattan_snap_max_scan_hold_sec = (
            self._positive_float_param(
                'rotate_post_manhattan_snap_max_scan_hold_sec',
                0.20,
            )
        )
        self.rotate_post_manhattan_snap_min_confidence = max(
            0.0,
            min(1.0, self.rotate_post_manhattan_snap_min_confidence),
        )

        self.grid_lateral_drift_diagnostics_enabled = self._bool_param(
            'grid_lateral_drift_diagnostics_enabled',
            True,
        )
        self.grid_lateral_drift_require_left_right = self._bool_param(
            'grid_lateral_drift_require_left_right',
            False,
        )
        self.grid_lateral_drift_min_confidence = self._nonnegative_float_param(
            'grid_lateral_drift_min_confidence',
            0.55,
        )

        self.timeout_margin_sec = self._nonnegative_float_param(
            'timeout_margin_sec',
            5.0,
        )
        self.wait_timeout_margin_sec = self._nonnegative_float_param(
            'wait_timeout_margin_sec',
            1.0,
        )

        self.stop_publish_duration_sec = self._positive_float_param(
            'stop_publish_duration_sec',
            0.30,
        )
        self.stop_publish_rate_hz = self._positive_float_param(
            'stop_publish_rate_hz',
            25.0,
        )
        self.settle_time_sec = self._nonnegative_float_param('settle_time_sec', 0.10)
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

    def _nonnegative_int_param(self, name: str, default: int) -> int:
        value = self.declare_parameter(name, int(default)).value
        try:
            value = int(value)
        except (TypeError, ValueError):
            raise ValueError(f'Parameter {name} must be an integer')
        if value < 0:
            raise ValueError(f'Parameter {name} must be nonnegative, got {value}')
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
        front_clearance = self._min_range_in_sector(msg, 0.0, self.collision_sector_deg)
        rear_clearance = self._min_range_in_sector(msg, math.pi, self.collision_sector_deg)
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
            self._rear_clearance_m = rear_clearance
            self._front_range_m = finite_median_or_nan(cardinal['front'])
            self._rear_range_m = finite_median_or_nan(cardinal['rear'])
            self._left_range_m = finite_median_or_nan(cardinal['left'])
            self._right_range_m = finite_median_or_nan(cardinal['right'])
            self._grid_alignment = grid_alignment

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
        lidar_required_mode = self.translation_progress_source == 'lidar_required'
        grid_context = grid_context_from_goal(request)
        self._reset_live_grid_controller('translation start')
        self._settle_consumed_for_transition = False

        if bool(request.collision_check_enabled):
            if self.require_scan_for_collision_check and not self._is_scan_fresh():
                if not lidar_required_mode:
                    return self._fail_goal(
                        goal_handle,
                        ExecuteMotionPrimitive.Result.OBSTACLE_TOO_CLOSE,
                        f'{name} requires fresh LiDAR scan',
                    )

            clearance, clearance_name, stop_distance = self._travel_clearance(direction)
            if not math.isfinite(clearance):
                clearance = float('inf')

            if clearance < stop_distance:
                return self._fail_goal(
                    goal_handle,
                    ExecuteMotionPrimitive.Result.OBSTACLE_TOO_CLOSE,
                    f'{name} blocked before start: '
                    f'{clearance_name}={clearance:.3f} m < {stop_distance:.3f} m',
                )

        start_snapshot = self._get_motion_snapshot()
        if start_snapshot is None:
            return self._odom_unavailable_result(goal_handle)

        start = start_snapshot.pose
        start_ranges = self._cardinal_range_snapshot()
        start_alignment = self._grid_alignment_snapshot()
        start_time = time.monotonic()
        self._reset_temporal_lidar_progress_tracker()

        self.get_logger().info(
            f'{name} start grid diagnostics: '
            f'context_valid={grid_context.valid}, '
            f'context_reason={grid_context.reason}, '
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
        settle_result_message = 'cell settle not run'
        final_position_error = target_distance
        final_heading_error = 0.0
        final_odom_progress = 0.0
        final_control_progress = 0.0
        final_progress_source_used = (
            'lidar_required_unavailable' if lidar_required_mode else 'odom'
        )
        final_control_progress_reason = ''
        translation_diagnostics = TranslationDiagnostics()
        grid_yaw_correction_ever_used = False
        lidar_required_invalid_consecutive_samples = 0
        preserve_failure_progress_diagnostics = False
        force_cell_settle_after_invalid_final_progress = False
        stall_anchor_time = start_time
        stall_anchor_progress_m = 0.0

        lidar_required_start_acquired = True
        if lidar_required_mode:
            start_diagnostics = self._translation_diagnostics(
                direction=direction,
                odom_progress_m=0.0,
                start_ranges=start_ranges,
                end_ranges=start_ranges,
                track_lidar_progress=True,
            )
            start_selection = self._select_translation_progress(
                odom_progress_m=0.0,
                diagnostics=start_diagnostics,
            )
            lidar_required_start_acquired = bool(start_selection.valid)

        while True:
            if self._cancel_or_stop_requested(goal_handle):
                self._reset_live_grid_controller('translation canceled')
                self._reset_temporal_lidar_progress_tracker()
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
            if lidar_required_mode and not lidar_required_start_acquired:
                current_diagnostics = self._translation_diagnostics(
                    direction=direction,
                    odom_progress_m=0.0,
                    start_ranges=current_ranges,
                    end_ranges=current_ranges,
                    track_lidar_progress=True,
                )
                progress_selection = self._select_translation_progress(
                    odom_progress_m=0.0,
                    diagnostics=current_diagnostics,
                )
                if progress_selection.valid:
                    start = current
                    start_ranges = current_ranges
                    start_alignment = current_alignment
                    self._reset_live_grid_controller('lidar baseline acquired')
                    self._reset_temporal_lidar_progress_tracker()
                    odom_progress = 0.0
                    final_odom_progress = 0.0
                    lidar_required_start_acquired = True
            else:
                current_diagnostics = self._translation_diagnostics(
                    direction=direction,
                    odom_progress_m=odom_progress,
                    start_ranges=start_ranges,
                    end_ranges=current_ranges,
                    track_lidar_progress=True,
                )
                progress_selection = self._select_translation_progress(
                    odom_progress_m=odom_progress,
                    diagnostics=current_diagnostics,
                )

            heading_error = normalize_angle(start.yaw - current.yaw)
            progress_selection = self._maybe_use_geometry_aware_odom_progress(
                progress_selection=progress_selection,
                grid_context=grid_context,
                alignment=current_alignment,
                odom_progress_m=odom_progress,
                heading_error_rad=heading_error,
                direction=direction,
                target_distance_m=target_distance,
            )

            if not progress_selection.valid:
                stall_anchor_time = time.monotonic()
                stall_anchor_progress_m = (
                    final_control_progress
                )

                if lidar_required_mode and self.translation_lidar_recovery_enabled:
                    lidar_required_invalid_consecutive_samples += 1
                    final_progress_source_used = progress_selection.source
                    final_control_progress_reason = progress_selection.reason
                    translation_diagnostics = replace(
                        current_diagnostics,
                        final_control_progress_m=final_control_progress,
                        progress_source_used=final_progress_source_used,
                        control_progress_reason=final_control_progress_reason,
                    )

                    clearance = float('inf')
                    clearance_name = 'none'
                    stop_distance = 0.0
                    if bool(request.collision_check_enabled):
                        clearance, clearance_name, stop_distance = self._travel_clearance(direction)
                        if not math.isfinite(clearance):
                            clearance = float('inf')

                    if bool(request.collision_check_enabled) and clearance < stop_distance:
                        # Safety still wins. Do not drive into a wall. But do not
                        # kill the action here; keep the action alive so fresh
                        # LiDAR/settle can recover if geometry becomes observable.
                        self.publish_zero_twist()
                        recovery_reason = (
                            f'LiDAR recovery hold: {clearance_name}={clearance:.3f} m '
                            f'< {stop_distance:.3f} m; '
                            f'progress_invalid={progress_selection.reason}'
                        )
                    else:
                        raw_heading_correction = self.k_heading * heading_error
                        live_observation = self._live_grid_observation(
                            grid_context,
                            current_alignment,
                            final_control_progress,
                        )
                        live_command = self._live_grid_command(
                            live_observation,
                            odom_heading_correction_radps=raw_heading_correction,
                        )

                        recovery_speed = min(
                            max_speed,
                            float(self.translation_lidar_recovery_linear_x_mps),
                        )
                        recovery_speed *= max(0.20, float(live_command.speed_scale))

                        cmd = Twist()
                        cmd.linear.x = direction * recovery_speed
                        cmd.linear.y = (
                            live_command.linear_y_mps
                            if live_command.lateral_active
                            else 0.0
                        )
                        cmd.angular.z = live_command.angular_z_radps
                        cmd = self._limiter.clamp(cmd, limits)
                        cmd = self._apply_acceleration_limits(cmd)
                        self._publish_drive_cmd(
                            forward=cmd.linear.x,
                            lateral=cmd.linear.y,
                            yaw_rate=cmd.angular.z,
                            allow_lateral=True,
                        )

                        grid_yaw_correction_ever_used = (
                            grid_yaw_correction_ever_used or live_command.yaw_active
                        )

                        recovery_reason = (
                            f'LiDAR recovery crawl: invalid_samples='
                            f'{lidar_required_invalid_consecutive_samples}; '
                            f'odom_progress={odom_progress:.3f} m used only for diagnostics; '
                            f'last_trusted_lidar_progress={final_control_progress:.3f} m; '
                            f'cmd_x={cmd.linear.x:.3f} m/s; '
                            f'cmd_y={cmd.linear.y:.3f} m/s; '
                            f'cmd_yaw={cmd.angular.z:.3f} rad/s; '
                            f'live_grid_mode={live_command.mode}; '
                            f'live_reason={live_command.reason}; '
                            f'progress_reason={progress_selection.reason}'
                        )

                    self._update_motion_state(
                        distance_remaining=max(0.0, target_distance - final_control_progress),
                        distance_traveled=max(0.0, final_control_progress),
                        heading_error=heading_error,
                        status=recovery_reason,
                    )
                    self._publish_feedback(
                        goal_handle,
                        progress=clamp(
                            final_control_progress / max(target_distance, 1e-6),
                            0.0,
                            1.0,
                        ),
                        distance_remaining=max(0.0, target_distance - final_control_progress),
                        heading_remaining=heading_error,
                        state=f'{name}_lidar_recovery',
                    )
                    time.sleep(1.0 / self.control_rate_hz)
                    continue

                result_code = ExecuteMotionPrimitive.Result.INTERNAL_ERROR
                result_message = (
                    f'{name} aborted: invalid translation progress source and '
                    f'LiDAR recovery disabled: '
                    f'{progress_selection.reason}'
                )
                translation_diagnostics = replace(
                    current_diagnostics,
                    final_control_progress_m=final_control_progress,
                    progress_source_used=progress_selection.source,
                    control_progress_reason=progress_selection.reason,
                )
                break

            lidar_required_invalid_consecutive_samples = 0
            control_progress = max(0.0, progress_selection.progress_m)
            final_control_progress = control_progress
            final_progress_source_used = progress_selection.source
            final_control_progress_reason = progress_selection.reason

            remaining = target_distance - control_progress

            final_position_error = abs(remaining)
            final_heading_error = abs(heading_error)

            now_monotonic = time.monotonic()

            last_commanded_forward_mps = abs(
                float(
                    self._last_commanded_twist.linear.x
                )
            )

            lidar_is_physical_progress = bool(
                progress_selection.valid
                and progress_selection.source == 'lidar'
            )

            stall_watch_active = bool(
                self.translation_stall_detection_enabled
                and lidar_is_physical_progress
                and elapsed
                >= self.translation_stall_grace_sec
                and remaining
                >= self.translation_stall_min_remaining_m
                and last_commanded_forward_mps
                >= self.translation_stall_min_command_mps
            )

            progress_since_anchor = max(
                0.0,
                control_progress
                - stall_anchor_progress_m,
            )

            if not stall_watch_active:
                stall_anchor_time = now_monotonic
                stall_anchor_progress_m = control_progress

            elif (
                progress_since_anchor
                >= self.translation_stall_min_progress_m
            ):
                stall_anchor_time = now_monotonic
                stall_anchor_progress_m = control_progress

            elif (
                now_monotonic
                - stall_anchor_time
                >= self.translation_stall_window_sec
            ):
                self._flush_stop(
                    n=10,
                    dt=0.03,
                )

                result_code = (
                    ExecuteMotionPrimitive.Result.TIMEOUT
                )

                result_message = (
                    f'{name} physical no-progress stall: '
                    f'trusted LiDAR progress increased only '
                    f'{progress_since_anchor:.3f} m in '
                    f'{now_monotonic - stall_anchor_time:.2f}s '
                    f'while commanding '
                    f'{last_commanded_forward_mps:.3f} m/s; '
                    f'control_progress={control_progress:.3f} m; '
                    f'remaining={remaining:.3f} m; '
                    f'odom_progress={odom_progress:.3f} m '
                    'diagnostic_only'
                )

                final_control_progress = control_progress
                final_progress_source_used = (
                    progress_selection.source
                )
                final_control_progress_reason = (
                    progress_selection.reason
                    + '; physical no-progress stall'
                )

                translation_diagnostics = replace(
                    current_diagnostics,
                    final_control_progress_m=(
                        final_control_progress
                    ),
                    progress_source_used=(
                        final_progress_source_used
                    ),
                    control_progress_reason=(
                        final_control_progress_reason
                    ),
                )

                preserve_failure_progress_diagnostics = True
                break

            if bool(request.collision_check_enabled):
                if self.require_scan_for_collision_check and not self._is_scan_fresh():
                    result_code = ExecuteMotionPrimitive.Result.OBSTACLE_TOO_CLOSE
                    result_message = f'{name} stopped: LiDAR scan became stale'
                    break

                clearance, clearance_name, stop_distance = self._travel_clearance(direction)
                if not math.isfinite(clearance):
                    clearance = float('inf')

                if clearance < stop_distance:
                    if (
                        self.grid_cell_settle_enabled
                        and remaining <= self.grid_cell_settle_abort_position_error_m
                    ):
                        self.publish_zero_twist()
                        final_control_progress = target_distance
                        final_progress_source_used = (
                            f'{progress_selection.source}_travel_guard_snap'
                        )
                        final_control_progress_reason = (
                            f'{name} near target but {clearance_name}={clearance:.3f} m '
                            f'< {stop_distance:.3f} m; entering cell settle instead of '
                            'aborting so settle can move away from the wall'
                        )
                        control_progress = target_distance
                        remaining = 0.0
                        final_position_error = 0.0
                    else:
                        result_code = ExecuteMotionPrimitive.Result.OBSTACLE_TOO_CLOSE
                        result_message = (
                            f'{name} stopped: {clearance_name}={clearance:.3f} m '
                            f'< {stop_distance:.3f} m'
                        )
                        break

            if remaining <= position_tol:
                self._flush_stop(n=6, dt=0.05)
                time.sleep(self.settle_time_sec)

                final_alignment = self._grid_alignment_snapshot()
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
                        track_lidar_progress=True,
                    )
                    final_selection = self._select_translation_progress(
                        odom_progress_m=odom_progress,
                        diagnostics=final_diagnostics,
                    )
                    final_heading_error_signed = normalize_angle(
                        start.yaw - final_snapshot.pose.yaw
                    )
                    final_selection = self._maybe_use_geometry_aware_odom_progress(
                        progress_selection=final_selection,
                        grid_context=grid_context,
                        alignment=final_alignment,
                        odom_progress_m=odom_progress,
                        heading_error_rad=final_heading_error_signed,
                        direction=direction,
                        target_distance_m=target_distance,
                    )

                    if not final_selection.valid:
                        # Do not invent odom progress. If we were close enough
                        # to enter final validation, use the last trusted LiDAR
                        # progress and let grid-cell settle reacquire immediate
                        # axial, side-wall, yaw, or travel-guard geometry.
                        force_cell_settle_after_invalid_final_progress = True
                        final_selection = replace(
                            final_selection,
                            valid=True,
                            progress_m=float(final_control_progress),
                            source='last_trusted_lidar_final_settle_recovery',
                            reason=(
                                'final LiDAR progress invalid; entering settle '
                                'from last trusted LiDAR progress instead of '
                                'using odom as distance truth; '
                                f'last_trusted_lidar_progress={final_control_progress:.3f} m; '
                                f'odom_progress={odom_progress:.3f} m diagnostic_only; '
                                f'original_final_lidar_reason='
                                f'{final_diagnostics.lidar_progress_reason}'
                            ),
                        )

                    final_control_progress = max(0.0, final_selection.progress_m)
                    final_progress_source_used = final_selection.source
                    final_control_progress_reason = final_selection.reason

                    remaining = target_distance - final_control_progress
                    heading_error = final_heading_error_signed
                    final_position_error = abs(remaining)
                    final_heading_error = abs(heading_error)

                final_yaw_observation = self._heading_observation_from_sources(
                    self._manhattan_yaw_observation(),
                    final_alignment,
                )
                self._grid_live_last_yaw_observation = final_yaw_observation

                (
                    final_heading_validation_error,
                    final_heading_validation_source,
                ) = choose_heading_validation_error(
                    odom_heading_error_rad=final_heading_error,
                    grid_yaw_error_rad=final_yaw_observation.yaw_error_rad,
                    grid_yaw_valid=final_yaw_observation.valid,
                    grid_yaw_correction_used=grid_yaw_correction_ever_used,
                    grid_alignment_confidence=final_yaw_observation.confidence,
                    min_grid_confidence=self.grid_manhattan_yaw_min_confidence,
                    max_grid_yaw_abs_error_rad=self.grid_manhattan_yaw_max_abs_error_rad,
                    grid_alignment_source=final_yaw_observation.source,
                )

                if (
                    self.enforce_final_error
                    and not force_cell_settle_after_invalid_final_progress
                    and (
                        final_position_error > self.grid_cell_settle_abort_position_error_m
                        or final_heading_validation_error
                        > self.grid_cell_settle_abort_heading_error_rad
                    )
                ):
                    result_code = ExecuteMotionPrimitive.Result.FINAL_ERROR_TOO_LARGE
                    result_message = (
                        f'{name} final error beyond settle abort threshold: '
                        f'pos={final_position_error:.3f} m '
                        f'> {self.grid_cell_settle_abort_position_error_m:.3f} m or '
                        f'heading={final_heading_validation_error:.3f} rad '
                        f'> {self.grid_cell_settle_abort_heading_error_rad:.3f} rad '
                        f'({final_heading_validation_source}; '
                        f'odom_heading={final_heading_error:.3f} rad)'
                    )
                    break

                if self.grid_cell_settle_enabled:
                    reverse_already_good = bool(
                        direction < 0.0
                        and final_position_error <= position_tol
                        and final_heading_validation_error <= heading_tol
                    )

                    if reverse_already_good:
                        settle_result_message = (
                            'reverse settle skipped: reverse motion already inside tolerance; '
                            f'pos_error={final_position_error:.3f} m <= '
                            f'{position_tol:.3f} m; '
                            f'heading_error={final_heading_validation_error:.3f} rad '
                            f'<= {heading_tol:.3f} rad'
                        )
                    else:
                        settle_result = self._settle_once_after_transit(
                            reverse=direction < 0.0,
                            goal_handle=goal_handle,
                            context=grid_context,
                            start=start,
                            direction=direction,
                            commanded_distance=target_distance,
                            start_ranges=start_ranges,
                            initial_progress_m=final_control_progress,
                            initial_progress_source=final_progress_source_used,
                            initial_progress_reason=final_control_progress_reason,
                            collision_check_enabled=bool(request.collision_check_enabled),
                            limits=limits,
                        )

                        settle_result_message = settle_result.message

                        if settle_result.canceled:
                            self._reset_live_grid_controller(
                                'translation canceled during cell settle'
                            )
                            self._reset_temporal_lidar_progress_tracker()
                            return self._cancel_result(goal_handle)

                        final_position_error = settle_result.position_error_m
                        final_heading_error = settle_result.heading_error_rad
                        final_heading_validation_error = settle_result.heading_error_rad
                        final_heading_validation_source = settle_result.heading_source
                        final_control_progress = settle_result.progress_m
                        final_odom_progress = settle_result.odom_progress_m
                        final_progress_source_used = settle_result.progress_source
                        final_control_progress_reason = settle_result.progress_reason
                        grid_yaw_correction_ever_used = (
                            grid_yaw_correction_ever_used
                            or settle_result.yaw_correction_used
                        )

                        if not settle_result.success:
                            result_code = settle_result.result_code
                            result_message = (
                                f'{name} cell settle failed: {settle_result.message}'
                            )
                            break

                if (
                    self.enforce_final_error
                    and (
                        final_position_error > position_tol * 1.5
                        or final_heading_validation_error > heading_tol * 1.5
                    )
                ):
                    result_success = False
                    result_code = ExecuteMotionPrimitive.Result.FINAL_ERROR_TOO_LARGE
                    result_message = (
                        f'{name} final error still high after settle: '
                        f'pos={final_position_error:.3f} m '
                        f'> allowed={position_tol * 1.5:.3f} m or '
                        f'heading={final_heading_validation_error:.3f} rad '
                        f'> allowed={heading_tol * 1.5:.3f} rad '
                        f'({final_heading_validation_source}; '
                        f'odom_heading={final_heading_error:.3f} rad)'
                    )
                    break

                result_success = True
                result_code = ExecuteMotionPrimitive.Result.SUCCESS
                result_message = (
                    f'{name} succeeded: '
                    f'pos_error={final_position_error:.3f} m, '
                    f'heading_error={final_heading_validation_error:.3f} rad '
                    f'({final_heading_validation_source}; '
                    f'odom_heading={final_heading_error:.3f} rad), '
                    f'progress={final_control_progress:.3f} m '
                    f'from {final_progress_source_used}; '
                    f'settle_result={settle_result_message}'
                )
                break

            speed_mag = clamp(
                self.k_distance * max(remaining, 0.0),
                self.min_linear_x_mps,
                max_speed,
            )
            raw_heading_correction = self.k_heading * heading_error

            live_observation = self._live_grid_observation(
                grid_context,
                current_alignment,
                control_progress,
            )
            live_command = self._live_grid_command(
                live_observation,
                odom_heading_correction_radps=raw_heading_correction,
            )

            if live_command.yaw_mode == YAW_RECOVERY:
                # Yaw recovery is not a fatal condition. Slow down and let the
                # live controller recover from LiDAR/Manhattan lines.
                speed_mag *= max(0.20, float(live_command.speed_scale))

            speed_mag *= live_command.speed_scale

            cmd = Twist()
            cmd.linear.x = direction * speed_mag
            cmd.linear.y = live_command.linear_y_mps if live_command.lateral_active else 0.0
            cmd.angular.z = live_command.angular_z_radps

            cmd = self._limiter.clamp(cmd, limits)
            cmd = self._apply_acceleration_limits(cmd)
            self._publish_drive_cmd(
                forward=cmd.linear.x,
                lateral=cmd.linear.y,
                yaw_rate=cmd.angular.z,
                allow_lateral=True,
            )
            self._set_grid_yaw_control_status(
                live_command.yaw_active,
                live_command.angular_z_radps if live_command.yaw_active else 0.0,
                live_command.reason,
            )
            grid_yaw_correction_ever_used = (
                grid_yaw_correction_ever_used or live_command.yaw_active
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
                    f'grid_yaw_mode={live_command.yaw_mode}, '
                    f'grid_lateral_mode={live_command.lateral_mode}, '
                    f'grid_context_valid={live_observation.context_valid}, '
                    f'virtual_cell={live_observation.virtual_cell.cell_idx}, '
                    f'cell_progress={live_observation.virtual_cell.distance_into_cell_m:.3f} m, '
                    f'boundary_zone={live_observation.virtual_cell.boundary_zone}, '
                    f'expected_walls=F{int(live_observation.expected.front)}'
                    f'B{int(live_observation.expected.rear)}'
                    f'L{int(live_observation.expected.left)}'
                    f'R{int(live_observation.expected.right)}, '
                    f'yaw_valid={live_observation.yaw.valid}, '
                    f'yaw_source={live_observation.yaw.source}, '
                    f'yaw_error={live_observation.yaw.yaw_error_rad:.3f} rad, '
                    f'yaw_conf={live_observation.yaw.confidence:.2f}, '
                    f'yaw_lines={live_observation.yaw.line_count}, '
                    f'yaw_conc={live_observation.yaw.concentration:.2f}, '
                    f'lat_valid={live_observation.centering.valid}, '
                    f'lat_source={live_observation.centering.source}, '
                    f'lat_error={live_observation.centering.lateral_error_m:.3f} m, '
                    f'lat_conf={live_observation.centering.confidence:.2f}, '
                    f'linear_y={cmd.linear.y:.3f} m/s, '
                    f'angular_z={cmd.angular.z:.3f} rad/s, '
                    f'live_reason={live_command.reason}'
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
        if not preserve_failure_progress_diagnostics:
            translation_diagnostics = self._translation_diagnostics(
                direction=direction,
                odom_progress_m=final_odom_progress,
                start_ranges=start_ranges,
                end_ranges=end_ranges,
                track_lidar_progress=True,
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
        last_cmd = self._grid_live_last_command

        translation_diagnostics = replace(
            translation_diagnostics,
            grid_yaw_correction_used=grid_yaw_correction_ever_used,
            final_grid_yaw_correction_radps=(
                last_cmd.angular_z_radps if last_cmd.yaw_active else 0.0
            ),
            grid_yaw_control_reason=last_cmd.reason,
            live_grid_mode=last_cmd.mode,
            live_grid_context_valid=self._grid_live_last_context_valid,
            live_grid_virtual_cell=self._grid_live_last_virtual_cell,
            live_grid_boundary_zone=self._grid_live_last_boundary_zone,
            live_grid_expected_front=self._grid_live_last_expected_front,
            live_grid_expected_rear=self._grid_live_last_expected_rear,
            live_grid_expected_left=self._grid_live_last_expected_left,
            live_grid_expected_right=self._grid_live_last_expected_right,
            live_grid_yaw_active=last_cmd.yaw_active,
            live_grid_lateral_active=last_cmd.lateral_active,
            live_grid_yaw_cmd_radps=last_cmd.angular_z_radps,
            live_grid_lateral_cmd_mps=last_cmd.linear_y_mps,
            live_grid_reason=last_cmd.reason,
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
            f'lidar_valid={translation_diagnostics.lidar_progress_valid}, '
            f'lidar_progress={translation_diagnostics.lidar_progress_m:.3f} m, '
            f'control_progress={translation_diagnostics.final_control_progress_m:.3f} m, '
            f'progress_source={translation_diagnostics.progress_source_used}, '
            f'live_grid_mode={translation_diagnostics.live_grid_mode}, '
            f'live_grid_lateral_active={translation_diagnostics.live_grid_lateral_active}'
        )

        self._reset_live_grid_controller('translation finished')
        self._reset_temporal_lidar_progress_tracker()

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

    def _pre_rotate_clearance_settle(self) -> str:
        if not self.rotate_pre_clearance_settle_enabled:
            return 'pre-rotate clearance settle disabled'

        start_time = time.monotonic()
        last_reason = 'not evaluated'

        while time.monotonic() - start_time <= self.rotate_pre_clearance_timeout_sec:
            ranges = self._cardinal_range_snapshot()
            if ranges is None:
                last_reason = 'no fresh cardinal LiDAR snapshot'
                break

            front_distance = self._range_value(ranges.front)
            rear_distance = self._range_value(ranges.rear)

            front_too_close = bool(
                ranges.front.valid
                and math.isfinite(front_distance)
                and front_distance > 0.0
                and front_distance < self.rotate_pre_clearance_front_min_m
            )
            rear_too_close = bool(
                ranges.rear.valid
                and math.isfinite(rear_distance)
                and rear_distance > 0.0
                and rear_distance < self.rotate_pre_clearance_rear_min_m
            )

            if not front_too_close and not rear_too_close:
                self.publish_zero_twist()
                return (
                    'pre-rotate clearance ok: '
                    f'front_valid={ranges.front.valid}; front={front_distance:.3f} m; '
                    f'rear_valid={ranges.rear.valid}; rear={rear_distance:.3f} m'
                )

            cmd = Twist()

            if front_too_close and not rear_too_close:
                error = self.rotate_pre_clearance_target_front_m - front_distance
                cmd.linear.x = -min(
                    self.rotate_pre_clearance_max_linear_x_mps,
                    max(self.min_linear_x_mps, abs(error) * self.k_distance),
                )
                last_reason = (
                    'front too close before rotate; backing up inside same cell: '
                    f'front={front_distance:.3f} m < '
                    f'{self.rotate_pre_clearance_front_min_m:.3f} m; '
                    f'cmd={cmd.linear.x:.3f} m/s'
                )
            elif rear_too_close and not front_too_close:
                error = self.rotate_pre_clearance_target_rear_m - rear_distance
                cmd.linear.x = min(
                    self.rotate_pre_clearance_max_linear_x_mps,
                    max(self.min_linear_x_mps, abs(error) * self.k_distance),
                )
                last_reason = (
                    'rear too close before rotate; moving forward inside same cell: '
                    f'rear={rear_distance:.3f} m < '
                    f'{self.rotate_pre_clearance_rear_min_m:.3f} m; '
                    f'cmd={cmd.linear.x:.3f} m/s'
                )
            else:
                self.publish_zero_twist()
                return (
                    'pre-rotate clearance blocked: both front and rear too close; '
                    f'front={front_distance:.3f} m; rear={rear_distance:.3f} m'
                )

            self._publish_drive_cmd(
                forward=cmd.linear.x,
                lateral=0.0,
                yaw_rate=0.0,
                allow_lateral=False,
            )
            time.sleep(1.0 / self.control_rate_hz)

        self.publish_zero_twist()
        return 'pre-rotate clearance settle timeout or unavailable: ' + last_reason

    def _post_rotate_manhattan_snap(
        self,
        goal_handle,
        limits: VelocityLimits,
    ) -> PostRotateManhattanSnapResult:
        """Apply a small, yaw-only Manhattan correction after odom rotation.

        This phase is deliberately independent of map context. It never commands
        translation and it is nonfatal when Manhattan geometry is unavailable.
        """
        if not self.rotate_post_manhattan_snap_enabled:
            return PostRotateManhattanSnapResult(
                canceled=False,
                correction_applied=False,
                converged=False,
                measured_abs_error_rad=None,
                reason='post-rotate Manhattan snap disabled',
            )

        if not self.grid_manhattan_yaw_enabled:
            return PostRotateManhattanSnapResult(
                canceled=False,
                correction_applied=False,
                converged=False,
                measured_abs_error_rad=None,
                reason='post-rotate Manhattan snap skipped: Manhattan yaw disabled',
            )

        max_yaw_rate = min(
            abs(float(limits.max_angular_z_radps)),
            float(self.rotate_post_manhattan_snap_max_yaw_radps),
        )
        if max_yaw_rate <= 0.0:
            return PostRotateManhattanSnapResult(
                canceled=False,
                correction_applied=False,
                converged=False,
                measured_abs_error_rad=None,
                reason='post-rotate Manhattan snap skipped: zero angular speed limit',
            )

        self.publish_zero_twist()

        # Discard the scan that existed when the snap phase began. The next
        # observation must come from a distinct scan received after stopping.
        _, last_scan_token = self._fresh_scan_copy_with_token()

        start_time = time.monotonic()
        acquire_deadline = (
            start_time + self.rotate_post_manhattan_snap_acquire_timeout_sec
        )
        deadline = start_time + self.rotate_post_manhattan_snap_timeout_sec

        previous_error: Optional[float] = None
        consistent_samples = 0
        stable_samples = 0
        correction_applied = False
        last_valid_error: Optional[float] = None
        last_reason = 'waiting for a distinct fresh LiDAR scan'

        while time.monotonic() <= deadline:
            if self._cancel_or_stop_requested(goal_handle):
                self.publish_zero_twist()
                return PostRotateManhattanSnapResult(
                    canceled=True,
                    correction_applied=correction_applied,
                    converged=False,
                    measured_abs_error_rad=(
                        abs(last_valid_error)
                        if last_valid_error is not None
                        else None
                    ),
                    reason='post-rotate Manhattan snap canceled',
                )

            scan, scan_token = self._fresh_scan_copy_with_token()
            now = time.monotonic()

            if scan is None or scan_token is None:
                self.publish_zero_twist()
                last_reason = 'no fresh LiDAR scan'

                if not correction_applied and now >= acquire_deadline:
                    break

                time.sleep(1.0 / self.control_rate_hz)
                continue

            if scan_token == last_scan_token:
                # Do not allow the last angular command to remain active
                # indefinitely while waiting for another scan.
                if (
                    correction_applied
                    and now - scan_token
                    > self.rotate_post_manhattan_snap_max_scan_hold_sec
                ):
                    self.publish_zero_twist()
                    last_reason = (
                        'waiting for a new LiDAR scan; angular command held for '
                        'too long and was stopped'
                    )

                if (
                    not correction_applied
                    and now >= acquire_deadline
                    and consistent_samples
                    < self.rotate_post_manhattan_snap_confirm_samples
                ):
                    last_reason = (
                        'not enough distinct consistent post-stop LiDAR scans '
                        'before acquisition timeout'
                    )
                    break

                time.sleep(1.0 / self.control_rate_hz)
                continue

            last_scan_token = scan_token

            observation = self._manhattan_yaw_observation_from_scan(
                scan,
                max_abs_yaw_error_rad=(
                    self.rotate_post_manhattan_snap_max_abs_error_rad
                ),
            )

            observation_valid = bool(
                observation.valid
                and observation.confidence
                >= self.rotate_post_manhattan_snap_min_confidence
            )

            if not observation_valid:
                self.publish_zero_twist()

                # Invalid geometry breaks the consecutive-observation chain.
                previous_error = None
                consistent_samples = 0
                stable_samples = 0

                last_reason = (
                    'invalid Manhattan observation: '
                    f'confidence={observation.confidence:.2f}; '
                    f'{observation.reason}'
                )

                if not correction_applied and now >= acquire_deadline:
                    break

                time.sleep(1.0 / self.control_rate_hz)
                continue

            error = float(observation.yaw_error_rad)
            last_valid_error = error

            if previous_error is None:
                consistent_samples = 1
            else:
                # Sign changes are allowed inside the final tolerance because
                # small measurement noise can move the estimate across zero.
                same_direction = bool(
                    error * previous_error >= 0.0
                    or abs(error)
                    <= self.rotate_post_manhattan_snap_tolerance_rad
                    or abs(previous_error)
                    <= self.rotate_post_manhattan_snap_tolerance_rad
                )
                close_enough = bool(
                    abs(error - previous_error)
                    <= self.rotate_post_manhattan_snap_max_sample_delta_rad
                )

                consistent_samples = (
                    consistent_samples + 1
                    if same_direction and close_enough
                    else 1
                )

            previous_error = error

            if abs(error) <= self.rotate_post_manhattan_snap_tolerance_rad:
                # Stop while confirming that the final heading remains stable.
                self.publish_zero_twist()
                stable_samples = (
                    stable_samples + 1 if consistent_samples > 1 else 1
                )
                angular_z = 0.0
            else:
                stable_samples = 0

                if (
                    consistent_samples
                    >= self.rotate_post_manhattan_snap_confirm_samples
                ):
                    angular_z = clamp(
                        self.k_grid_live_yaw * error,
                        -max_yaw_rate,
                        max_yaw_rate,
                    )
                    self._publish_rotate_only(angular_z)
                    correction_applied = True
                else:
                    # Never rotate from only one Manhattan observation.
                    angular_z = 0.0
                    self.publish_zero_twist()

            last_reason = (
                f'error={error:.3f} rad; '
                f'confidence={observation.confidence:.2f}; '
                f'consistent={consistent_samples}/'
                f'{self.rotate_post_manhattan_snap_confirm_samples}; '
                f'stable={stable_samples}/'
                f'{self.rotate_post_manhattan_snap_stable_samples}; '
                f'cmd_yaw={angular_z:.3f} rad/s; '
                f'{observation.reason}'
            )

            self._update_motion_state(
                distance_remaining=0.0,
                distance_traveled=0.0,
                heading_error=error,
                status='ROTATE_RELATIVE Manhattan snap: ' + last_reason,
            )
            self._publish_feedback(
                goal_handle,
                progress=1.0,
                distance_remaining=0.0,
                heading_remaining=error,
                state='ROTATE_RELATIVE_MANHATTAN_SNAP',
            )

            if stable_samples >= self.rotate_post_manhattan_snap_stable_samples:
                self.publish_zero_twist()
                return PostRotateManhattanSnapResult(
                    canceled=False,
                    correction_applied=correction_applied,
                    converged=True,
                    measured_abs_error_rad=abs(error),
                    reason='post-rotate Manhattan snap converged: ' + last_reason,
                )

            if (
                not correction_applied
                and now >= acquire_deadline
                and consistent_samples
                < self.rotate_post_manhattan_snap_confirm_samples
            ):
                last_reason = (
                    'post-rotate Manhattan snap skipped: observations did not '
                    'become consistent before acquisition timeout; '
                    + last_reason
                )
                break

            time.sleep(1.0 / self.control_rate_hz)

        self.publish_zero_twist()

        measured_error = (
            abs(last_valid_error)
            if last_valid_error is not None
            else None
        )
        outcome = 'timed out' if correction_applied else 'skipped'

        return PostRotateManhattanSnapResult(
            canceled=False,
            correction_applied=correction_applied,
            converged=False,
            measured_abs_error_rad=measured_error,
            reason=f'post-rotate Manhattan snap {outcome}: {last_reason}',
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

        self._publish_zero_for_duration()
        pre_rotate_clearance_reason = (
            'pre-rotate clearance disabled; ROTATE_RELATIVE is rotation-only'
        )
        self._flush_stop(n=5, dt=0.05)
        time.sleep(0.20)
        self._reset_live_grid_controller('rotation active')
        self._set_grid_yaw_control_status(False, 0.0, 'rotation active')

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
            f'source={start_alignment.source}; '
            f'pre_rotate_clearance={pre_rotate_clearance_reason}'
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
        soft_timeout_warned = False
        hard_timeout_s = timeout_s + float(self.rotate_timeout_recovery_extra_sec)
        rotation_stable_count = 0
        rotation_stable_tolerance = min(float(heading_tol), 0.025)

        while True:
            if self._cancel_or_stop_requested(goal_handle):
                return self._cancel_result(goal_handle)

            elapsed = time.monotonic() - start_time

            if elapsed > timeout_s and not soft_timeout_warned:
                soft_timeout_warned = True
                self.get_logger().warning(
                    'ROTATE_RELATIVE exceeded soft timeout; continuing recovery: '
                    f'elapsed={elapsed:.1f}s; soft_timeout={timeout_s:.1f}s; '
                    f'hard_timeout={hard_timeout_s:.1f}s; '
                    f'current_heading_error={final_heading_error:.3f} rad'
                )

            if elapsed > hard_timeout_s:
                timeout_heading_error_valid = False
                snapshot = self._get_motion_snapshot()
                if snapshot is not None:
                    remaining = normalize_angle(target_yaw - snapshot.pose.yaw)
                    final_heading_error = abs(remaining)
                    timeout_heading_error_valid = True

                accept_near_target_timeout = (
                    timeout_heading_error_valid
                    and rotation_timeout_accepts_heading_error(
                        final_heading_error,
                        self.rotate_timeout_accept_heading_error_rad,
                    )
                )
                if accept_near_target_timeout:
                    self._flush_stop(n=8, dt=0.05)
                    time.sleep(0.30)
                    result_success = True
                    result_code = ExecuteMotionPrimitive.Result.SUCCESS
                    result_message = (
                        'ROTATE_RELATIVE accepted near target after recovery timeout: '
                        f'odom_heading_error={final_heading_error:.3f} rad'
                    )
                else:
                    result_code = ExecuteMotionPrimitive.Result.TIMEOUT
                    result_message = (
                        f'ROTATE_RELATIVE hard timeout after {elapsed:.1f}s; '
                        f'heading_error={final_heading_error:.3f} rad'
                    )
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
            if final_heading_error < rotation_stable_tolerance:
                rotation_stable_count += 1
            else:
                rotation_stable_count = 0

            rotated = abs(normalize_angle(current.yaw - start.yaw))
            target_abs = max(abs(target_angle), 1e-6)

            if rotation_stable_count >= 5:
                self._flush_stop(n=8, dt=0.05)
                time.sleep(0.30)

                final_snapshot = self._get_motion_snapshot()
                if final_snapshot is not None:
                    remaining = normalize_angle(target_yaw - final_snapshot.pose.yaw)
                    final_heading_error = abs(remaining)

                if self.enforce_final_error and final_heading_error > heading_tol * 1.5:
                    if time.monotonic() - start_time < hard_timeout_s:
                        self.get_logger().warn(
                            'ROTATE_RELATIVE post-stop heading drift exceeded final '
                            f'check: {final_heading_error:.3f} rad > '
                            f'{heading_tol * 1.5:.3f} rad; correcting again'
                        )
                        continue

                    if rotation_timeout_accepts_heading_error(
                        final_heading_error,
                        self.rotate_timeout_accept_heading_error_rad,
                    ):
                        result_success = True
                        result_code = ExecuteMotionPrimitive.Result.SUCCESS
                        result_message = (
                            'ROTATE_RELATIVE accepted near target after post-stop drift: '
                            f'odom_heading_error={final_heading_error:.3f} rad'
                        )
                        break

                    result_code = ExecuteMotionPrimitive.Result.FINAL_ERROR_TOO_LARGE
                    result_message = (
                        f'ROTATE_RELATIVE final heading error too large: '
                        f'{final_heading_error:.3f} rad'
                    )
                    break

                result_success = True
                result_code = ExecuteMotionPrimitive.Result.SUCCESS
                result_message = (
                    'ROTATE_RELATIVE odom phase succeeded: '
                    f'odom_heading_error={final_heading_error:.3f} rad'
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
            self._publish_rotate_only(cmd.angular.z)

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

        if result_success:
            odom_heading_error_before_snap = final_heading_error

            snap_result = self._post_rotate_manhattan_snap(
                goal_handle,
                limits,
            )

            if snap_result.canceled:
                return self._cancel_result(goal_handle)

            # A skipped, unconfirmed observation must not replace the accepted
            # odometry result. Use Manhattan error only when it converged or when
            # it actually commanded a correction.
            if (
                snap_result.measured_abs_error_rad is not None
                and (
                    snap_result.converged
                    or snap_result.correction_applied
                )
            ):
                final_heading_error = snap_result.measured_abs_error_rad

            result_message = (
                f'{result_message}; '
                f'odom_heading_error_before_snap='
                f'{odom_heading_error_before_snap:.3f} rad; '
                f'{snap_result.reason}'
            )

        final_alignment = self._grid_alignment_snapshot()
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
        return self._min_range_in_sector(scan, 0.0, self.collision_sector_deg)

    def _rear_clearance(self) -> float:
        scan = self._fresh_scan_copy()
        if scan is None:
            return float('inf')
        return self._min_range_in_sector(scan, math.pi, self.collision_sector_deg)

    def _travel_clearance(self, direction: float) -> tuple[float, str, float]:
        if direction >= 0.0:
            return (
                self._front_clearance(),
                'front_clearance',
                float(self.front_stop_distance_m),
            )

        return (
            self._rear_clearance(),
            'rear_clearance',
            float(self.rear_stop_distance_m),
        )

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
            adjacent_wall_tolerance_m=self.grid_alignment_adjacent_wall_tolerance_m,
            pair_width_tolerance_m=self.grid_alignment_pair_width_tolerance_m,
        )

    def _grid_alignment_snapshot(self) -> GridAlignmentEstimate:
        scan = self._fresh_scan_copy()
        return self._grid_alignment_from_scan(scan)

    def _manhattan_yaw_observation_from_scan(
        self,
        scan: Optional[LaserScan],
        *,
        max_abs_yaw_error_rad: float,
    ) -> GridYawObservation:
        if not self.grid_manhattan_yaw_enabled:
            return invalid_grid_yaw('grid Manhattan yaw disabled')

        if scan is None:
            return invalid_grid_yaw('no fresh scan for Manhattan yaw')

        return estimate_manhattan_grid_yaw(
            scan,
            min_range_m=self.grid_manhattan_yaw_min_range_m,
            max_range_m=self.grid_manhattan_yaw_max_range_m,
            max_point_gap_m=self.grid_manhattan_yaw_max_point_gap_m,
            max_range_jump_m=self.grid_manhattan_yaw_max_range_jump_m,
            min_cluster_points=self.grid_manhattan_yaw_min_cluster_points,
            min_segment_points=self.grid_manhattan_yaw_min_segment_points,
            min_segment_length_m=self.grid_manhattan_yaw_min_segment_length_m,
            max_line_rms_m=self.grid_manhattan_yaw_max_line_rms_m,
            min_line_count=self.grid_manhattan_yaw_min_line_count,
            min_total_weight=self.grid_manhattan_yaw_min_total_weight,
            min_concentration=self.grid_manhattan_yaw_min_concentration,
            max_abs_yaw_error_rad=max_abs_yaw_error_rad,
        )

    def _manhattan_yaw_observation(self) -> GridYawObservation:
        return self._manhattan_yaw_observation_from_scan(
            self._fresh_scan_copy(),
            max_abs_yaw_error_rad=self.grid_manhattan_yaw_max_abs_error_rad,
        )

    def _choose_heading_error(
        self,
        *,
        manhattan_valid: bool,
        manhattan_yaw: float,
        manhattan_conf: float,
        side_wall_valid: bool,
        side_wall_yaw: float,
    ) -> tuple[bool, float, str]:
        """
        Prefer high-confidence Manhattan yaw.
        Reject side-wall yaw when it disagrees with Manhattan yaw.
        """
        MANHATTAN_CONF_MIN = 0.80
        YAW_DISAGREE_REJECT_RAD = 0.05

        if manhattan_valid and manhattan_conf >= MANHATTAN_CONF_MIN:
            if (
                side_wall_valid
                and abs(float(side_wall_yaw) - float(manhattan_yaw))
                > YAW_DISAGREE_REJECT_RAD
            ):
                return True, float(manhattan_yaw), (
                    'manhattan_yaw accepted; side_wall_yaw rejected: '
                    f'side={float(side_wall_yaw):.3f}, '
                    f'manhattan={float(manhattan_yaw):.3f}'
                )

            return True, float(manhattan_yaw), 'manhattan_yaw accepted'

        if side_wall_valid:
            return (
                True,
                float(side_wall_yaw),
                'side_wall_yaw accepted; manhattan unavailable/weak',
            )

        if manhattan_valid:
            return True, float(manhattan_yaw), 'weak manhattan_yaw accepted; no side wall'

        return False, 0.0, 'no valid yaw source'

    def _heading_observation_from_sources(
        self,
        manhattan: GridYawObservation,
        alignment: GridAlignmentEstimate,
    ) -> GridYawObservation:
        side_wall_valid = bool(
            alignment.yaw_valid
            and abs(float(alignment.yaw_error_rad))
            <= float(self.grid_manhattan_yaw_max_abs_error_rad)
        )

        heading_valid, heading_error, heading_reason = self._choose_heading_error(
            manhattan_valid=bool(manhattan.valid),
            manhattan_yaw=float(manhattan.yaw_error_rad),
            manhattan_conf=float(manhattan.confidence),
            side_wall_valid=side_wall_valid,
            side_wall_yaw=float(alignment.yaw_error_rad),
        )

        if not heading_valid:
            return invalid_grid_yaw(heading_reason)

        if 'side_wall_yaw accepted' in heading_reason:
            line_count = 0
            if alignment.left.valid:
                line_count += 1
            if alignment.right.valid:
                line_count += 1
            return GridYawObservation(
                valid=True,
                yaw_error_rad=float(heading_error),
                confidence=float(alignment.confidence),
                source=f'side_wall/{alignment.source}',
                line_count=line_count,
                dominant_axis_rad=0.0,
                total_weight=float(line_count),
                concentration=float(alignment.confidence),
                reason=(
                    f'{heading_reason}; side_wall_source={alignment.source}; '
                    f'side_wall_reason={alignment.reason}; '
                    f'manhattan_reason={manhattan.reason}'
                ),
            )

        return replace(
            manhattan,
            valid=True,
            yaw_error_rad=float(heading_error),
            reason=f'{heading_reason}; {manhattan.reason}',
        )

    def _reset_live_grid_controller(self, reason: str = '') -> None:
        self._grid_live_mode = UNAVAILABLE
        self._grid_yaw_mode = UNAVAILABLE
        self._grid_lateral_mode = UNAVAILABLE
        self._grid_live_reacquire_samples = 0
        self._grid_live_last_observation = None
        self._grid_live_last_yaw_observation = invalid_grid_yaw(reason or 'reset')
        self._grid_live_last_centering_observation = invalid_centering(reason or 'reset')
        self._grid_live_last_command = LiveGridCommand(
            mode=UNAVAILABLE,
            yaw_mode=UNAVAILABLE,
            lateral_mode=UNAVAILABLE,
            yaw_active=False,
            lateral_active=False,
            angular_z_radps=0.0,
            linear_y_mps=0.0,
            speed_scale=1.0,
            reason=reason or 'reset',
        )
        self._grid_live_last_context_valid = False
        self._grid_live_last_virtual_cell = 0
        self._grid_live_last_virtual_cell_progress_m = 0.0
        self._grid_live_last_boundary_zone = False
        self._grid_live_last_expected_front = False
        self._grid_live_last_expected_rear = False
        self._grid_live_last_expected_left = False
        self._grid_live_last_expected_right = False
        self._grid_live_last_reason = reason or 'reset'

    def _live_grid_observation(
        self,
        context: GridRunContext,
        alignment: GridAlignmentEstimate,
        progress_m: float,
    ) -> GridObservation:
        yaw = self._heading_observation_from_sources(
            self._manhattan_yaw_observation(),
            alignment,
        )

        virtual, expected, centering = observe_centering_from_expected_side_walls(
            context=context,
            alignment=alignment,
            progress_m=progress_m,
            cell_length_m=self.cell_length_m,
            boundary_margin_m=self.grid_virtual_cell_boundary_margin_m,
            expected_half_width_m=self.grid_alignment_expected_half_width_m,
            adjacent_wall_tolerance_m=self.grid_alignment_adjacent_wall_tolerance_m,
            pair_width_tolerance_m=self.grid_alignment_pair_width_tolerance_m,
            max_abs_yaw_error_rad=self.grid_live_max_abs_yaw_error_rad,
            max_rms_error_m=self.grid_live_max_rms_error_m,
            min_span_x_m=self.grid_live_min_span_x_m,
            min_support_count=self.grid_live_min_support_count,
            min_confidence=self.grid_lateral_min_confidence,
        )

        if not context.valid:
            centering = self._mapless_centering_from_alignment(alignment)

        return make_grid_observation(
            context=context,
            yaw=yaw,
            centering=centering,
            virtual=virtual,
            expected=expected,
        )

    def _live_grid_command(
        self,
        observation: GridObservation,
        odom_heading_correction_radps: float,
    ) -> LiveGridCommand:
        if not self.grid_live_control_enabled:
            return LiveGridCommand(
                mode=HEADING_COAST,
                yaw_mode=HEADING_COAST,
                lateral_mode=LATERAL_COAST,
                yaw_active=False,
                lateral_active=False,
                angular_z_radps=float(odom_heading_correction_radps),
                linear_y_mps=0.0,
                speed_scale=1.0,
                reason='grid live control disabled',
            )

        if self._grid_yaw_mode == HEADING_COAST and observation.yaw.valid:
            self._grid_live_reacquire_samples += 1
        elif observation.yaw.valid:
            self._grid_live_reacquire_samples = self.grid_reacquire_stable_samples
        else:
            self._grid_live_reacquire_samples = 0

        command = live_grid_command(
            observation=observation,
            previous_yaw_mode=self._grid_yaw_mode,
            odom_heading_correction_radps=odom_heading_correction_radps,
            k_yaw=self.k_grid_live_yaw,
            max_yaw_correction_radps=self.max_grid_live_yaw_correction_radps,
            k_lateral=self.k_grid_lateral,
            max_lateral_mps=min(self.max_grid_lateral_mps, self.max_linear_y_mps),
            yaw_min_confidence=self.grid_manhattan_yaw_min_confidence,
            lateral_min_confidence=self.grid_lateral_min_confidence,
            reacquire_stable_samples=self.grid_reacquire_stable_samples,
            current_reacquire_samples=self._grid_live_reacquire_samples,
            small_reacquire_yaw_rad=self.grid_reacquire_small_yaw_rad,
            large_reacquire_yaw_rad=self.grid_reacquire_large_yaw_rad,
            reacquire_speed_scale=self.grid_reacquire_speed_scale,
            heading_coast_max_odom_yaw_radps=(
                self.grid_heading_coast_max_odom_yaw_radps
            ),
            heading_coast_speed_scale=self.grid_heading_coast_speed_scale,
            heading_coast_stop_odom_yaw_radps=(
                self.grid_heading_coast_stop_odom_yaw_radps
            ),
        )

        self._grid_live_mode = command.mode
        self._grid_yaw_mode = command.yaw_mode
        self._grid_lateral_mode = command.lateral_mode
        self._grid_live_last_command = command
        self._grid_live_last_observation = observation
        self._grid_live_last_yaw_observation = observation.yaw
        self._grid_live_last_centering_observation = observation.centering
        self._grid_live_last_context_valid = observation.context_valid
        self._grid_live_last_virtual_cell = observation.virtual_cell.cell_idx
        self._grid_live_last_virtual_cell_progress_m = (
            observation.virtual_cell.distance_into_cell_m
        )
        self._grid_live_last_boundary_zone = observation.virtual_cell.boundary_zone
        self._grid_live_last_expected_front = observation.expected.front
        self._grid_live_last_expected_rear = observation.expected.rear
        self._grid_live_last_expected_left = observation.expected.left
        self._grid_live_last_expected_right = observation.expected.right
        self._grid_live_last_reason = command.reason

        return command

    def _mapless_centering_from_alignment(
        self,
        alignment: GridAlignmentEstimate,
    ) -> GridCenteringObservation:
        if not alignment.valid or not alignment.lateral_valid:
            return invalid_centering(
                f'no observed adjacent side-wall centering: {alignment.reason}'
            )

        if alignment.confidence < self.grid_lateral_min_confidence:
            return invalid_centering(
                f'observed side-wall confidence {alignment.confidence:.2f} '
                f'< {self.grid_lateral_min_confidence:.2f}: {alignment.reason}'
            )

        if abs(alignment.yaw_error_rad) > self.grid_live_max_abs_yaw_error_rad:
            return invalid_centering(
                f'observed side-wall yaw {alignment.yaw_error_rad:.3f} rad '
                f'> {self.grid_live_max_abs_yaw_error_rad:.3f} rad: '
                f'{alignment.reason}'
            )

        return GridCenteringObservation(
            valid=True,
            lateral_error_m=float(alignment.lateral_error_m),
            confidence=float(alignment.confidence),
            source=f'observed_{alignment.source}',
            matched_expected_wall_ids=(),
            left_usable=bool(alignment.left.valid),
            right_usable=bool(alignment.right.valid),
            reason=(
                'mapless observed side-wall centering; '
                f'source={alignment.source}; {alignment.reason}'
            ),
        )

    def _mapless_axial_cell_centering_estimate(
        self,
        ranges: Optional[LidarRangeSnapshot],
    ) -> AxialCellCenterEstimate:
        if ranges is None:
            return AxialCellCenterEstimate(
                valid=False,
                error_m=0.0,
                source='none',
                front_usable=False,
                rear_usable=False,
                disagreement_m=0.0,
                reason='no fresh cardinal LiDAR snapshot',
            )

        front_distance = self._range_value(ranges.front)
        rear_distance = self._range_value(ranges.rear)
        front_error = abs(front_distance - self.grid_center_expected_front_distance_m)
        rear_error = abs(rear_distance - self.grid_center_expected_rear_distance_m)

        front_local = bool(
            ranges.front.valid
            and math.isfinite(front_distance)
            and front_error <= self.grid_cell_settle_abort_position_error_m
        )
        rear_local = bool(
            ranges.rear.valid
            and math.isfinite(rear_distance)
            and rear_error <= self.grid_cell_settle_abort_position_error_m
        )

        estimate = choose_axial_cell_centering(
            expected_front=front_local,
            front_valid=bool(ranges.front.valid),
            front_distance_m=front_distance,
            expected_front_distance_m=self.grid_center_expected_front_distance_m,
            expected_rear=rear_local,
            rear_valid=bool(ranges.rear.valid),
            rear_distance_m=rear_distance,
            expected_rear_distance_m=self.grid_center_expected_rear_distance_m,
            max_disagreement_m=self.grid_center_front_rear_agreement_tolerance_m,
        )

        if estimate.valid:
            return replace(
                estimate,
                reason=(
                    'mapless local front/rear cell-center reference; '
                    f'{estimate.reason}'
                ),
            )

        if estimate.source == 'front_rear_rejected':
            return AxialCellCenterEstimate(
                valid=False,
                error_m=0.0,
                source='none',
                front_usable=False,
                rear_usable=False,
                disagreement_m=estimate.disagreement_m,
                reason=(
                    'ignoring mapless front/rear local references because they disagree; '
                    f'{estimate.reason}'
                ),
            )

        return replace(
            estimate,
            reason=(
                'no mapless local front/rear cell-center reference: '
                f'front_valid={ranges.front.valid}; front_distance={front_distance:.3f} m; '
                f'front_error={front_error:.3f} m; '
                f'rear_valid={ranges.rear.valid}; rear_distance={rear_distance:.3f} m; '
                f'rear_error={rear_error:.3f} m'
            ),
        )

    def _safety_axial_cell_centering_estimate(
        self,
        ranges: Optional[LidarRangeSnapshot],
    ) -> AxialCellCenterEstimate:
        if ranges is None:
            return AxialCellCenterEstimate(
                valid=False,
                error_m=0.0,
                source='none',
                front_usable=False,
                rear_usable=False,
                disagreement_m=0.0,
                reason='no fresh cardinal LiDAR snapshot for safety axial settle',
            )

        front_distance = self._range_value(ranges.front)
        rear_distance = self._range_value(ranges.rear)
        tolerance = float(self.grid_cell_settle_position_tolerance_m)

        candidates: list[AxialCellCenterEstimate] = []

        if (
            ranges.front.valid
            and math.isfinite(front_distance)
            and front_distance > 0.0
        ):
            front_error = (
                float(front_distance)
                - float(self.grid_center_expected_front_distance_m)
            )
            if front_error < -tolerance:
                candidates.append(
                    AxialCellCenterEstimate(
                        valid=True,
                        error_m=float(front_error),
                        source='front_safety',
                        front_usable=True,
                        rear_usable=False,
                        disagreement_m=0.0,
                        reason=(
                            'front wall is physically too close for cell center; '
                            f'front_distance={front_distance:.3f} m; '
                            f'expected_front_distance='
                            f'{self.grid_center_expected_front_distance_m:.3f} m; '
                            f'front_error={front_error:.3f} m; '
                            'commanding reverse settle'
                        ),
                    )
                )

        if (
            ranges.rear.valid
            and math.isfinite(rear_distance)
            and rear_distance > 0.0
        ):
            rear_error = (
                float(self.grid_center_expected_rear_distance_m)
                - float(rear_distance)
            )
            if rear_error > tolerance:
                candidates.append(
                    AxialCellCenterEstimate(
                        valid=True,
                        error_m=float(rear_error),
                        source='rear_safety',
                        front_usable=False,
                        rear_usable=True,
                        disagreement_m=0.0,
                        reason=(
                            'rear wall is physically too close for cell center; '
                            f'rear_distance={rear_distance:.3f} m; '
                            f'expected_rear_distance='
                            f'{self.grid_center_expected_rear_distance_m:.3f} m; '
                            f'rear_error={rear_error:.3f} m; '
                            'commanding forward settle'
                        ),
                    )
                )

        if not candidates:
            return AxialCellCenterEstimate(
                valid=False,
                error_m=0.0,
                source='none',
                front_usable=False,
                rear_usable=False,
                disagreement_m=0.0,
                reason=(
                    'no safety axial correction needed: '
                    f'front_valid={ranges.front.valid}; '
                    f'front_distance={front_distance:.3f} m; '
                    f'rear_valid={ranges.rear.valid}; '
                    f'rear_distance={rear_distance:.3f} m'
                ),
            )

        candidates.sort(key=lambda estimate: abs(estimate.error_m), reverse=True)
        chosen = candidates[0]
        if len(candidates) == 1:
            return chosen

        rejected = candidates[1]
        return replace(
            chosen,
            disagreement_m=abs(float(chosen.error_m) - float(rejected.error_m)),
            reason=(
                f'{chosen.reason}; both front and rear safety references were close; '
                f'chose larger correction {chosen.source}={chosen.error_m:.3f} m '
                f'over {rejected.source}={rejected.error_m:.3f} m'
            ),
        )

    def _axial_cell_centering_estimate(
        self,
        expected,
        ranges: Optional[LidarRangeSnapshot],
    ) -> AxialCellCenterEstimate:
        if ranges is None:
            return AxialCellCenterEstimate(
                valid=False,
                error_m=0.0,
                source='none',
                front_usable=False,
                rear_usable=False,
                disagreement_m=0.0,
                reason='no fresh cardinal LiDAR snapshot',
            )

        front_distance = self._range_value(ranges.front)
        rear_distance = self._range_value(ranges.rear)

        return choose_axial_cell_centering(
            expected_front=bool(expected.front),
            front_valid=bool(ranges.front.valid),
            front_distance_m=front_distance,
            expected_front_distance_m=self.grid_center_expected_front_distance_m,
            expected_rear=bool(expected.rear),
            rear_valid=bool(ranges.rear.valid),
            rear_distance_m=rear_distance,
            expected_rear_distance_m=self.grid_center_expected_rear_distance_m,
            max_disagreement_m=self.grid_center_front_rear_agreement_tolerance_m,
        )

    def _reacquire_single_axial_cell_centering(
        self,
        *,
        expected,
        ranges: Optional[LidarRangeSnapshot],
        rejected: AxialCellCenterEstimate,
        direction: float,
    ) -> AxialCellCenterEstimate:
        if (
            not self.grid_cell_settle_rejected_axial_single_wall_enabled
            or ranges is None
            or rejected.source != 'front_rear_rejected'
        ):
            return rejected

        max_error_m = min(
            float(self.grid_cell_settle_abort_position_error_m),
            float(self.grid_cell_settle_rejected_axial_single_wall_max_error_m),
        )
        preferred_source = 'front' if float(direction) >= 0.0 else 'rear'
        candidates: list[tuple[int, float, float, AxialCellCenterEstimate]] = []

        front_distance = self._range_value(ranges.front)
        if (
            bool(expected.front)
            and ranges.front.valid
            and math.isfinite(front_distance)
            and math.isfinite(self.grid_center_expected_front_distance_m)
            and self.grid_center_expected_front_distance_m > 0.0
        ):
            front_error = (
                float(front_distance)
                - float(self.grid_center_expected_front_distance_m)
            )
            if abs(front_error) <= max_error_m:
                candidates.append(
                    (
                        0 if preferred_source == 'front' else 1,
                        abs(front_error),
                        float(front_distance),
                        AxialCellCenterEstimate(
                            valid=True,
                            error_m=float(front_error),
                            source='front_reacquired',
                            front_usable=True,
                            rear_usable=False,
                            disagreement_m=rejected.disagreement_m,
                            reason=(
                                'reacquired expected front wall after front/rear '
                                'axial disagreement; '
                                f'front_distance={front_distance:.3f} m; '
                                f'expected_front_distance='
                                f'{self.grid_center_expected_front_distance_m:.3f} m; '
                                f'front_error={front_error:.3f} m; '
                                f'max_single_wall_error={max_error_m:.3f} m; '
                                f'original_rejection={rejected.reason}'
                            ),
                        ),
                    )
                )

        rear_distance = self._range_value(ranges.rear)
        if (
            bool(expected.rear)
            and ranges.rear.valid
            and math.isfinite(rear_distance)
            and math.isfinite(self.grid_center_expected_rear_distance_m)
            and self.grid_center_expected_rear_distance_m > 0.0
        ):
            rear_error = (
                float(self.grid_center_expected_rear_distance_m)
                - float(rear_distance)
            )
            if abs(rear_error) <= max_error_m:
                candidates.append(
                    (
                        0 if preferred_source == 'rear' else 1,
                        abs(rear_error),
                        float(rear_distance),
                        AxialCellCenterEstimate(
                            valid=True,
                            error_m=float(rear_error),
                            source='rear_reacquired',
                            front_usable=False,
                            rear_usable=True,
                            disagreement_m=rejected.disagreement_m,
                            reason=(
                                'reacquired expected rear wall after front/rear '
                                'axial disagreement; '
                                f'rear_distance={rear_distance:.3f} m; '
                                f'expected_rear_distance='
                                f'{self.grid_center_expected_rear_distance_m:.3f} m; '
                                f'rear_error={rear_error:.3f} m; '
                                f'max_single_wall_error={max_error_m:.3f} m; '
                                f'original_rejection={rejected.reason}'
                            ),
                        ),
                    )
                )

        if not candidates:
            return rejected

        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return candidates[0][3]

    def _projected_axial_wall_cell_centering_estimate(
        self,
        *,
        reference: AxialWallReference,
        ranges: Optional[LidarRangeSnapshot],
    ) -> AxialCellCenterEstimate:
        """Use bounded mapped far-wall LiDAR as absolute longitudinal settle."""
        if not self.grid_cell_settle_projected_axial_enabled:
            return AxialCellCenterEstimate(
                valid=False,
                error_m=0.0,
                source='none',
                front_usable=False,
                rear_usable=False,
                disagreement_m=0.0,
                reason='projected axial settle disabled',
            )

        if ranges is None:
            return AxialCellCenterEstimate(
                valid=False,
                error_m=0.0,
                source='none',
                front_usable=False,
                rear_usable=False,
                disagreement_m=0.0,
                reason='projected axial unavailable: no fresh cardinal ranges',
            )

        if not reference.valid:
            return AxialCellCenterEstimate(
                valid=False,
                error_m=0.0,
                source='none',
                front_usable=False,
                rear_usable=False,
                disagreement_m=0.0,
                reason=f'projected axial unavailable: {reference.reason}',
            )

        max_open_cells = int(self.grid_cell_settle_projected_axial_max_open_cells)
        max_distance = float(self.grid_cell_settle_projected_axial_max_distance_m)
        max_error = float(self.grid_cell_settle_projected_axial_max_error_m)

        candidates: list[tuple[float, float, AxialCellCenterEstimate]] = []

        front_distance = self._range_value(ranges.front)
        if (
            reference.front
            and ranges.front.valid
            and math.isfinite(front_distance)
            and math.isfinite(reference.front_expected_distance_m)
            and reference.front_expected_distance_m > 0.0
            and reference.front_open_cells <= max_open_cells
            and front_distance <= max_distance
            and reference.front_expected_distance_m <= max_distance
        ):
            front_error = (
                float(front_distance)
                - float(reference.front_expected_distance_m)
            )
            if abs(front_error) <= max_error:
                candidates.append(
                    (
                        float(front_distance),
                        abs(front_error),
                        AxialCellCenterEstimate(
                            valid=True,
                            error_m=float(front_error),
                            source='projected_front',
                            front_usable=True,
                            rear_usable=False,
                            disagreement_m=0.0,
                            reason=(
                                'bounded projected front wall accepted; '
                                f'front_distance={front_distance:.3f} m; '
                                f'expected={reference.front_expected_distance_m:.3f} m; '
                                f'open_cells={reference.front_open_cells}; '
                                f'wall_cell={reference.front_wall_cell_idx}; '
                                f'error={front_error:.3f} m; '
                                f'reference={reference.reason}'
                            ),
                        ),
                    )
                )

        rear_distance = self._range_value(ranges.rear)
        if (
            reference.rear
            and ranges.rear.valid
            and math.isfinite(rear_distance)
            and math.isfinite(reference.rear_expected_distance_m)
            and reference.rear_expected_distance_m > 0.0
            and reference.rear_open_cells <= max_open_cells
            and rear_distance <= max_distance
            and reference.rear_expected_distance_m <= max_distance
        ):
            rear_error = (
                float(reference.rear_expected_distance_m)
                - float(rear_distance)
            )
            if abs(rear_error) <= max_error:
                candidates.append(
                    (
                        float(rear_distance),
                        abs(rear_error),
                        AxialCellCenterEstimate(
                            valid=True,
                            error_m=float(rear_error),
                            source='projected_rear',
                            front_usable=False,
                            rear_usable=True,
                            disagreement_m=0.0,
                            reason=(
                                'bounded projected rear wall accepted; '
                                f'rear_distance={rear_distance:.3f} m; '
                                f'expected={reference.rear_expected_distance_m:.3f} m; '
                                f'open_cells={reference.rear_open_cells}; '
                                f'wall_cell={reference.rear_wall_cell_idx}; '
                                f'error={rear_error:.3f} m; '
                                f'reference={reference.reason}'
                            ),
                        ),
                    )
                )

        if not candidates:
            return AxialCellCenterEstimate(
                valid=False,
                error_m=0.0,
                source='none',
                front_usable=False,
                rear_usable=False,
                disagreement_m=0.0,
                reason=(
                    'no bounded projected axial wall usable; '
                    f'front_valid={ranges.front.valid}; front={front_distance:.3f} m; '
                    f'rear_valid={ranges.rear.valid}; rear={rear_distance:.3f} m; '
                    f'max_distance={max_distance:.3f} m; '
                    f'max_open_cells={max_open_cells}; '
                    f'max_error={max_error:.3f} m; '
                    f'reference={reference.reason}'
                ),
            )

        if len(candidates) == 1:
            return candidates[0][2]

        first = candidates[0][2]
        second = candidates[1][2]
        disagreement = abs(float(first.error_m) - float(second.error_m))

        if disagreement <= float(
            self.grid_cell_settle_projected_axial_front_rear_agreement_m
        ):
            averaged_error = 0.5 * (float(first.error_m) + float(second.error_m))
            return AxialCellCenterEstimate(
                valid=True,
                error_m=averaged_error,
                source='projected_front_rear',
                front_usable=True,
                rear_usable=True,
                disagreement_m=disagreement,
                reason=(
                    'bounded projected front/rear walls agreed; '
                    f'front_error={first.error_m:.3f} m; '
                    f'rear_error={second.error_m:.3f} m; '
                    f'average_error={averaged_error:.3f} m; '
                    f'disagreement={disagreement:.3f} m'
                ),
            )

        # If far references disagree, trust the nearest measured wall.
        candidates.sort(key=lambda item: (item[0], item[1]))
        chosen = candidates[0][2]
        rejected = candidates[1][2]
        return replace(
            chosen,
            disagreement_m=disagreement,
            reason=(
                f'{chosen.reason}; projected front/rear disagreed; '
                f'chose nearest measured wall {chosen.source} over '
                f'{rejected.source}; disagreement={disagreement:.3f} m'
            ),
        )

    def _open_corridor_lattice_axial_cell_centering_estimate(
        self,
        ranges: Optional[LidarRangeSnapshot],
    ) -> AxialCellCenterEstimate:
        """Use visible front/rear LiDAR snapped to the cell-length lattice.

        This is mainly for task 2 exploration, where the full maze is not known
        yet. It does not use odom as distance truth.
        """
        if not self.grid_cell_settle_open_corridor_lattice_enabled:
            return AxialCellCenterEstimate(
                valid=False,
                error_m=0.0,
                source='none',
                front_usable=False,
                rear_usable=False,
                disagreement_m=0.0,
                reason='open-corridor lattice axial settle disabled',
            )

        if ranges is None:
            return AxialCellCenterEstimate(
                valid=False,
                error_m=0.0,
                source='none',
                front_usable=False,
                rear_usable=False,
                disagreement_m=0.0,
                reason='open-corridor lattice unavailable: no fresh cardinal ranges',
            )

        cell_length = max(float(self.cell_length_m), 1.0e-6)
        max_distance = float(self.grid_cell_settle_open_corridor_lattice_max_distance_m)
        max_error = float(self.grid_cell_settle_open_corridor_lattice_max_error_m)

        candidates: list[tuple[float, AxialCellCenterEstimate]] = []

        def add_candidate(
            *,
            source: str,
            sector,
            base_expected_m: float,
            use_front_sign: bool,
        ) -> None:
            distance = self._range_value(sector)
            if (
                not sector.valid
                or not math.isfinite(distance)
                or distance <= 0.0
                or distance > max_distance
            ):
                return

            lattice_cells = int(
                round((float(distance) - float(base_expected_m)) / cell_length)
            )
            if lattice_cells < 0:
                return

            expected = float(base_expected_m) + float(lattice_cells) * cell_length
            if expected <= 0.0 or expected > max_distance:
                return

            if use_front_sign:
                error = float(distance) - expected
            else:
                error = expected - float(distance)

            if abs(error) > max_error:
                return

            candidates.append(
                (
                    float(distance),
                    AxialCellCenterEstimate(
                        valid=True,
                        error_m=float(error),
                        source=source,
                        front_usable=bool(use_front_sign),
                        rear_usable=not bool(use_front_sign),
                        disagreement_m=0.0,
                        reason=(
                            f'open-corridor lattice {source} accepted; '
                            f'distance={distance:.3f} m; '
                            f'expected={expected:.3f} m; '
                            f'lattice_cells={lattice_cells}; '
                            f'cell_length={cell_length:.3f} m; '
                            f'error={error:.3f} m'
                        ),
                    ),
                )
            )

        add_candidate(
            source='open_lattice_front',
            sector=ranges.front,
            base_expected_m=self.grid_center_expected_front_distance_m,
            use_front_sign=True,
        )
        add_candidate(
            source='open_lattice_rear',
            sector=ranges.rear,
            base_expected_m=self.grid_center_expected_rear_distance_m,
            use_front_sign=False,
        )

        if not candidates:
            return AxialCellCenterEstimate(
                valid=False,
                error_m=0.0,
                source='none',
                front_usable=False,
                rear_usable=False,
                disagreement_m=0.0,
                reason=(
                    'open-corridor lattice unavailable: no bounded front/rear '
                    f'range within {max_distance:.3f} m and lattice error '
                    f'<= {max_error:.3f} m'
                ),
            )

        if len(candidates) == 1:
            return candidates[0][1]

        first = candidates[0][1]
        second = candidates[1][1]
        disagreement = abs(float(first.error_m) - float(second.error_m))

        if disagreement <= float(
            self.grid_cell_settle_open_corridor_lattice_agreement_m
        ):
            averaged_error = 0.5 * (float(first.error_m) + float(second.error_m))
            return AxialCellCenterEstimate(
                valid=True,
                error_m=float(averaged_error),
                source='open_lattice_front_rear',
                front_usable=True,
                rear_usable=True,
                disagreement_m=float(disagreement),
                reason=(
                    'open-corridor lattice front/rear agreed; '
                    f'front_error={first.error_m:.3f} m; '
                    f'rear_error={second.error_m:.3f} m; '
                    f'average_error={averaged_error:.3f} m; '
                    f'disagreement={disagreement:.3f} m'
                ),
            )

        # If front/rear disagree, trust the nearer physical wall. It is usually
        # less noisy than a far wall seen through several cells.
        candidates.sort(key=lambda item: item[0])
        chosen = candidates[0][1]
        rejected = candidates[1][1]

        return replace(
            chosen,
            disagreement_m=float(disagreement),
            reason=(
                f'{chosen.reason}; open-corridor lattice front/rear disagreed; '
                f'chose nearer wall {chosen.source} over {rejected.source}; '
                f'disagreement={disagreement:.3f} m'
            ),
        )

    @staticmethod
    def _settle_axis_command(
        error_m: float,
        tolerance_m: float,
        gain: float,
        max_speed: float,
        min_speed: float = 0.0,
    ) -> float:
        if not math.isfinite(error_m) or abs(error_m) <= float(tolerance_m):
            return 0.0

        cap = abs(float(max_speed))
        if cap <= 0.0:
            return 0.0

        floor = min(abs(float(min_speed)), cap)
        magnitude = clamp(abs(float(gain) * float(error_m)), floor, cap)
        return sign(float(error_m)) * magnitude

    def _settle_once_after_transit(
        self,
        *,
        reverse: bool = False,
        goal_handle,
        context: GridRunContext,
        start: Pose2D,
        direction: float,
        commanded_distance: float,
        start_ranges: Optional[LidarRangeSnapshot],
        initial_progress_m: float,
        initial_progress_source: str,
        initial_progress_reason: str,
        collision_check_enabled: bool,
        limits: VelocityLimits,
    ) -> GridCellSettleResult:
        if getattr(self, '_settle_consumed_for_transition', False):
            self.get_logger().warn('settle skipped: already consumed for this transition')
            return GridCellSettleResult(
                canceled=False,
                success=True,
                result_code=ExecuteMotionPrimitive.Result.SUCCESS,
                message='settle skipped: already consumed for this transition',
                position_error_m=abs(
                    float(commanded_distance) - float(initial_progress_m)
                ),
                heading_error_rad=0.0,
                heading_source='settle_consumed',
                progress_m=float(initial_progress_m),
                odom_progress_m=0.0,
                progress_source=str(initial_progress_source),
                progress_reason=str(initial_progress_reason),
                yaw_correction_used=False,
            )

        self._settle_consumed_for_transition = True

        if reverse:
            return self._compact_settle_after_reverse(
                goal_handle=goal_handle,
                context=context,
                start=start,
                direction=direction,
                commanded_distance=commanded_distance,
                start_ranges=start_ranges,
                initial_progress_m=initial_progress_m,
                initial_progress_source=initial_progress_source,
                initial_progress_reason=initial_progress_reason,
            )

        return self._settle_grid_cell_after_translation(
            goal_handle=goal_handle,
            context=context,
            start=start,
            direction=direction,
            commanded_distance=commanded_distance,
            start_ranges=start_ranges,
            initial_progress_m=initial_progress_m,
            initial_progress_source=initial_progress_source,
            initial_progress_reason=initial_progress_reason,
            collision_check_enabled=collision_check_enabled,
            limits=limits,
        )

    def _compact_settle_after_reverse(
        self,
        *,
        goal_handle,
        context: GridRunContext,
        start: Pose2D,
        direction: float,
        commanded_distance: float,
        start_ranges: Optional[LidarRangeSnapshot],
        initial_progress_m: float,
        initial_progress_source: str,
        initial_progress_reason: str,
    ) -> GridCellSettleResult:
        """
        Reverse/backtrack settle:
        stop, wait for fresh LiDAR, accept if inside tolerance.
        Do not perform another axial correction unless the error is large.
        """
        self._flush_stop(n=8, dt=0.05)
        time.sleep(0.30)

        stable_count = 0
        timeout_s = 1.50
        start_time = time.monotonic()

        last_axial_valid = False
        last_axial_error = abs(float(commanded_distance) - float(initial_progress_m))
        last_lateral_valid = False
        last_lateral_error = 0.0
        last_heading_valid = False
        last_heading_error_signed = 0.0
        last_heading_error = 0.0
        last_heading_source = 'none'
        last_progress = float(initial_progress_m)
        last_odom_progress = 0.0
        last_progress_source = str(initial_progress_source)
        last_progress_reason = str(initial_progress_reason)
        last_reason = 'not evaluated'

        destination_cell = 0
        settle_context: Optional[GridRunContext] = None
        if context.valid:
            destination_cell = advance_cells(
                context.start_idx,
                context.heading,
                context.run_cells,
                context.n,
                context.m,
            )
            if destination_cell > 0:
                settle_context = GridRunContext(
                    True,
                    context.n,
                    context.m,
                    destination_cell,
                    context.heading,
                    1,
                    context.l,
                    'destination cell compact reverse settle context',
                    robot_heading=context.robot_heading,
                )

        settle_virtual_progress_m = 0.5 * self.cell_length_m

        while time.monotonic() - start_time < timeout_s:
            if self._cancel_or_stop_requested(goal_handle):
                return GridCellSettleResult(
                    canceled=True,
                    success=False,
                    result_code=ExecuteMotionPrimitive.Result.CANCELED,
                    message='compact reverse settle canceled',
                    position_error_m=last_axial_error,
                    heading_error_rad=last_heading_error,
                    heading_source=last_heading_source,
                    progress_m=last_progress,
                    odom_progress_m=last_odom_progress,
                    progress_source=last_progress_source,
                    progress_reason=last_progress_reason,
                    yaw_correction_used=False,
                )

            snapshot = self._get_motion_snapshot()
            if snapshot is not None:
                progress_signed, _cross_track = self._translation_errors(
                    start,
                    snapshot.pose,
                    direction,
                )
                last_odom_progress = max(0.0, progress_signed)
                diagnostics = self._translation_diagnostics(
                    direction=direction,
                    odom_progress_m=last_odom_progress,
                    start_ranges=start_ranges,
                    end_ranges=self._cardinal_range_snapshot(),
                    track_lidar_progress=False,
                )
                progress_selection = self._select_translation_progress(
                    odom_progress_m=last_odom_progress,
                    diagnostics=diagnostics,
                )
                if progress_selection.valid:
                    last_progress = max(0.0, progress_selection.progress_m)
                    last_progress_source = progress_selection.source
                    last_progress_reason = progress_selection.reason

            current_ranges = self._cardinal_range_snapshot()
            alignment = self._grid_alignment_snapshot()

            if settle_context is not None:
                virtual, expected, centering = observe_centering_from_expected_side_walls(
                    context=settle_context,
                    alignment=alignment,
                    progress_m=settle_virtual_progress_m,
                    cell_length_m=self.cell_length_m,
                    boundary_margin_m=self.grid_virtual_cell_boundary_margin_m,
                    expected_half_width_m=self.grid_alignment_expected_half_width_m,
                    adjacent_wall_tolerance_m=self.grid_alignment_adjacent_wall_tolerance_m,
                    pair_width_tolerance_m=self.grid_alignment_pair_width_tolerance_m,
                    max_abs_yaw_error_rad=self.grid_live_max_abs_yaw_error_rad,
                    max_rms_error_m=self.grid_live_max_rms_error_m,
                    min_span_x_m=self.grid_live_min_span_x_m,
                    min_support_count=self.grid_live_min_support_count,
                    min_confidence=self.grid_lateral_min_confidence,
                )
                axial = self._axial_cell_centering_estimate(expected, current_ranges)
                if axial.source == 'front_rear_rejected':
                    axial = self._reacquire_single_axial_cell_centering(
                        expected=expected,
                        ranges=current_ranges,
                        rejected=axial,
                        direction=direction,
                    )
            else:
                virtual = VirtualCellEstimate(
                    valid=False,
                    cell_idx=0,
                    completed_cells=0,
                    progress_m=last_progress,
                    distance_into_cell_m=0.0,
                    boundary_zone=False,
                    reason='no known grid context for compact reverse settle',
                )
                centering = self._mapless_centering_from_alignment(alignment)
                axial = self._mapless_axial_cell_centering_estimate(current_ranges)

            yaw_observation = self._heading_observation_from_sources(
                self._manhattan_yaw_observation(),
                alignment,
            )
            heading_valid = bool(
                yaw_observation.valid
                and yaw_observation.confidence >= self.grid_manhattan_yaw_min_confidence
                and abs(yaw_observation.yaw_error_rad)
                <= self.grid_manhattan_yaw_max_abs_error_rad
            )
            heading_error_signed = (
                float(yaw_observation.yaw_error_rad) if heading_valid else 0.0
            )

            last_axial_valid = bool(axial.valid)
            last_axial_error = abs(float(axial.error_m)) if axial.valid else 0.0
            last_lateral_valid = bool(centering.valid)
            last_lateral_error = (
                abs(float(centering.lateral_error_m)) if centering.valid else 0.0
            )
            last_heading_valid = heading_valid
            last_heading_error_signed = heading_error_signed
            last_heading_error = abs(heading_error_signed)
            last_heading_source = (
                f'grid_yaw/{yaw_observation.source}' if heading_valid else 'none'
            )

            axial_ok = (not last_axial_valid) or last_axial_error <= 0.025
            lateral_ok = (not last_lateral_valid) or last_lateral_error <= 0.020
            yaw_ok = (not last_heading_valid) or last_heading_error <= 0.035

            if axial_ok and lateral_ok and yaw_ok:
                stable_count += 1
            else:
                stable_count = 0

            last_reason = (
                f'compact reverse settle: destination_cell={virtual.cell_idx}; '
                f'axial_valid={last_axial_valid}; axial_error={last_axial_error:.3f} m; '
                f'axial_source={axial.source}; '
                f'lateral_valid={last_lateral_valid}; '
                f'lateral_error={last_lateral_error:.3f} m; '
                f'heading_valid={last_heading_valid}; '
                f'heading_error={last_heading_error:.3f} rad; '
                f'yaw_reason={yaw_observation.reason}; '
                f'axial_reason={axial.reason}; lateral_reason={centering.reason}'
            )

            self._update_motion_state(
                distance_remaining=max(0.0, float(commanded_distance) - last_progress),
                distance_traveled=max(0.0, last_progress),
                heading_error=last_heading_error_signed,
                status=last_reason,
            )

            self._publish_feedback(
                goal_handle,
                progress=1.0,
                distance_remaining=max(0.0, float(commanded_distance) - last_progress),
                heading_remaining=last_heading_error_signed,
                state='compact_reverse_settle',
            )

            if stable_count >= 5:
                self._flush_stop(n=3, dt=0.03)
                return GridCellSettleResult(
                    canceled=False,
                    success=True,
                    result_code=ExecuteMotionPrimitive.Result.SUCCESS,
                    message=last_reason,
                    position_error_m=last_axial_error,
                    heading_error_rad=last_heading_error,
                    heading_source=last_heading_source,
                    progress_m=last_progress,
                    odom_progress_m=last_odom_progress,
                    progress_source=last_progress_source,
                    progress_reason=last_progress_reason,
                    yaw_correction_used=False,
                )

            time.sleep(0.05)

        self._flush_stop(n=5, dt=0.05)
        return GridCellSettleResult(
            canceled=False,
            success=False,
            result_code=ExecuteMotionPrimitive.Result.FINAL_ERROR_TOO_LARGE,
            message='compact reverse settle timeout: ' + last_reason,
            position_error_m=last_axial_error,
            heading_error_rad=last_heading_error,
            heading_source=last_heading_source,
            progress_m=last_progress,
            odom_progress_m=last_odom_progress,
            progress_source=last_progress_source,
            progress_reason=last_progress_reason,
            yaw_correction_used=False,
        )

    def _settle_grid_cell_after_translation(
        self,
        *,
        goal_handle,
        context: GridRunContext,
        start: Pose2D,
        direction: float,
        commanded_distance: float,
        start_ranges: Optional[LidarRangeSnapshot],
        initial_progress_m: float,
        initial_progress_source: str,
        initial_progress_reason: str,
        collision_check_enabled: bool,
        limits: VelocityLimits,
    ) -> GridCellSettleResult:
        settle_limits = VelocityLimits(
            max_linear_x_mps=min(
                limits.max_linear_x_mps,
                self.grid_cell_settle_max_linear_x_mps,
            ),
            max_linear_y_mps=min(
                limits.max_linear_y_mps,
                self.max_linear_y_mps,
                self.grid_cell_settle_max_linear_y_mps,
            ),
            max_angular_z_radps=min(
                limits.max_angular_z_radps,
                self.grid_cell_settle_max_yaw_radps,
            ),
        )

        start_time = time.monotonic()
        last_position_error = abs(float(commanded_distance) - float(initial_progress_m))
        last_heading_error = 0.0
        last_heading_source = 'odom'
        last_progress = float(initial_progress_m)
        last_odom_progress = 0.0
        last_progress_source = str(initial_progress_source)
        last_progress_reason = str(initial_progress_reason)
        yaw_correction_used = False
        stable_settle_samples = 0

        compact_transit_settle = bool(
            self.grid_cell_settle_compact_transit_accept_enabled
            and context.valid
            and int(context.run_cells)
            >= int(self.grid_cell_settle_compact_transit_min_run_cells)
        )

        effective_position_tolerance_m = float(
            self.grid_cell_settle_position_tolerance_m
        )
        effective_lateral_tolerance_m = float(
            self.grid_cell_settle_lateral_tolerance_m
        )
        effective_heading_tolerance_rad = float(
            self.grid_cell_settle_heading_tolerance_rad
        )

        if compact_transit_settle:
            effective_position_tolerance_m = max(
                effective_position_tolerance_m,
                float(self.grid_cell_settle_compact_transit_position_tolerance_m),
            )
            effective_lateral_tolerance_m = float(
                self.grid_cell_settle_lateral_tolerance_m
            )
            effective_heading_tolerance_rad = max(
                effective_heading_tolerance_rad,
                float(self.grid_cell_settle_compact_transit_heading_tolerance_rad),
            )

        destination_cell = 0
        settle_context: Optional[GridRunContext] = None
        projected_axial_reference: Optional[AxialWallReference] = None
        immediate_axial_seen = False
        axial_seen = False

        if context.valid:
            destination_cell = advance_cells(
                context.start_idx,
                context.heading,
                context.run_cells,
                context.n,
                context.m,
            )
            if destination_cell == 0:
                self.publish_zero_twist()
                return GridCellSettleResult(
                    canceled=False,
                    success=False,
                    result_code=ExecuteMotionPrimitive.Result.INTERNAL_ERROR,
                    message=(
                        'destination cell outside maze during settle: '
                        f'start_idx={context.start_idx}; '
                        f'heading={context.heading}; '
                        f'run_cells={context.run_cells}'
                    ),
                    position_error_m=last_position_error,
                    heading_error_rad=last_heading_error,
                    heading_source=last_heading_source,
                    progress_m=last_progress,
                    odom_progress_m=last_odom_progress,
                    progress_source=last_progress_source,
                    progress_reason=last_progress_reason,
                    yaw_correction_used=yaw_correction_used,
                )

            settle_context = GridRunContext(
                True,
                context.n,
                context.m,
                destination_cell,
                context.heading,
                1,
                context.l,
                'destination cell settle context',
                robot_heading=context.robot_heading,
            )

        if (
            self.grid_cell_settle_projected_axial_enabled
            and settle_context is not None
            and settle_context.valid
            and destination_cell > 0
            and int(context.run_cells)
            >= int(self.grid_cell_settle_projected_axial_min_run_cells)
        ):
            projected_axial_reference = nearest_axial_wall_reference(
                context=settle_context,
                cell_idx=destination_cell,
                cell_length_m=self.cell_length_m,
                expected_front_distance_m=self.grid_center_expected_front_distance_m,
                expected_rear_distance_m=self.grid_center_expected_rear_distance_m,
            )

        # Use the middle of a one-cell settle context so side-wall centering is
        # not weakened by the live-run boundary-zone penalty.
        settle_virtual_progress_m = 0.5 * self.cell_length_m

        while True:
            if self._cancel_or_stop_requested(goal_handle):
                return GridCellSettleResult(
                    canceled=True,
                    success=False,
                    result_code=ExecuteMotionPrimitive.Result.CANCELED,
                    message='cell settle canceled',
                    position_error_m=last_position_error,
                    heading_error_rad=last_heading_error,
                    heading_source=last_heading_source,
                    progress_m=last_progress,
                    odom_progress_m=last_odom_progress,
                    progress_source=last_progress_source,
                    progress_reason=last_progress_reason,
                    yaw_correction_used=yaw_correction_used,
                )

            elapsed = time.monotonic() - start_time
            if elapsed > self.grid_cell_settle_timeout_sec:
                self.publish_zero_twist()

                timeout_ok = bool(
                    last_position_error <= effective_position_tolerance_m
                    and last_heading_error <= effective_heading_tolerance_rad
                )

                return GridCellSettleResult(
                    canceled=False,
                    success=timeout_ok,
                    result_code=(
                        ExecuteMotionPrimitive.Result.SUCCESS
                        if timeout_ok
                        else ExecuteMotionPrimitive.Result.FINAL_ERROR_TOO_LARGE
                    ),
                    message=(
                        f'cell settle timed out after {elapsed:.2f}s; '
                        f'timeout_ok={timeout_ok}; '
                        f'last_pos_error={last_position_error:.3f} m; '
                        f'pos_tol={effective_position_tolerance_m:.3f} m; '
                        f'last_heading_error={last_heading_error:.3f} rad; '
                        f'heading_tol={effective_heading_tolerance_rad:.3f} rad; '
                        f'progress_source={last_progress_source}; '
                        f'{last_progress_reason}'
                    ),
                    position_error_m=last_position_error,
                    heading_error_rad=last_heading_error,
                    heading_source=f'{last_heading_source}/settle_timeout',
                    progress_m=last_progress,
                    odom_progress_m=last_odom_progress,
                    progress_source=last_progress_source,
                    progress_reason=last_progress_reason,
                    yaw_correction_used=yaw_correction_used,
                )

            snapshot = self._get_motion_snapshot()
            if snapshot is None:
                self.publish_zero_twist()
                return GridCellSettleResult(
                    canceled=False,
                    success=False,
                    result_code=ExecuteMotionPrimitive.Result.ODOM_UNAVAILABLE,
                    message='fresh odom unavailable during cell settle',
                    position_error_m=last_position_error,
                    heading_error_rad=last_heading_error,
                    heading_source=last_heading_source,
                    progress_m=last_progress,
                    odom_progress_m=last_odom_progress,
                    progress_source=last_progress_source,
                    progress_reason=last_progress_reason,
                    yaw_correction_used=yaw_correction_used,
                )

            progress_signed, _cross_track = self._translation_errors(
                start,
                snapshot.pose,
                direction,
            )
            odom_progress = max(0.0, progress_signed)
            last_odom_progress = odom_progress

            current_ranges = self._cardinal_range_snapshot()
            diagnostics = self._translation_diagnostics(
                direction=direction,
                odom_progress_m=odom_progress,
                start_ranges=start_ranges,
                end_ranges=current_ranges,
                track_lidar_progress=False,
            )
            progress_selection = self._select_translation_progress(
                odom_progress_m=odom_progress,
                diagnostics=diagnostics,
            )

            alignment = self._grid_alignment_snapshot()
            progress_selection = self._maybe_use_geometry_aware_odom_progress(
                progress_selection=progress_selection,
                grid_context=context,
                alignment=alignment,
                odom_progress_m=odom_progress,
                heading_error_rad=normalize_angle(start.yaw - snapshot.pose.yaw),
                direction=direction,
                target_distance_m=commanded_distance,
            )

            if progress_selection.valid:
                last_progress = max(0.0, progress_selection.progress_m)
                last_progress_source = progress_selection.source
                last_progress_reason = progress_selection.reason
                progress_error = abs(float(commanded_distance) - last_progress)
            else:
                progress_error = abs(float(commanded_distance) - last_progress)
                last_progress_source = progress_selection.source
                last_progress_reason = progress_selection.reason

            if settle_context is not None:
                virtual, expected, centering = observe_centering_from_expected_side_walls(
                    context=settle_context,
                    alignment=alignment,
                    progress_m=settle_virtual_progress_m,
                    cell_length_m=self.cell_length_m,
                    boundary_margin_m=self.grid_virtual_cell_boundary_margin_m,
                    expected_half_width_m=self.grid_alignment_expected_half_width_m,
                    adjacent_wall_tolerance_m=self.grid_alignment_adjacent_wall_tolerance_m,
                    pair_width_tolerance_m=self.grid_alignment_pair_width_tolerance_m,
                    max_abs_yaw_error_rad=self.grid_live_max_abs_yaw_error_rad,
                    max_rms_error_m=self.grid_live_max_rms_error_m,
                    min_span_x_m=self.grid_live_min_span_x_m,
                    min_support_count=self.grid_live_min_support_count,
                    min_confidence=self.grid_lateral_min_confidence,
                )

                if not virtual.valid or not expected.valid:
                    self.publish_zero_twist()
                    return GridCellSettleResult(
                        canceled=False,
                        success=False,
                        result_code=ExecuteMotionPrimitive.Result.INTERNAL_ERROR,
                        message=(
                            'invalid destination cell geometry during settle: '
                            f'destination_cell={destination_cell}; '
                            f'{virtual.reason}; {expected.reason}'
                        ),
                        position_error_m=progress_error,
                        heading_error_rad=last_heading_error,
                        heading_source=last_heading_source,
                        progress_m=last_progress,
                        odom_progress_m=last_odom_progress,
                        progress_source=last_progress_source,
                        progress_reason=last_progress_reason,
                        yaw_correction_used=yaw_correction_used,
                    )

                axial = self._axial_cell_centering_estimate(expected, current_ranges)
                if axial.source == 'front_rear_rejected':
                    axial = self._reacquire_single_axial_cell_centering(
                        expected=expected,
                        ranges=current_ranges,
                        rejected=axial,
                        direction=direction,
                    )
                axial_reference_expected = bool(expected.front or expected.rear)
                lateral_reference_expected = bool(expected.left or expected.right)
            else:
                virtual = VirtualCellEstimate(
                    valid=False,
                    cell_idx=0,
                    completed_cells=0,
                    progress_m=last_progress,
                    distance_into_cell_m=0.0,
                    boundary_zone=False,
                    reason='no known grid context for settle',
                )
                expected = ExpectedWalls(
                    valid=False,
                    cell_idx=0,
                    cell_value=0,
                    front=False,
                    rear=False,
                    left=False,
                    right=False,
                    reason='no known grid context for settle',
                )
                centering = self._mapless_centering_from_alignment(alignment)
                axial = self._mapless_axial_cell_centering_estimate(current_ranges)
                axial_reference_expected = axial.valid
                lateral_reference_expected = centering.valid

            immediate_axial_sources = (
                'front',
                'rear',
                'front_rear',
                'front_reacquired',
                'rear_reacquired',
                'front_safety',
                'rear_safety',
            )

            if axial.valid and axial.source in immediate_axial_sources:
                immediate_axial_seen = True
                axial_seen = True

            if (
                projected_axial_reference is not None
                and not (
                    self.grid_cell_settle_latch_immediate_axial_once_seen
                    and immediate_axial_seen
                )
                and (
                    not axial.valid
                    or axial.source in ('none', 'front_rear_rejected')
                )
            ):
                projected_axial = (
                    self._projected_axial_wall_cell_centering_estimate(
                        reference=projected_axial_reference,
                        ranges=current_ranges,
                    )
                )
                if projected_axial.valid:
                    axial = projected_axial
                    axial_reference_expected = True
                elif not axial.valid:
                    axial = replace(
                        axial,
                        reason=(
                            f'{axial.reason}; projected_axial='
                            f'{projected_axial.reason}'
                        ),
                    )

            if (
                not axial.valid
                and not axial_reference_expected
                and not immediate_axial_seen
            ):
                open_lattice_axial = (
                    self._open_corridor_lattice_axial_cell_centering_estimate(
                        current_ranges
                    )
                )
                if open_lattice_axial.valid:
                    axial = open_lattice_axial
                    axial_reference_expected = True
                else:
                    axial = replace(
                        axial,
                        reason=(
                            f'{axial.reason}; open_corridor_lattice='
                            f'{open_lattice_axial.reason}'
                        ),
                    )

            safety_axial = self._safety_axial_cell_centering_estimate(current_ranges)
            if safety_axial.valid and (
                not axial.valid
                or abs(safety_axial.error_m) > abs(axial.error_m)
            ):
                axial = safety_axial
                axial_reference_expected = True

            if axial.valid:
                axial_seen = True
                if axial.source in immediate_axial_sources:
                    immediate_axial_seen = True

            progress_reference_allowed = not (
                self.grid_cell_settle_disable_progress_after_axial_seen
                and axial_seen
            )

            compact_transit_use_immediate_axial_reference = bool(
                compact_transit_settle
                and axial.valid
                and axial.source in (
                    'front',
                    'rear',
                    'front_rear',
                    'front_reacquired',
                    'rear_reacquired',
                    'projected_front',
                    'projected_rear',
                    'projected_front_rear',
                    'open_lattice_front',
                    'open_lattice_rear',
                    'open_lattice_front_rear',
                    'front_safety',
                    'rear_safety',
                )
            )

            compact_transit_use_progress_reference = bool(
                compact_transit_settle
                and progress_selection.valid
                and progress_reference_allowed
                and not compact_transit_use_immediate_axial_reference
                and axial.source not in ('front_safety', 'rear_safety')
            )

            compact_transit_use_lidar_progress_reference = bool(
                compact_transit_use_progress_reference
                and self.grid_cell_settle_compact_transit_lidar_progress_strict
                and progress_selection.valid
                and progress_selection.source == 'lidar'
            )

            longitudinal_tolerance_m = float(effective_position_tolerance_m)
            if (
                compact_transit_use_immediate_axial_reference
                or compact_transit_use_lidar_progress_reference
            ):
                longitudinal_tolerance_m = float(
                    self.grid_cell_settle_position_tolerance_m
                )

            yaw_observation = self._heading_observation_from_sources(
                self._manhattan_yaw_observation(),
                alignment,
            )
            odom_heading_error = normalize_angle(start.yaw - snapshot.pose.yaw)
            yaw_valid = bool(
                yaw_observation.valid
                and yaw_observation.confidence >= self.grid_manhattan_yaw_min_confidence
                and abs(yaw_observation.yaw_error_rad)
                <= self.grid_manhattan_yaw_max_abs_error_rad
            )

            if yaw_valid:
                heading_error_signed = float(yaw_observation.yaw_error_rad)
                heading_error = abs(heading_error_signed)
                heading_source = f'grid_yaw/{yaw_observation.source}'
            else:
                heading_error_signed = odom_heading_error
                heading_error = abs(odom_heading_error)
                heading_source = 'odom'

            last_heading_error = heading_error
            last_heading_source = heading_source

            longitudinal_valid = False
            longitudinal_error_m = 0.0
            longitudinal_source = 'none'
            longitudinal_reason = 'no longitudinal correction reference'

            if axial.valid:
                longitudinal_valid = True
                longitudinal_error_m = float(axial.error_m)
                longitudinal_source = f'axial/{axial.source}'
                longitudinal_reason = axial.reason
            elif progress_selection.valid and progress_reference_allowed:
                longitudinal_valid = True
                longitudinal_error_m = float(direction) * (
                    float(commanded_distance) - float(last_progress)
                )
                longitudinal_source = f'progress/{progress_selection.source}'
                longitudinal_reason = progress_selection.reason
            else:
                longitudinal_reason = progress_selection.reason
                if progress_selection.valid and not progress_reference_allowed:
                    longitudinal_reason = (
                        'progress settle ignored because axial LiDAR was already '
                        f'used in this settle pass; progress_source={progress_selection.source}; '
                        f'progress_reason={progress_selection.reason}'
                    )

            travel_guard_active = False
            travel_guard_distance_m = float('inf')
            travel_guard_source = 'unavailable'
            travel_guard_label = 'none'
            travel_guard_threshold_m = float('inf')
            travel_guard_reason = 'travel guard inactive'

            stale_travel_error_m = float(commanded_distance) - float(last_progress)
            settle_command_direction = 0.0

            if (
                longitudinal_valid
                and abs(longitudinal_error_m)
                > longitudinal_tolerance_m
            ):
                settle_command_direction = sign(longitudinal_error_m)
            elif (
                not longitudinal_valid
                and stale_travel_error_m > longitudinal_tolerance_m
            ):
                settle_command_direction = sign(direction)

            travel_guard_should_check = bool(
                self.grid_cell_settle_travel_guard_enabled
                and settle_command_direction != 0.0
            )

            if travel_guard_should_check:
                (
                    travel_guard_distance_m,
                    travel_guard_source,
                    travel_guard_label,
                    travel_guard_threshold_m,
                ) = self._settle_travel_guard_distance(
                    settle_command_direction,
                    current_ranges,
                )

                travel_guard_active = bool(
                    math.isfinite(travel_guard_distance_m)
                    and travel_guard_distance_m <= travel_guard_threshold_m
                )

                if travel_guard_active:
                    travel_guard_reason = (
                        f'{travel_guard_label} guard active: {travel_guard_source}='
                        f'{travel_guard_distance_m:.3f} m <= '
                        f'{travel_guard_threshold_m:.3f} m; '
                        'accepting longitudinal settle without creeping farther toward the wall'
                    )
                else:
                    travel_guard_reason = (
                        f'{travel_guard_label} guard clear: {travel_guard_source}='
                        f'{travel_guard_distance_m:.3f} m > '
                        f'{travel_guard_threshold_m:.3f} m'
                    )

            if travel_guard_active and not longitudinal_valid:
                longitudinal_valid = True
                longitudinal_error_m = 0.0
                longitudinal_source = f'{travel_guard_label}_guard/{travel_guard_source}'
                longitudinal_reason = (
                    f'{travel_guard_reason}; original longitudinal unavailable: '
                    f'{progress_selection.reason}'
                )

            if heading_error > self.grid_cell_settle_abort_heading_error_rad:
                heading_source = f'{heading_source}/large_error_continue_settle'

            if (
                progress_selection.valid
                and progress_error > self.grid_cell_settle_abort_position_error_m
                and not compact_transit_use_immediate_axial_reference
            ):
                # Do not abort. Large progress error means "keep settling" unless
                # travel guard blocks motion. If LiDAR is valid, it is exactly the
                # signal we should use to correct the robot.
                pass

            if (
                axial.valid
                and axial.source not in ('front_safety', 'rear_safety')
                and abs(axial.error_m) > self.grid_cell_settle_abort_position_error_m
            ):
                # Do not abort on a large valid axial LiDAR error.
                # A large valid axial error is a correction command, not a failure.
                pass

            if axial.source == 'front_rear_rejected':
                axial_reference_expected = False
                if not progress_selection.valid:
                    longitudinal_valid = False
                    longitudinal_error_m = 0.0
                    longitudinal_source = 'none'
                    longitudinal_reason = (
                        'front/rear cell-center references disagree; '
                        f'ignoring axial settle instead of aborting: {axial.reason}'
                    )

            if compact_transit_use_immediate_axial_reference:
                axial_reference_expected = True
                longitudinal_valid = True
                longitudinal_error_m = float(axial.error_m)
                longitudinal_source = f'axial/immediate_{axial.source}'
                longitudinal_reason = (
                    'compact known-transit settle uses axial LiDAR reference '
                    '(immediate or bounded projected mapped wall); '
                    f'run_cells={context.run_cells}; '
                    f'axial_error={axial.error_m:.3f} m; '
                    f'axial_source={axial.source}; '
                    f'axial_reason={axial.reason}; '
                    f'commanded_distance={commanded_distance:.3f} m; '
                    f'progress={last_progress:.3f} m; '
                    f'progress_source={progress_selection.source}; '
                    f'progress_reason={progress_selection.reason}'
                )
            elif compact_transit_use_progress_reference:
                axial_reference_expected = False
                longitudinal_valid = True
                longitudinal_error_m = float(direction) * (
                    float(commanded_distance) - float(last_progress)
                )
                longitudinal_source = 'progress/compact_transit'
                longitudinal_reason = (
                    'compact known-transit settle uses progress as the '
                    'longitudinal reference because no strong endpoint LiDAR '
                    'axial correction is available; '
                    f'run_cells={context.run_cells}; '
                    f'commanded_distance={commanded_distance:.3f} m; '
                    f'progress={last_progress:.3f} m; '
                    f'progress_error={abs(float(commanded_distance) - float(last_progress)):.3f} m; '
                    f'original_progress_reason={progress_selection.reason}'
                )

            if longitudinal_valid:
                last_position_error = abs(longitudinal_error_m)
            else:
                last_position_error = abs(float(commanded_distance) - last_progress)

            if travel_guard_active:
                last_position_error = min(
                    last_position_error,
                    self.grid_cell_settle_position_tolerance_m,
                )

            if (
                self.grid_cell_settle_require_longitudinal_reference
                and not longitudinal_valid
            ):
                self.publish_zero_twist()
                return GridCellSettleResult(
                    canceled=False,
                    success=False,
                    result_code=ExecuteMotionPrimitive.Result.FINAL_ERROR_TOO_LARGE,
                    message=(
                        'longitudinal reference required but unavailable; '
                        f'{longitudinal_reason}'
                    ),
                    position_error_m=last_position_error,
                    heading_error_rad=heading_error,
                    heading_source=heading_source,
                    progress_m=last_progress,
                    odom_progress_m=last_odom_progress,
                    progress_source=last_progress_source,
                    progress_reason=last_progress_reason,
                    yaw_correction_used=yaw_correction_used,
                )

            longitudinal_reference_expected = bool(
                axial_reference_expected
                or (progress_selection.valid and progress_reference_allowed)
            )

            longitudinal_ok = True
            if longitudinal_reference_expected:
                longitudinal_ok = bool(
                    travel_guard_active
                    or (
                        longitudinal_valid
                        and abs(longitudinal_error_m)
                        <= longitudinal_tolerance_m
                    )
                )

            lateral_ok = True
            if lateral_reference_expected and centering.valid:
                lateral_ok = bool(
                    abs(centering.lateral_error_m)
                    <= effective_lateral_tolerance_m
                )

            heading_ok = heading_error <= effective_heading_tolerance_rad

            settle_stable_now = bool(longitudinal_ok and lateral_ok and heading_ok)

            if elapsed < self.grid_cell_settle_min_duration_sec:
                settle_stable_now = False

            if settle_stable_now:
                stable_settle_samples += 1
            else:
                stable_settle_samples = 0

            if (
                stable_settle_samples
                >= self.grid_cell_settle_required_stable_samples
            ):
                self.publish_zero_twist()

                returned_position_error = last_position_error
                returned_progress_m = last_progress
                if (
                    self.grid_cell_settle_compact_transit_mask_returned_progress_error
                    and compact_transit_settle
                    and not compact_transit_use_immediate_axial_reference
                    and not compact_transit_use_lidar_progress_reference
                ):
                    returned_position_error = min(
                        returned_position_error,
                        self.grid_cell_settle_position_tolerance_m,
                    )
                    returned_progress_m = max(
                        returned_progress_m,
                        float(commanded_distance),
                    )

                return GridCellSettleResult(
                    canceled=False,
                    success=True,
                    result_code=ExecuteMotionPrimitive.Result.SUCCESS,
                    message=(
                        f'cell settled: destination_cell={virtual.cell_idx}; '
                        f'compact_transit_settle={compact_transit_settle}; '
                        f'run_cells={context.run_cells if context.valid else 0}; '
                        f'longitudinal_valid={longitudinal_valid}; '
                        f'longitudinal_error={longitudinal_error_m:.3f} m; '
                        f'longitudinal_source={longitudinal_source}; '
                        f'travel_guard_active={travel_guard_active}; '
                        f'travel_guard_label={travel_guard_label}; '
                        f'travel_guard_distance={travel_guard_distance_m:.3f} m; '
                        f'travel_guard_source={travel_guard_source}; '
                        f'axial_valid={axial.valid}; axial_error={axial.error_m:.3f} m; '
                        f'axial_source={axial.source}; '
                        f'lateral_valid={centering.valid}; '
                        f'lateral_error={centering.lateral_error_m:.3f} m; '
                        f'heading_error={heading_error:.3f} rad; '
                        f'heading_source={heading_source}'
                    ),
                    position_error_m=returned_position_error,
                    heading_error_rad=heading_error,
                    heading_source=heading_source,
                    progress_m=returned_progress_m,
                    odom_progress_m=last_odom_progress,
                    progress_source=last_progress_source,
                    progress_reason=last_progress_reason,
                    yaw_correction_used=yaw_correction_used,
                )

            linear_x = 0.0
            if longitudinal_valid and not travel_guard_active:
                linear_x = self._settle_axis_command(
                    longitudinal_error_m,
                    longitudinal_tolerance_m,
                    self.k_distance,
                    settle_limits.max_linear_x_mps,
                    self.grid_cell_settle_min_linear_x_mps,
                )

            if linear_x != 0.0 and collision_check_enabled:
                clearance, clearance_name, stop_distance = self._travel_clearance(linear_x)
                if not math.isfinite(clearance):
                    clearance = float('inf')

                if clearance < stop_distance:
                    linear_x = 0.0
                    travel_guard_active = True
                    travel_guard_distance_m = clearance
                    travel_guard_source = clearance_name
                    travel_guard_label = (
                        'front' if clearance_name == 'front_clearance' else 'rear'
                    )
                    travel_guard_threshold_m = stop_distance
                    travel_guard_reason = (
                        f'{travel_guard_label} settle hold: '
                        f'{clearance_name}={clearance:.3f} m < {stop_distance:.3f} m; '
                        'not aborting during settle'
                    )

            linear_y = 0.0
            if centering.valid:
                linear_y = self._settle_axis_command(
                    centering.lateral_error_m,
                    self.grid_cell_settle_lateral_tolerance_m,
                    self.k_grid_lateral,
                    settle_limits.max_linear_y_mps,
                    0.0,
                )

            angular_z = self._settle_axis_command(
                heading_error_signed,
                self.grid_cell_settle_heading_tolerance_rad,
                self.k_grid_live_yaw if yaw_valid else self.k_heading,
                settle_limits.max_angular_z_radps,
                0.0,
            )

            AXIAL_DEADBAND_M = 0.015
            LATERAL_DEADBAND_M = 0.010
            YAW_DEADBAND_RAD = 0.025

            if longitudinal_valid and abs(longitudinal_error_m) < AXIAL_DEADBAND_M:
                linear_x = 0.0

            if centering.valid and abs(centering.lateral_error_m) < LATERAL_DEADBAND_M:
                linear_y = 0.0

            if abs(heading_error_signed) < YAW_DEADBAND_RAD:
                angular_z = 0.0

            yaw_correction_used = yaw_correction_used or bool(yaw_valid and angular_z != 0.0)

            cmd = Twist()
            cmd.linear.x = linear_x
            cmd.linear.y = linear_y
            cmd.angular.z = angular_z
            cmd = self._limiter.clamp(cmd, settle_limits)
            cmd = self._apply_acceleration_limits(cmd)
            self._publish_drive_cmd(
                forward=cmd.linear.x,
                lateral=cmd.linear.y,
                yaw_rate=cmd.angular.z,
                allow_lateral=True,
            )
            self._set_grid_yaw_control_status(
                bool(yaw_valid and angular_z != 0.0),
                angular_z if yaw_valid else 0.0,
                (
                    f'cell settle: destination_cell={virtual.cell_idx}; '
                    f'longitudinal={longitudinal_source}: {longitudinal_reason}; '
                    f'travel_guard={travel_guard_reason}; '
                    f'axial={axial.reason}; lateral={centering.reason}; '
                    f'yaw={yaw_observation.reason}'
                ),
            )

            self._update_motion_state(
                distance_remaining=max(0.0, float(commanded_distance) - last_progress),
                distance_traveled=max(0.0, last_progress),
                heading_error=heading_error_signed,
                status=(
                    f'cell_settle: destination_cell={virtual.cell_idx}, '
                    f'expected_walls=F{int(expected.front)}B{int(expected.rear)}'
                    f'L{int(expected.left)}R{int(expected.right)}, '
                    f'long_valid={longitudinal_valid}, '
                    f'long_error={longitudinal_error_m:.3f} m, '
                    f'long_source={longitudinal_source}, '
                    f'travel_guard={travel_guard_active}, '
                    f'travel_guard_label={travel_guard_label}, '
                    f'travel_guard_dist={travel_guard_distance_m:.3f} m, '
                    f'axial_valid={axial.valid}, axial_error={axial.error_m:.3f} m, '
                    f'axial_source={axial.source}, '
                    f'lat_valid={centering.valid}, '
                    f'lat_error={centering.lateral_error_m:.3f} m, '
                    f'heading_error={heading_error:.3f} rad, '
                    f'heading_source={heading_source}, '
                    f'cmd_x={cmd.linear.x:.3f}, '
                    f'cmd_y={cmd.linear.y:.3f}, '
                    f'cmd_yaw={cmd.angular.z:.3f}'
                ),
            )

            self._publish_feedback(
                goal_handle,
                progress=1.0,
                distance_remaining=max(0.0, float(commanded_distance) - last_progress),
                heading_remaining=heading_error_signed,
                state='cell_settle',
            )

            time.sleep(1.0 / self.control_rate_hz)

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

    def _settle_front_distance(
        self,
        ranges: Optional[LidarRangeSnapshot],
    ) -> tuple[float, str]:
        if ranges is not None and ranges.front.valid:
            distance = self._range_value(ranges.front)
            if math.isfinite(distance) and distance > 0.0:
                return distance, 'front_cardinal_median'

        clearance = self._front_clearance()
        if math.isfinite(clearance) and clearance > 0.0:
            return float(clearance), 'front_sector_min'

        return float('inf'), 'unavailable'

    def _settle_rear_distance(
        self,
        ranges: Optional[LidarRangeSnapshot],
    ) -> tuple[float, str]:
        if ranges is not None and ranges.rear.valid:
            distance = self._range_value(ranges.rear)
            if math.isfinite(distance) and distance > 0.0:
                return distance, 'rear_cardinal_median'

        clearance = self._rear_clearance()
        if math.isfinite(clearance) and clearance > 0.0:
            return float(clearance), 'rear_sector_min'

        return float('inf'), 'unavailable'

    def _settle_travel_guard_distance(
        self,
        command_direction: float,
        ranges: Optional[LidarRangeSnapshot],
    ) -> tuple[float, str, str, float]:
        if command_direction >= 0.0:
            distance, source = self._settle_front_distance(ranges)
            return (
                distance,
                source,
                'front',
                float(self.grid_cell_settle_min_front_distance_m),
            )

        distance, source = self._settle_rear_distance(ranges)
        return (
            distance,
            source,
            'rear',
            float(self.grid_cell_settle_min_rear_distance_m),
        )

    @staticmethod
    def _range_value(measurement: SectorRange) -> float:
        return float(measurement.median_m) if measurement.valid else 0.0

    def _reset_temporal_lidar_progress_tracker(self) -> None:
        self._temporal_lidar_progress_valid = False
        self._temporal_lidar_progress_m = 0.0
        self._temporal_lidar_progress_source = 'none'
        self._temporal_lidar_degraded_samples = 0

    def _choose_tracked_lidar_progress(
        self,
        raw: LidarProgressEstimate,
        *,
        front_valid: bool,
        front_progress_m: float,
        rear_valid: bool,
        rear_progress_m: float,
    ) -> TemporalLidarProgressEstimate:
        if not self.lidar_progress_temporal_filter_enabled:
            if raw.valid:
                return TemporalLidarProgressEstimate(
                    valid=True,
                    progress_m=raw.progress_m,
                    source=raw.source,
                    degraded=False,
                    degraded_samples=0,
                    disagreement_m=raw.disagreement_m,
                    reason=raw.reason,
                )
            return TemporalLidarProgressEstimate(
                valid=False,
                progress_m=0.0,
                source='raw_invalid',
                degraded=False,
                degraded_samples=0,
                disagreement_m=raw.disagreement_m,
                reason=raw.reason,
            )

        tracked = choose_temporal_lidar_progress(
            raw=raw,
            previous_valid=self._temporal_lidar_progress_valid,
            previous_progress_m=self._temporal_lidar_progress_m,
            previous_source=self._temporal_lidar_progress_source,
            front_valid=front_valid,
            front_progress_m=front_progress_m,
            rear_valid=rear_valid,
            rear_progress_m=rear_progress_m,
            degraded_samples=self._temporal_lidar_degraded_samples,
            max_backtrack_m=self.lidar_progress_temporal_max_backtrack_m,
            max_jump_m=self.lidar_progress_temporal_max_jump_m,
            max_degraded_samples=self.lidar_progress_temporal_max_degraded_samples,
        )

        if tracked.valid:
            self._temporal_lidar_progress_valid = True
            self._temporal_lidar_progress_m = tracked.progress_m
            self._temporal_lidar_progress_source = tracked.source
            self._temporal_lidar_degraded_samples = tracked.degraded_samples
        else:
            self._temporal_lidar_degraded_samples = tracked.degraded_samples

        return tracked

    def _translation_diagnostics(
        self,
        direction: float,
        odom_progress_m: float,
        start_ranges: Optional[LidarRangeSnapshot],
        end_ranges: Optional[LidarRangeSnapshot],
        track_lidar_progress: bool = False,
    ) -> TranslationDiagnostics:
        if start_ranges is None or end_ranges is None:
            return TranslationDiagnostics(
                odom_progress_m=float(max(0.0, odom_progress_m)),
                lidar_progress_valid=False,
                lidar_progress_source='none',
                lidar_progress_reason='LiDAR range snapshot unavailable or stale',
            )

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
            odom_progress_m=odom_progress_m,
            odom_arbitration_tolerance_m=(
                self.lidar_progress_odom_arbitration_tolerance_m
            ),
            odom_arbitration_min_margin_m=(
                self.lidar_progress_odom_arbitration_min_margin_m
            ),
        )

        temporal_degraded = False
        temporal_reason = ''
        if track_lidar_progress:
            tracked_lidar_progress = self._choose_tracked_lidar_progress(
                lidar_estimate,
                front_valid=front_valid,
                front_progress_m=front_progress,
                rear_valid=rear_valid,
                rear_progress_m=rear_progress,
            )
            lidar_estimate = LidarProgressEstimate(
                valid=tracked_lidar_progress.valid,
                progress_m=tracked_lidar_progress.progress_m,
                source=tracked_lidar_progress.source,
                disagreement_m=tracked_lidar_progress.disagreement_m,
                reason=tracked_lidar_progress.reason,
            )
            temporal_degraded = tracked_lidar_progress.degraded
            temporal_reason = tracked_lidar_progress.reason

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
            lidar_progress_temporal_degraded=temporal_degraded,
            lidar_progress_temporal_reason=temporal_reason,
        )

    def _maybe_use_geometry_aware_odom_progress(
        self,
        *,
        progress_selection,
        grid_context: GridRunContext,
        alignment: GridAlignmentEstimate,
        odom_progress_m: float,
        heading_error_rad: float,
        direction: float,
        target_distance_m: float,
    ):
        """No odom progress fallback.

        Odom is deliberately not used as longitudinal distance truth.
        It is still passed into choose_lidar_progress() earlier as a referee
        between disagreeing front/rear LiDAR candidates. If LiDAR progress is
        unavailable here, the translation loop enters LiDAR recovery crawl
        instead of manufacturing odom progress.
        """
        return progress_selection

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
            f'grid_lateral_drift_reason={diagnostics.grid_lateral_drift_reason}; '
            f'live_grid_mode={diagnostics.live_grid_mode}; '
            f'live_grid_context_valid={diagnostics.live_grid_context_valid}; '
            f'live_grid_virtual_cell={diagnostics.live_grid_virtual_cell}; '
            f'live_grid_boundary_zone={diagnostics.live_grid_boundary_zone}; '
            f'live_grid_expected_front={diagnostics.live_grid_expected_front}; '
            f'live_grid_expected_rear={diagnostics.live_grid_expected_rear}; '
            f'live_grid_expected_left={diagnostics.live_grid_expected_left}; '
            f'live_grid_expected_right={diagnostics.live_grid_expected_right}; '
            f'live_grid_yaw_active={diagnostics.live_grid_yaw_active}; '
            f'live_grid_lateral_active={diagnostics.live_grid_lateral_active}; '
            f'live_grid_yaw_cmd={diagnostics.live_grid_yaw_cmd_radps:.3f} rad/s; '
            f'live_grid_lateral_cmd={diagnostics.live_grid_lateral_cmd_mps:.3f} m/s; '
            f'live_grid_reason={diagnostics.live_grid_reason}; '
            f'lidar_progress_temporal_degraded='
            f'{diagnostics.lidar_progress_temporal_degraded}; '
            f'lidar_progress_temporal_reason='
            f'{diagnostics.lidar_progress_temporal_reason}'
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

    def _fresh_scan_copy_with_token(
        self,
    ) -> tuple[Optional[LaserScan], Optional[float]]:
        with self._scan_lock:
            if self._latest_scan is None or self._last_scan_monotonic is None:
                return None, None

            if time.monotonic() - self._last_scan_monotonic > self.scan_timeout_sec:
                return None, None

            return (
                deepcopy(self._latest_scan),
                float(self._last_scan_monotonic),
            )

    def _fresh_scan_copy(self) -> Optional[LaserScan]:
        scan, _ = self._fresh_scan_copy_with_token()
        return scan

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
        fallback = abs(angle) / max(abs(speed), 1e-3) + self.rotate_timeout_margin_sec
        fallback = max(fallback, 4.0)
        return self._goal_timeout(requested, fallback)

    def _ros_time_seconds(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

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
        zero = _zero_twist()
        self._cmd_vel_pub.publish(zero)
        self._last_commanded_twist = zero
        self._last_command_time = time.monotonic()
        self._set_grid_yaw_control_status(False, 0.0, '')

    def _flush_stop(self, n: int = 5, dt: float = 0.05) -> None:
        """Hard stop. Use before/after rotate and before sensing/settling."""
        msg = _zero_twist()
        for _ in range(max(0, int(n))):
            self._cmd_vel_pub.publish(msg)
            self._last_commanded_twist = msg
            self._last_command_time = time.monotonic()
            self._set_grid_yaw_control_status(False, 0.0, '')
            time.sleep(max(0.0, float(dt)))

    def _publish_rotate_only(self, yaw_rate: float) -> None:
        """Rotation must never carry lateral/forward correction."""
        msg = _zero_twist()
        msg.angular.z = float(yaw_rate)
        self._cmd_vel_pub.publish(msg)
        self._last_commanded_twist = msg
        self._last_command_time = time.monotonic()

    def _publish_drive_cmd(
        self,
        forward: float,
        lateral: float = 0.0,
        yaw_rate: float = 0.0,
        *,
        allow_lateral: bool = True,
    ) -> None:
        msg = _zero_twist()
        msg.linear.x = float(forward)
        msg.linear.y = float(lateral if allow_lateral else 0.0)
        msg.angular.z = float(yaw_rate)
        self._cmd_vel_pub.publish(msg)
        self._last_commanded_twist = msg
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
            grid_alignment = self._grid_alignment
            grid_yaw_correction_active = self._grid_yaw_correction_active
            grid_yaw_correction_radps = self._grid_yaw_correction_radps
            grid_yaw_control_reason = self._grid_yaw_control_reason
            grid_live_mode = self._grid_live_last_command.mode
            grid_context_valid = self._grid_live_last_context_valid
            grid_virtual_cell = self._grid_live_last_virtual_cell
            grid_virtual_cell_progress = self._grid_live_last_virtual_cell_progress_m
            grid_boundary_zone = self._grid_live_last_boundary_zone
            grid_expected_front = self._grid_live_last_expected_front
            grid_expected_rear = self._grid_live_last_expected_rear
            grid_expected_left = self._grid_live_last_expected_left
            grid_expected_right = self._grid_live_last_expected_right
            grid_live_yaw_active = self._grid_live_last_command.yaw_active
            grid_live_lateral_active = self._grid_live_last_command.lateral_active
            grid_live_yaw_cmd = self._grid_live_last_command.angular_z_radps
            grid_live_lateral_cmd = self._grid_live_last_command.linear_y_mps
            grid_live_reason = self._grid_live_last_command.reason
            yaw_obs = self._grid_live_last_yaw_observation
            center_obs = self._grid_live_last_centering_observation
            grid_yaw_mode = self._grid_yaw_mode
            grid_lateral_mode = self._grid_lateral_mode

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
        msg.grid_alignment_source = str(grid_alignment.source)
        msg.grid_alignment_confidence = float(grid_alignment.confidence)
        msg.grid_yaw_correction_active = bool(grid_yaw_correction_active)
        msg.grid_yaw_correction_radps = float(grid_yaw_correction_radps)
        msg.grid_yaw_control_reason = str(grid_yaw_control_reason)
        msg.grid_live_mode = str(grid_live_mode)
        msg.grid_context_valid = bool(grid_context_valid)
        msg.grid_virtual_cell = int(grid_virtual_cell)
        msg.grid_virtual_cell_progress_m = float(grid_virtual_cell_progress)
        msg.grid_virtual_cell_boundary_zone = bool(grid_boundary_zone)
        msg.grid_expected_front_wall = bool(grid_expected_front)
        msg.grid_expected_rear_wall = bool(grid_expected_rear)
        msg.grid_expected_left_wall = bool(grid_expected_left)
        msg.grid_expected_right_wall = bool(grid_expected_right)
        msg.grid_live_yaw_active = bool(grid_live_yaw_active)
        msg.grid_live_lateral_active = bool(grid_live_lateral_active)
        msg.grid_live_yaw_cmd_radps = float(grid_live_yaw_cmd)
        msg.grid_live_lateral_cmd_mps = float(grid_live_lateral_cmd)
        msg.grid_live_reason = str(grid_live_reason)
        msg.grid_yaw_mode = str(grid_yaw_mode)
        msg.grid_yaw_valid = bool(yaw_obs.valid)
        msg.grid_yaw_error_rad = float(yaw_obs.yaw_error_rad)
        msg.grid_yaw_confidence = float(yaw_obs.confidence)
        msg.grid_yaw_source = str(yaw_obs.source)
        msg.grid_yaw_line_count = int(yaw_obs.line_count)
        msg.grid_yaw_dominant_axis_rad = float(yaw_obs.dominant_axis_rad)
        msg.grid_yaw_total_weight = float(yaw_obs.total_weight)
        msg.grid_yaw_concentration = float(yaw_obs.concentration)
        msg.grid_yaw_reason = str(yaw_obs.reason)
        msg.grid_lateral_mode = str(grid_lateral_mode)
        msg.grid_lateral_valid = bool(center_obs.valid)
        msg.grid_lateral_error_m = float(center_obs.lateral_error_m)
        msg.grid_lateral_confidence = float(center_obs.confidence)
        msg.grid_lateral_source = str(center_obs.source)
        msg.grid_lateral_reason = str(center_obs.reason)

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
