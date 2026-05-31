from copy import deepcopy
import math
import threading
import time

from darth_maul_control.geometry import (
    normalize_angle,
    planar_distance,
    yaw_from_quaternion,
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
from std_srvs.srv import Trigger


class DarthMaulControlNode(Node):
    def __init__(self):
        super().__init__('darth_maul_control')

        self.control_rate_hz = self._positive_float_param('control_rate_hz', 20.0)
        self.k_position = self._nonnegative_float_param('k_position', 0.6)
        self.k_heading = self._nonnegative_float_param('k_heading', 1.2)
        self.default_position_tolerance_m = self._positive_float_param(
            'default_position_tolerance_m',
            0.04,
        )
        self.default_heading_tolerance_rad = self._positive_float_param(
            'default_heading_tolerance_rad',
            0.10,
        )
        self.default_timeout_sec = self._positive_float_param(
            'default_timeout_sec',
            30.0,
        )
        self.odom_timeout_sec = self._positive_float_param('odom_timeout_sec', 0.5)

        self.odom_topic = self._string_param('odom_topic', 'odom')
        self.cmd_vel_topic = self._string_param('cmd_vel_topic', 'controller/cmd_vel')
        self.status_topic = self._string_param(
            'status_topic',
            '/darth_maul_control/status',
        )

        self._limiter = VelocityLimiter(
            self._nonnegative_float_param('max_linear_x_mps', 0.20),
            self._nonnegative_float_param('max_linear_y_mps', 0.20),
            self._nonnegative_float_param('max_angular_z_radps', 0.50),
        )

        self._state_lock = threading.RLock()
        self._odom_lock = threading.RLock()
        self._state = ControlStatus.STATE_IDLE
        self._status = 'ready'
        self._command_enabled = True
        self._active_primitive_type = 0
        self._distance_remaining_m = 0.0
        self._distance_traveled_m = 0.0
        self._heading_error_rad = 0.0
        self._active_goal = False
        self._stop_requested = False

        self._current_odom = None
        self._last_odom_monotonic = None

        self._callback_group = ReentrantCallbackGroup()
        self._cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self._status_pub = self.create_publisher(ControlStatus, self.status_topic, 10)
        self._odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self._odom_callback,
            10,
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

        self.get_logger().warn(
            'Do not run joystick_control or teleop_key_control concurrently with '
            'darth_maul_control unless cmd_vel is remapped or muxed.'
        )
        self.get_logger().warn(
            'This primitive controller depends on odometry quality; validate forward, '
            'backward, strafe, rotation, stop, and odom response before trusting it.'
        )
        self.publish_zero_twist()
        self._publish_status()

    def _string_param(self, name, default):
        value = self.declare_parameter(name, default).value
        if not isinstance(value, str) or not value:
            self.get_logger().warn(f'Parameter {name} is invalid; using {default!r}.')
            return default
        return value

    def _positive_float_param(self, name, default):
        value = self.declare_parameter(name, default).value
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0.0:
            self.get_logger().warn(f'Parameter {name} is invalid; using {default}.')
            return float(default)
        return float(value)

    def _nonnegative_float_param(self, name, default):
        value = self.declare_parameter(name, default).value
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0.0:
            self.get_logger().warn(f'Parameter {name} is invalid; using {default}.')
            return float(default)
        return float(value)

    def _odom_callback(self, msg):
        with self._odom_lock:
            self._current_odom = msg
            self._last_odom_monotonic = time.monotonic()

    def _goal_callback(self, goal_request):
        del goal_request
        return self._claim_motion_goal('execute_motion_primitive')

    def _primitive_goal_callback(self, goal_request):
        return self._goal_callback(goal_request)

    def _claim_motion_goal(self, action_name):
        with self._state_lock:
            if self._active_goal:
                self.get_logger().warn(
                    f'Rejecting {action_name} goal; another goal is active.'
                )
                return GoalResponse.REJECT
            self._active_goal = True
            self._stop_requested = False
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        del goal_handle
        self._request_stop('action cancel requested')
        return CancelResponse.ACCEPT

    def _handle_stop(self, request, response):
        del request
        with self._state_lock:
            active_goal = self._active_goal

        if active_goal:
            self._request_stop('stop service requested')
            response.message = 'Stop requested; zero Twist published.'
        else:
            with self._state_lock:
                self._stop_requested = False
                self._command_enabled = True
                self._state = ControlStatus.STATE_IDLE
                self._status = 'stopped'
                self._active_primitive_type = 0
                self._distance_remaining_m = 0.0
                self._distance_traveled_m = 0.0
                self._heading_error_rad = 0.0
            self.publish_zero_twist()
            self._publish_status()
            response.message = 'Stopped; zero Twist published.'

        response.success = True
        return response

    def _request_stop(self, reason):
        with self._state_lock:
            self._stop_requested = True
            self._command_enabled = False
            self._state = ControlStatus.STATE_STOPPING
            self._status = reason
        self.publish_zero_twist()
        self._publish_status()

    def _execute_callback(self, goal_handle):
        goal = goal_handle.request
        start_pose = None
        try:
            valid, message = self._validate_goal(goal)
            if not valid:
                return self._finish_action(
                    goal_handle,
                    'abort',
                    ControlStatus.STATE_FAILED,
                    ExecuteMotionPrimitive.Result.RESULT_INVALID_GOAL,
                    False,
                    message,
                )

            position_tolerance_m = (
                goal.position_tolerance_m
                if goal.position_tolerance_m > 0.0
                else self.default_position_tolerance_m
            )
            heading_tolerance_rad = (
                goal.heading_tolerance_rad
                if goal.heading_tolerance_rad > 0.0
                else self.default_heading_tolerance_rad
            )
            timeout_sec = self._goal_timeout_sec(goal.timeout)
            limits = self._velocity_limits_for_primitive(goal)

            if not self._is_odom_fresh():
                return self._finish_action(
                    goal_handle,
                    'abort',
                    ControlStatus.STATE_FAILED,
                    ExecuteMotionPrimitive.Result.RESULT_ODOM_UNAVAILABLE,
                    False,
                    'Odometry is unavailable or stale.',
                )

            start_pose = self._get_current_pose_stamped()
            if start_pose is None:
                return self._finish_action(
                    goal_handle,
                    'abort',
                    ControlStatus.STATE_FAILED,
                    ExecuteMotionPrimitive.Result.RESULT_ODOM_UNAVAILABLE,
                    False,
                    'Odometry is unavailable.',
                )

            primitive_type = goal.primitive_type
            status = self._primitive_status_text(primitive_type)
            if primitive_type in self._translation_primitive_types():
                target_pose = self._translation_target_pose(
                    primitive_type,
                    start_pose,
                    goal.value,
                )
                remaining_distance = planar_distance(start_pose, target_pose)
                target_heading = yaw_from_quaternion(start_pose.pose.orientation)
                self._start_execution_status(
                    primitive_type,
                    status,
                    remaining_distance,
                    0.0,
                    0.0,
                )
                return self._run_translation(
                    goal_handle,
                    primitive_type,
                    start_pose,
                    target_pose,
                    target_heading,
                    position_tolerance_m,
                    timeout_sec,
                    limits,
                    status,
                )

            current_yaw = yaw_from_quaternion(start_pose.pose.orientation)
            target_heading = self._rotation_target_heading(current_yaw, goal.value)
            heading_error = normalize_angle(target_heading - current_yaw)
            self._start_execution_status(
                primitive_type,
                status,
                0.0,
                0.0,
                heading_error,
            )
            return self._run_rotation(
                goal_handle,
                start_pose,
                target_heading,
                heading_tolerance_rad,
                timeout_sec,
                limits,
                status,
            )
        except Exception as exc:
            self.get_logger().error(f'Motion primitive failed: {exc}')
            final_pose, distance_traveled, position_error, heading_error = (
                self._result_values(start_pose)
            )
            return self._finish_action(
                goal_handle,
                'abort',
                ControlStatus.STATE_FAILED,
                ExecuteMotionPrimitive.Result.RESULT_CONTROL_FAILED,
                False,
                f'Motion primitive failed: {exc}',
                start_pose,
                final_pose,
                distance_traveled,
                position_error,
                heading_error,
            )
        finally:
            with self._state_lock:
                self._active_goal = False
                self._command_enabled = True
                self._stop_requested = False
            self._publish_status()

    def _execute_primitive_callback(self, goal_handle):
        return self._execute_callback(goal_handle)

    def _validate_goal(self, goal):
        valid_types = {
            ExecuteMotionPrimitive.Goal.DRIVE_FORWARD,
            ExecuteMotionPrimitive.Goal.DRIVE_BACKWARD,
            ExecuteMotionPrimitive.Goal.STRAFE_LEFT,
            ExecuteMotionPrimitive.Goal.STRAFE_RIGHT,
            ExecuteMotionPrimitive.Goal.ROTATE_RELATIVE,
        }
        if goal.primitive_type not in valid_types:
            return False, 'primitive_type must be a supported motion primitive.'

        if not math.isfinite(goal.value):
            return False, 'value must be finite.'

        if goal.primitive_type in self._translation_primitive_types() and goal.value <= 0.0:
            return False, 'Translation primitive value must be > 0.'

        if (
            not math.isfinite(goal.position_tolerance_m)
            or goal.position_tolerance_m < 0.0
        ):
            return False, 'position_tolerance_m must be finite and nonnegative.'
        if (
            not math.isfinite(goal.heading_tolerance_rad)
            or goal.heading_tolerance_rad < 0.0
        ):
            return False, 'heading_tolerance_rad must be finite and nonnegative.'

        if (
            goal.timeout.sec < 0
            or goal.timeout.nanosec < 0
            or goal.timeout.nanosec >= 1000000000
        ):
            return False, 'timeout must be nonnegative and nanosec < 1000000000.'

        primitive_limits = (
            ('max_linear_x_mps', goal.max_linear_x_mps),
            ('max_linear_y_mps', goal.max_linear_y_mps),
            ('max_angular_z_radps', goal.max_angular_z_radps),
        )
        for field_name, value in primitive_limits:
            if not math.isfinite(value) or value < 0.0:
                return False, f'{field_name} must be finite and nonnegative.'

        return True, ''

    def _validate_primitive_goal(self, goal):
        return self._validate_goal(goal)

    def _goal_timeout_sec(self, timeout_msg):
        if timeout_msg.sec == 0 and timeout_msg.nanosec == 0:
            return self.default_timeout_sec
        return self._duration_to_seconds(timeout_msg)

    @staticmethod
    def _duration_to_seconds(duration_msg):
        return float(duration_msg.sec) + float(duration_msg.nanosec) * 1e-9

    def _run_translation(
        self,
        goal_handle,
        primitive_type,
        start_pose,
        target_pose,
        target_heading,
        position_tolerance_m,
        timeout_sec,
        limits,
        status,
    ):
        start_time = time.monotonic()
        period_sec = 1.0 / self.control_rate_hz

        while rclpy.ok():
            interruption = self._check_interruption(
                goal_handle,
                start_time,
                timeout_sec,
                start_pose,
                target_pose=target_pose,
                target_heading=target_heading,
            )
            if interruption is not None:
                return interruption

            current_pose = self._get_current_pose_stamped()
            if current_pose is None:
                return self._finish_with_current_values(
                    goal_handle,
                    'abort',
                    ControlStatus.STATE_FAILED,
                    ExecuteMotionPrimitive.Result.RESULT_ODOM_UNAVAILABLE,
                    False,
                    'Odometry is unavailable.',
                    start_pose,
                    target_pose=target_pose,
                    target_heading=target_heading,
                )

            remaining_distance = planar_distance(current_pose, target_pose)
            distance_traveled = planar_distance(start_pose, current_pose)
            twist, heading_error = self._translation_twist(
                primitive_type,
                current_pose,
                target_pose,
                target_heading,
            )
            self._update_execution_status(
                primitive_type,
                remaining_distance,
                distance_traveled,
                heading_error,
                status,
            )
            self._publish_feedback(
                goal_handle,
                current_pose,
                distance_traveled,
                remaining_distance,
                heading_error,
                status,
            )

            if remaining_distance <= position_tolerance_m:
                break

            self._publish_twist(twist, limits)
            time.sleep(period_sec)

        return self._finish_successful_translation(
            goal_handle,
            start_pose,
            target_pose,
            target_heading,
        )

    def _run_rotation(
        self,
        goal_handle,
        start_pose,
        target_heading,
        heading_tolerance_rad,
        timeout_sec,
        limits,
        status,
    ):
        start_time = time.monotonic()
        period_sec = 1.0 / self.control_rate_hz

        while rclpy.ok():
            interruption = self._check_interruption(
                goal_handle,
                start_time,
                timeout_sec,
                start_pose,
                target_heading=target_heading,
            )
            if interruption is not None:
                return interruption

            current_pose = self._get_current_pose_stamped()
            if current_pose is None:
                return self._finish_with_current_values(
                    goal_handle,
                    'abort',
                    ControlStatus.STATE_FAILED,
                    ExecuteMotionPrimitive.Result.RESULT_ODOM_UNAVAILABLE,
                    False,
                    'Odometry is unavailable.',
                    start_pose,
                    target_heading=target_heading,
                )

            current_yaw = yaw_from_quaternion(current_pose.pose.orientation)
            heading_error = normalize_angle(target_heading - current_yaw)
            distance_traveled = planar_distance(start_pose, current_pose)
            self._update_execution_status(
                ExecuteMotionPrimitive.Goal.ROTATE_RELATIVE,
                0.0,
                distance_traveled,
                heading_error,
                status,
            )
            self._publish_feedback(
                goal_handle,
                current_pose,
                distance_traveled,
                0.0,
                heading_error,
                status,
            )

            if abs(heading_error) <= heading_tolerance_rad:
                break

            twist = Twist()
            twist.angular.z = self.k_heading * heading_error
            self._publish_twist(twist, limits)
            time.sleep(period_sec)

        return self._finish_successful_rotation(goal_handle, start_pose, target_heading)

    def _check_interruption(
        self,
        goal_handle,
        start_time,
        timeout_sec,
        start_pose,
        target_pose=None,
        target_heading=None,
    ):
        if goal_handle.is_cancel_requested:
            return self._finish_with_current_values(
                goal_handle,
                'cancel',
                ControlStatus.STATE_CANCELED,
                ExecuteMotionPrimitive.Result.RESULT_CANCELED,
                False,
                'Action cancel requested.',
                start_pose,
                target_pose=target_pose,
                target_heading=target_heading,
            )

        with self._state_lock:
            stop_requested = self._stop_requested
        if stop_requested:
            return self._finish_with_current_values(
                goal_handle,
                'abort',
                ControlStatus.STATE_CANCELED,
                ExecuteMotionPrimitive.Result.RESULT_CANCELED,
                False,
                'Stop requested.',
                start_pose,
                target_pose=target_pose,
                target_heading=target_heading,
            )

        if time.monotonic() - start_time > timeout_sec:
            return self._finish_with_current_values(
                goal_handle,
                'abort',
                ControlStatus.STATE_FAILED,
                ExecuteMotionPrimitive.Result.RESULT_TIMEOUT,
                False,
                'Motion primitive timed out.',
                start_pose,
                target_pose=target_pose,
                target_heading=target_heading,
            )

        if not self._is_odom_fresh():
            return self._finish_with_current_values(
                goal_handle,
                'abort',
                ControlStatus.STATE_FAILED,
                ExecuteMotionPrimitive.Result.RESULT_ODOM_UNAVAILABLE,
                False,
                'Odometry became stale during execution.',
                start_pose,
                target_pose=target_pose,
                target_heading=target_heading,
            )

        return None

    def _check_primitive_interruption(self, *args, **kwargs):
        return self._check_interruption(*args, **kwargs)

    @staticmethod
    def _translation_primitive_types():
        return (
            ExecuteMotionPrimitive.Goal.DRIVE_FORWARD,
            ExecuteMotionPrimitive.Goal.DRIVE_BACKWARD,
            ExecuteMotionPrimitive.Goal.STRAFE_LEFT,
            ExecuteMotionPrimitive.Goal.STRAFE_RIGHT,
        )

    def _translation_target_pose(self, primitive_type, start_pose, distance_m):
        start_yaw = yaw_from_quaternion(start_pose.pose.orientation)
        direction_yaw = start_yaw
        if primitive_type == ExecuteMotionPrimitive.Goal.DRIVE_BACKWARD:
            direction_yaw = start_yaw + math.pi
        elif primitive_type == ExecuteMotionPrimitive.Goal.STRAFE_LEFT:
            direction_yaw = start_yaw + math.pi / 2.0
        elif primitive_type == ExecuteMotionPrimitive.Goal.STRAFE_RIGHT:
            direction_yaw = start_yaw - math.pi / 2.0

        target_pose = deepcopy(start_pose)
        target_pose.pose.position.x = (
            start_pose.pose.position.x + distance_m * math.cos(direction_yaw)
        )
        target_pose.pose.position.y = (
            start_pose.pose.position.y + distance_m * math.sin(direction_yaw)
        )
        return target_pose

    def _drive_primitive_target_pose(self, primitive_type, start_pose, distance_m):
        return self._translation_target_pose(primitive_type, start_pose, distance_m)

    @staticmethod
    def _rotation_target_heading(current_yaw, value):
        return normalize_angle(current_yaw + value)

    @staticmethod
    def _rotation_primitive_target_heading(primitive_type, current_yaw, value):
        del primitive_type
        return DarthMaulControlNode._rotation_target_heading(current_yaw, value)

    @staticmethod
    def _primitive_status_text(primitive_type):
        if primitive_type == ExecuteMotionPrimitive.Goal.DRIVE_FORWARD:
            return 'driving forward'
        if primitive_type == ExecuteMotionPrimitive.Goal.DRIVE_BACKWARD:
            return 'driving backward'
        if primitive_type == ExecuteMotionPrimitive.Goal.STRAFE_LEFT:
            return 'strafing left'
        if primitive_type == ExecuteMotionPrimitive.Goal.STRAFE_RIGHT:
            return 'strafing right'
        if primitive_type == ExecuteMotionPrimitive.Goal.ROTATE_RELATIVE:
            return 'rotating relative'
        return 'unknown primitive'

    def _translation_twist(
        self,
        primitive_type,
        current_pose,
        target_pose,
        target_heading,
    ):
        dx = target_pose.pose.position.x - current_pose.pose.position.x
        dy = target_pose.pose.position.y - current_pose.pose.position.y
        yaw = yaw_from_quaternion(current_pose.pose.orientation)
        body_x_error = math.cos(yaw) * dx + math.sin(yaw) * dy
        body_y_error = -math.sin(yaw) * dx + math.cos(yaw) * dy

        twist = Twist()
        if primitive_type in (
            ExecuteMotionPrimitive.Goal.DRIVE_FORWARD,
            ExecuteMotionPrimitive.Goal.DRIVE_BACKWARD,
        ):
            twist.linear.x = self.k_position * body_x_error
        else:
            twist.linear.y = self.k_position * body_y_error

        heading_error = normalize_angle(target_heading - yaw)
        twist.angular.z = self.k_heading * heading_error
        return twist, heading_error

    def _compute_drive_primitive_twist(self, current_pose, target_pose, target_heading):
        return self._translation_twist(
            ExecuteMotionPrimitive.Goal.DRIVE_FORWARD,
            current_pose,
            target_pose,
            target_heading,
        )

    def _velocity_limits_for_primitive(self, goal):
        sanitized = self._limiter.sanitize_goal_limits(
            goal.max_linear_x_mps,
            goal.max_linear_y_mps,
            goal.max_angular_z_radps,
        )

        if goal.primitive_type in (
            ExecuteMotionPrimitive.Goal.DRIVE_FORWARD,
            ExecuteMotionPrimitive.Goal.DRIVE_BACKWARD,
        ):
            return VelocityLimits(
                sanitized.max_linear_x_mps,
                0.0,
                sanitized.max_angular_z_radps,
            )
        if goal.primitive_type in (
            ExecuteMotionPrimitive.Goal.STRAFE_LEFT,
            ExecuteMotionPrimitive.Goal.STRAFE_RIGHT,
        ):
            return VelocityLimits(
                0.0,
                sanitized.max_linear_y_mps,
                sanitized.max_angular_z_radps,
            )
        return VelocityLimits(0.0, 0.0, sanitized.max_angular_z_radps)

    def _primitive_velocity_limits(self, goal):
        return self._velocity_limits_for_primitive(goal)

    def _start_execution_status(
        self,
        primitive_type,
        status,
        distance_remaining,
        distance_traveled,
        heading_error,
    ):
        with self._state_lock:
            self._state = ControlStatus.STATE_EXECUTING
            self._status = status
            self._command_enabled = True
            self._stop_requested = False
            self._active_primitive_type = primitive_type
            self._distance_remaining_m = distance_remaining
            self._distance_traveled_m = distance_traveled
            self._heading_error_rad = heading_error
        self._publish_status()

    def _start_primitive_status(self, status, distance, heading_error):
        self._start_execution_status(0, status, distance, 0.0, heading_error)

    def _update_execution_status(
        self,
        primitive_type,
        distance_remaining,
        distance_traveled,
        heading_error,
        status,
    ):
        with self._state_lock:
            self._active_primitive_type = primitive_type
            self._distance_remaining_m = distance_remaining
            self._distance_traveled_m = distance_traveled
            self._heading_error_rad = heading_error
            self._status = status
        self._publish_status()

    def _publish_feedback(
        self,
        goal_handle,
        current_pose,
        distance_traveled,
        remaining_distance,
        heading_error,
        status,
    ):
        feedback = ExecuteMotionPrimitive.Feedback()
        feedback.current_pose = current_pose
        feedback.distance_traveled_m = distance_traveled
        feedback.remaining_distance_m = remaining_distance
        feedback.heading_error_rad = heading_error
        feedback.status = status
        goal_handle.publish_feedback(feedback)

    def _publish_primitive_feedback(self, *args, **kwargs):
        return self._publish_feedback(*args, **kwargs)

    def _finish_successful_translation(
        self,
        goal_handle,
        start_pose,
        target_pose,
        target_heading,
    ):
        final_pose, distance_traveled, position_error, heading_error = (
            self._result_values(
                start_pose,
                target_pose=target_pose,
                target_heading=target_heading,
            )
        )
        return self._finish_action(
            goal_handle,
            'succeed',
            ControlStatus.STATE_SUCCEEDED,
            ExecuteMotionPrimitive.Result.RESULT_SUCCESS,
            True,
            'Motion primitive succeeded.',
            start_pose,
            final_pose,
            distance_traveled,
            position_error,
            heading_error,
        )

    def _finish_successful_drive_primitive(
        self,
        goal_handle,
        start_pose,
        target_pose,
        target_heading,
    ):
        return self._finish_successful_translation(
            goal_handle,
            start_pose,
            target_pose,
            target_heading,
        )

    def _finish_successful_rotation(self, goal_handle, start_pose, target_heading):
        final_pose, distance_traveled, position_error, heading_error = (
            self._result_values(start_pose, target_heading=target_heading)
        )
        return self._finish_action(
            goal_handle,
            'succeed',
            ControlStatus.STATE_SUCCEEDED,
            ExecuteMotionPrimitive.Result.RESULT_SUCCESS,
            True,
            'Motion primitive succeeded.',
            start_pose,
            final_pose,
            distance_traveled,
            position_error,
            heading_error,
        )

    def _finish_successful_rotate_primitive(
        self,
        goal_handle,
        start_pose,
        target_heading,
    ):
        return self._finish_successful_rotation(goal_handle, start_pose, target_heading)

    def _finish_with_current_values(
        self,
        goal_handle,
        terminal_transition,
        state,
        result_code,
        success,
        message,
        start_pose,
        target_pose=None,
        target_heading=None,
    ):
        final_pose, distance_traveled, position_error, heading_error = (
            self._result_values(
                start_pose,
                target_pose=target_pose,
                target_heading=target_heading,
            )
        )
        return self._finish_action(
            goal_handle,
            terminal_transition,
            state,
            result_code,
            success,
            message,
            start_pose,
            final_pose,
            distance_traveled,
            position_error,
            heading_error,
        )

    def _finish_primitive_with_current_values(self, *args, **kwargs):
        return self._finish_with_current_values(*args, **kwargs)

    def _result_values(self, start_pose, target_pose=None, target_heading=None):
        final_pose = self._get_current_pose_stamped()
        if start_pose is None or final_pose is None:
            distance_traveled = self._last_distance_traveled()
        else:
            distance_traveled = planar_distance(start_pose, final_pose)

        position_error = 0.0
        if target_pose is not None:
            position_error = (
                planar_distance(final_pose, target_pose)
                if final_pose is not None
                else self._last_distance_remaining()
            )

        heading_error = 0.0
        if target_heading is not None:
            heading_error = (
                normalize_angle(
                    target_heading - yaw_from_quaternion(final_pose.pose.orientation)
                )
                if final_pose is not None
                else self._last_heading_error()
            )

        return final_pose, distance_traveled, position_error, heading_error

    def _primitive_result_values(self, *args, **kwargs):
        return self._result_values(*args, **kwargs)

    def _finish_action(
        self,
        goal_handle,
        terminal_transition,
        state,
        result_code,
        success,
        message,
        start_pose=None,
        final_pose=None,
        distance_traveled=0.0,
        final_position_error=0.0,
        final_heading_error=0.0,
        publish_zero=True,
    ):
        if publish_zero:
            self.publish_zero_twist()
        if final_pose is None:
            final_pose = self._get_current_pose_stamped()
        with self._state_lock:
            self._state = state
            self._status = message
            self._command_enabled = True
            self._active_primitive_type = 0
            self._distance_remaining_m = final_position_error
            self._distance_traveled_m = distance_traveled
            self._heading_error_rad = final_heading_error
        self._publish_status()

        if terminal_transition == 'succeed':
            goal_handle.succeed()
        elif terminal_transition == 'cancel':
            goal_handle.canceled()
        else:
            goal_handle.abort()

        result = ExecuteMotionPrimitive.Result()
        result.success = success
        result.result_code = result_code
        result.message = message
        result.start_pose = start_pose or PoseStamped()
        result.final_pose = final_pose or PoseStamped()
        result.distance_traveled_m = distance_traveled
        result.final_position_error_m = final_position_error
        result.final_heading_error_rad = final_heading_error
        return result

    def _finish_primitive_action(self, *args, **kwargs):
        return self._finish_action(*args, **kwargs)

    def publish_zero_twist(self):
        try:
            if not rclpy.ok():
                return False
            self._publish_twist(self._limiter.zero_twist())
        except Exception as exc:
            try:
                self.get_logger().warn(f'Failed to publish zero Twist: {exc}')
            except Exception:
                pass
            return False
        return True

    def _publish_twist(self, twist, limits=None):
        publish_twist = twist
        if self._has_nonzero_motion(twist):
            with self._state_lock:
                command_enabled = self._command_enabled
            if not command_enabled:
                publish_twist = self._limiter.zero_twist()
        self._cmd_vel_pub.publish(self._limiter.clamp(publish_twist, limits))

    @staticmethod
    def _has_nonzero_motion(twist):
        return any((
            abs(twist.linear.x) > 0.0,
            abs(twist.linear.y) > 0.0,
            abs(twist.linear.z) > 0.0,
            abs(twist.angular.x) > 0.0,
            abs(twist.angular.y) > 0.0,
            abs(twist.angular.z) > 0.0,
        ))

    def _publish_status(self):
        with self._state_lock:
            state = self._state
            status = self._status
            command_enabled = self._command_enabled
            active_primitive_type = self._active_primitive_type
            distance_remaining = self._distance_remaining_m
            distance_traveled = self._distance_traveled_m
            heading_error = self._heading_error_rad

        msg = ControlStatus()
        msg.stamp = self.get_clock().now().to_msg()
        msg.state = state
        msg.status = status
        msg.odom_available = self._is_odom_fresh()
        msg.command_enabled = command_enabled
        msg.active_primitive_type = active_primitive_type
        msg.distance_remaining_m = distance_remaining
        msg.distance_traveled_m = distance_traveled
        msg.heading_error_rad = heading_error
        self._status_pub.publish(msg)

    def _is_odom_fresh(self):
        with self._odom_lock:
            if self._last_odom_monotonic is None:
                return False
            return time.monotonic() - self._last_odom_monotonic <= self.odom_timeout_sec

    def _get_current_odom(self):
        with self._odom_lock:
            if self._current_odom is None:
                return None
            return deepcopy(self._current_odom)

    def _get_current_pose_stamped(self):
        odom = self._get_current_odom()
        if odom is None:
            return None
        pose = PoseStamped()
        pose.header = odom.header
        pose.pose = odom.pose.pose
        return pose

    def _last_distance_remaining(self):
        with self._state_lock:
            return self._distance_remaining_m

    def _last_distance_traveled(self):
        with self._state_lock:
            return self._distance_traveled_m

    def _last_distance(self):
        return self._last_distance_remaining()

    def _last_heading_error(self):
        with self._state_lock:
            return self._heading_error_rad

    def destroy_node(self):
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
