from copy import deepcopy
import math
import threading
import time

from darth_maul_control.geometry import (
    is_finite_pose_stamped,
    normalize_angle,
    planar_distance,
    yaw_from_quaternion,
)
from darth_maul_control.velocity_limiter import VelocityLimiter
from darth_maul_control_interfaces.action import FollowWaypoints
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
        self._active_waypoint_index = 0
        self._waypoint_count = 0
        self._distance_to_active_waypoint_m = 0.0
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
            FollowWaypoints,
            '/darth_maul_control/follow_waypoints',
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
            'This initial controller depends on odometry quality; validate forward, '
            'backward, strafe, rotation, and odom response before trusting pose control.'
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
        with self._state_lock:
            if self._active_goal:
                self.get_logger().warn('Rejecting follow_waypoints goal; another goal is active.')
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
                self._active_waypoint_index = 0
                self._waypoint_count = 0
                self._distance_to_active_waypoint_m = 0.0
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
        try:
            valid, message, goal_frame = self._validate_goal(goal)
            if not valid:
                return self._finish_action(
                    goal_handle,
                    'abort',
                    ControlStatus.STATE_FAILED,
                    FollowWaypoints.Result.RESULT_INVALID_GOAL,
                    False,
                    message,
                    0.0,
                    0.0,
                )

            if not self._is_odom_fresh():
                return self._finish_action(
                    goal_handle,
                    'abort',
                    ControlStatus.STATE_FAILED,
                    FollowWaypoints.Result.RESULT_ODOM_UNAVAILABLE,
                    False,
                    'Odometry is unavailable or stale.',
                    0.0,
                    0.0,
                )

            frame_ok, frame_message = self._goal_frame_matches_odom(goal_frame)
            if not frame_ok:
                return self._finish_action(
                    goal_handle,
                    'abort',
                    ControlStatus.STATE_FAILED,
                    FollowWaypoints.Result.RESULT_INVALID_GOAL,
                    False,
                    frame_message,
                    0.0,
                    0.0,
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
            limits = self._limiter.sanitize_goal_limits(
                goal.max_linear_x_mps,
                goal.max_linear_y_mps,
                goal.max_angular_z_radps,
            )

            with self._state_lock:
                self._state = ControlStatus.STATE_EXECUTING
                self._status = 'executing follow_waypoints'
                self._command_enabled = True
                self._stop_requested = False
                self._active_waypoint_index = 0
                self._waypoint_count = len(goal.waypoints)
                self._distance_to_active_waypoint_m = 0.0
                self._heading_error_rad = 0.0
            self._publish_status()

            return self._run_goal(
                goal_handle,
                goal,
                position_tolerance_m,
                heading_tolerance_rad,
                timeout_sec,
                limits,
            )
        except Exception as exc:
            self.get_logger().error(f'Control loop failed: {exc}')
            return self._finish_action(
                goal_handle,
                'abort',
                ControlStatus.STATE_FAILED,
                FollowWaypoints.Result.RESULT_CONTROL_FAILED,
                False,
                f'Control loop failed: {exc}',
                self._last_distance(),
                self._last_heading_error(),
            )
        finally:
            self.publish_zero_twist()
            with self._state_lock:
                self._active_goal = False
                self._command_enabled = True
                self._stop_requested = False
            self._publish_status()

    def _run_goal(
        self,
        goal_handle,
        goal,
        position_tolerance_m,
        heading_tolerance_rad,
        timeout_sec,
        limits,
    ):
        start_time = time.monotonic()
        period_sec = 1.0 / self.control_rate_hz
        waypoint_count = len(goal.waypoints)

        for waypoint_index, waypoint in enumerate(goal.waypoints):
            while rclpy.ok():
                interruption = self._check_interruption(goal_handle, start_time, timeout_sec)
                if interruption is not None:
                    return interruption

                current_pose = self._get_current_pose_stamped()
                if current_pose is None:
                    return self._finish_action(
                        goal_handle,
                        'abort',
                        ControlStatus.STATE_FAILED,
                        FollowWaypoints.Result.RESULT_ODOM_UNAVAILABLE,
                        False,
                        'Odometry is unavailable.',
                        self._last_distance(),
                        self._last_heading_error(),
                    )

                distance = planar_distance(current_pose, waypoint)
                heading_error = self._feedback_heading_error(goal, current_pose)
                self._update_execution_status(
                    waypoint_index,
                    waypoint_count,
                    distance,
                    heading_error,
                    'moving to waypoint',
                )
                self._publish_feedback(
                    goal_handle,
                    waypoint_index,
                    waypoint_count,
                    current_pose,
                    distance,
                    heading_error,
                    'moving to waypoint',
                )

                if distance <= position_tolerance_m:
                    self.publish_zero_twist()
                    break

                self._publish_twist(
                    self._compute_position_twist(current_pose, waypoint),
                    limits,
                )
                time.sleep(period_sec)

        if goal.use_final_heading:
            while rclpy.ok():
                interruption = self._check_interruption(goal_handle, start_time, timeout_sec)
                if interruption is not None:
                    return interruption

                current_pose = self._get_current_pose_stamped()
                if current_pose is None:
                    return self._finish_action(
                        goal_handle,
                        'abort',
                        ControlStatus.STATE_FAILED,
                        FollowWaypoints.Result.RESULT_ODOM_UNAVAILABLE,
                        False,
                        'Odometry is unavailable.',
                        self._last_distance(),
                        self._last_heading_error(),
                    )

                heading_error = normalize_angle(
                    goal.final_heading_rad - yaw_from_quaternion(current_pose.pose.orientation)
                )
                distance = planar_distance(current_pose, goal.waypoints[-1])
                active_index = waypoint_count - 1
                self._update_execution_status(
                    active_index,
                    waypoint_count,
                    distance,
                    heading_error,
                    'aligning final heading',
                )
                self._publish_feedback(
                    goal_handle,
                    active_index,
                    waypoint_count,
                    current_pose,
                    distance,
                    heading_error,
                    'aligning final heading',
                )

                if abs(heading_error) <= heading_tolerance_rad:
                    self.publish_zero_twist()
                    break

                twist = Twist()
                twist.angular.z = self.k_heading * heading_error
                self._publish_twist(twist, limits)
                time.sleep(period_sec)

        final_pose = self._get_current_pose_stamped()
        final_position_error = 0.0
        final_heading_error = 0.0
        if final_pose is not None:
            final_position_error = planar_distance(final_pose, goal.waypoints[-1])
            if goal.use_final_heading:
                final_heading_error = normalize_angle(
                    goal.final_heading_rad - yaw_from_quaternion(final_pose.pose.orientation)
                )

        with self._state_lock:
            self._active_waypoint_index = waypoint_count
            self._distance_to_active_waypoint_m = final_position_error
            self._heading_error_rad = final_heading_error

        return self._finish_action(
            goal_handle,
            'succeed',
            ControlStatus.STATE_SUCCEEDED,
            FollowWaypoints.Result.RESULT_SUCCESS,
            True,
            'Waypoint goal succeeded.',
            final_position_error,
            final_heading_error,
        )

    def _check_interruption(self, goal_handle, start_time, timeout_sec):
        if goal_handle.is_cancel_requested:
            return self._finish_action(
                goal_handle,
                'cancel',
                ControlStatus.STATE_CANCELED,
                FollowWaypoints.Result.RESULT_CANCELED,
                False,
                'Action cancel requested.',
                self._last_distance(),
                self._last_heading_error(),
            )

        with self._state_lock:
            stop_requested = self._stop_requested
        if stop_requested:
            return self._finish_action(
                goal_handle,
                'abort',
                ControlStatus.STATE_CANCELED,
                FollowWaypoints.Result.RESULT_CANCELED,
                False,
                'Stop requested.',
                self._last_distance(),
                self._last_heading_error(),
            )

        if time.monotonic() - start_time > timeout_sec:
            return self._finish_action(
                goal_handle,
                'abort',
                ControlStatus.STATE_FAILED,
                FollowWaypoints.Result.RESULT_TIMEOUT,
                False,
                'Waypoint goal timed out.',
                self._last_distance(),
                self._last_heading_error(),
            )

        if not self._is_odom_fresh():
            return self._finish_action(
                goal_handle,
                'abort',
                ControlStatus.STATE_FAILED,
                FollowWaypoints.Result.RESULT_ODOM_UNAVAILABLE,
                False,
                'Odometry became stale during execution.',
                self._last_distance(),
                self._last_heading_error(),
            )

        return None

    def _validate_goal(self, goal):
        if not goal.waypoints:
            return False, 'Goal must contain at least one waypoint.', ''

        frame_id = goal.waypoints[0].header.frame_id.strip()
        if not frame_id:
            return False, 'Waypoint frame_id must not be empty.', ''

        for index, waypoint in enumerate(goal.waypoints):
            if waypoint.header.frame_id.strip() != frame_id:
                return False, 'All waypoint frame_id values must match.', ''
            if not is_finite_pose_stamped(waypoint):
                return False, f'Waypoint {index} contains NaN or infinite pose values.', ''

        if (
            not math.isfinite(goal.position_tolerance_m)
            or goal.position_tolerance_m < 0.0
        ):
            return False, 'position_tolerance_m must be finite and nonnegative.', ''
        if (
            not math.isfinite(goal.heading_tolerance_rad)
            or goal.heading_tolerance_rad < 0.0
        ):
            return False, 'heading_tolerance_rad must be finite and nonnegative.', ''
        if goal.timeout.sec < 0 or goal.timeout.nanosec >= 1000000000:
            return False, 'timeout must be nonnegative.', ''
        if goal.use_final_heading and not math.isfinite(goal.final_heading_rad):
            return False, 'final_heading_rad must be finite when final heading is enabled.', ''

        goal_limits = (
            goal.max_linear_x_mps,
            goal.max_linear_y_mps,
            goal.max_angular_z_radps,
        )
        if not all(math.isfinite(value) for value in goal_limits):
            return False, 'Velocity limits must be finite.', ''

        return True, '', frame_id

    def _goal_frame_matches_odom(self, goal_frame):
        odom = self._get_current_odom()
        if odom is None:
            return False, 'Odometry is unavailable.'
        odom_frame = odom.header.frame_id.strip()
        if not odom_frame:
            return False, 'Odometry frame_id is empty.'
        if odom_frame != goal_frame:
            return (
                False,
                f'Waypoint frame {goal_frame!r} does not match odometry frame {odom_frame!r}.',
            )
        return True, ''

    def _goal_timeout_sec(self, timeout_msg):
        if timeout_msg.sec == 0 and timeout_msg.nanosec == 0:
            return self.default_timeout_sec
        return self._duration_to_seconds(timeout_msg)

    @staticmethod
    def _duration_to_seconds(duration_msg):
        return float(duration_msg.sec) + float(duration_msg.nanosec) * 1e-9

    def _compute_position_twist(self, current_pose, waypoint):
        dx = waypoint.pose.position.x - current_pose.pose.position.x
        dy = waypoint.pose.position.y - current_pose.pose.position.y
        vx_world = self.k_position * dx
        vy_world = self.k_position * dy
        yaw = yaw_from_quaternion(current_pose.pose.orientation)

        twist = Twist()
        twist.linear.x = math.cos(yaw) * vx_world + math.sin(yaw) * vy_world
        twist.linear.y = -math.sin(yaw) * vx_world + math.cos(yaw) * vy_world
        twist.angular.z = 0.0
        return twist

    def _feedback_heading_error(self, goal, current_pose):
        if goal.use_final_heading:
            return normalize_angle(
                goal.final_heading_rad - yaw_from_quaternion(current_pose.pose.orientation)
            )
        return 0.0

    def _publish_feedback(
        self,
        goal_handle,
        active_waypoint_index,
        waypoint_count,
        current_pose,
        distance,
        heading_error,
        status,
    ):
        feedback = FollowWaypoints.Feedback()
        feedback.active_waypoint_index = active_waypoint_index
        feedback.waypoint_count = waypoint_count
        feedback.current_pose = current_pose
        feedback.distance_to_active_waypoint_m = distance
        feedback.heading_error_rad = heading_error
        feedback.status = status
        goal_handle.publish_feedback(feedback)

    def _update_execution_status(
        self,
        active_waypoint_index,
        waypoint_count,
        distance,
        heading_error,
        status,
    ):
        with self._state_lock:
            self._active_waypoint_index = active_waypoint_index
            self._waypoint_count = waypoint_count
            self._distance_to_active_waypoint_m = distance
            self._heading_error_rad = heading_error
            self._status = status
        self._publish_status()

    def _finish_action(
        self,
        goal_handle,
        terminal_transition,
        state,
        result_code,
        success,
        message,
        final_position_error,
        final_heading_error,
    ):
        self.publish_zero_twist()
        with self._state_lock:
            self._state = state
            self._status = message
            self._command_enabled = True
            self._distance_to_active_waypoint_m = final_position_error
            self._heading_error_rad = final_heading_error
        self._publish_status()

        if terminal_transition == 'succeed':
            goal_handle.succeed()
        elif terminal_transition == 'cancel':
            goal_handle.canceled()
        else:
            goal_handle.abort()

        result = FollowWaypoints.Result()
        result.success = success
        result.result_code = result_code
        result.message = message
        result.final_pose = self._get_current_pose_stamped() or PoseStamped()
        result.final_position_error_m = final_position_error
        result.final_heading_error_rad = final_heading_error
        return result

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
            active_waypoint_index = self._active_waypoint_index
            waypoint_count = self._waypoint_count
            distance = self._distance_to_active_waypoint_m
            heading_error = self._heading_error_rad

        msg = ControlStatus()
        msg.stamp = self.get_clock().now().to_msg()
        msg.state = state
        msg.status = status
        msg.odom_available = self._is_odom_fresh()
        msg.command_enabled = command_enabled
        msg.active_waypoint_index = active_waypoint_index
        msg.waypoint_count = waypoint_count
        msg.distance_to_active_waypoint_m = distance
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

    def _last_distance(self):
        with self._state_lock:
            return self._distance_to_active_waypoint_m

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
