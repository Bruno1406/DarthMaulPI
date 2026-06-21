import math
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Twist
from nav2_msgs.action import BackUp, DriveOnHeading, Spin
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from darth_maul_control_interfaces.action import ExecuteMotionPrimitive


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


class MotionPrimitiveServer(Node):
    SUCCESS = 0
    REJECTED_INVALID_GOAL = 1
    NAV2_SERVER_UNAVAILABLE = 2
    NAV2_GOAL_REJECTED = 3
    NAV2_FAILED = 4
    TIMEOUT = 5
    OBSTACLE_TOO_CLOSE = 6
    FINAL_ERROR_TOO_LARGE = 7
    CANCELED = 8
    INTERNAL_ERROR = 9

    def __init__(self):
        super().__init__("motion_primitive_server")

        self.cb_group = ReentrantCallbackGroup()
        self.lock = threading.RLock()

        self._declare_parameters()

        self.cell_length_m = self.get_parameter("cell_length_m").value

        self.default_position_tolerance_m = self.get_parameter("default_position_tolerance_m").value
        self.default_heading_tolerance_rad = self.get_parameter("default_heading_tolerance_rad").value

        self.default_linear_speed_mps = self.get_parameter("default_linear_speed_mps").value
        self.default_reverse_speed_mps = self.get_parameter("default_reverse_speed_mps").value
        self.default_angular_speed_radps = self.get_parameter("default_angular_speed_radps").value

        self.max_linear_speed_mps = self.get_parameter("max_linear_speed_mps").value
        self.max_angular_speed_radps = self.get_parameter("max_angular_speed_radps").value

        self.timeout_margin_s = self.get_parameter("timeout_margin_s").value

        self.scan_topic = self.get_parameter("scan_topic").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.final_cmd_vel_topic = self.get_parameter("final_cmd_vel_topic").value

        self.front_sector_deg = self.get_parameter("front_sector_deg").value
        self.front_stop_distance_m = self.get_parameter("front_stop_distance_m").value

        self.require_scan_for_forward = self.get_parameter("require_scan_for_forward").value
        self.require_odom_for_result_check = self.get_parameter("require_odom_for_result_check").value

        self.nav2_drive_action = self.get_parameter("nav2_drive_action").value
        self.nav2_backup_action = self.get_parameter("nav2_backup_action").value
        self.nav2_spin_action = self.get_parameter("nav2_spin_action").value

        self.latest_scan: Optional[LaserScan] = None
        self.latest_scan_time = None
        self.latest_odom: Optional[Odometry] = None
        self.latest_odom_time = None

        self.active_nav_goal_handle = None

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self._scan_cb,
            qos_profile_sensor_data,
            callback_group=self.cb_group,
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self._odom_cb,
            10,
            callback_group=self.cb_group,
        )

        self.zero_pub = self.create_publisher(Twist, self.final_cmd_vel_topic, 10)

        self.drive_client = ActionClient(
            self,
            DriveOnHeading,
            self.nav2_drive_action,
            callback_group=self.cb_group,
        )
        self.backup_client = ActionClient(
            self,
            BackUp,
            self.nav2_backup_action,
            callback_group=self.cb_group,
        )
        self.spin_client = ActionClient(
            self,
            Spin,
            self.nav2_spin_action,
            callback_group=self.cb_group,
        )

        self.action_server = ActionServer(
            self,
            ExecuteMotionPrimitive,
            "/darth_maul_control/execute_motion_primitive",
            execute_callback=self.execute_cb,
            goal_callback=self.goal_cb,
            cancel_callback=self.cancel_cb,
            callback_group=self.cb_group,
        )

        self.get_logger().info("Motion primitive server ready on /darth_maul_control/execute_motion_primitive")

    def _declare_parameters(self):
        self.declare_parameter("cell_length_m", 0.254)

        self.declare_parameter("default_position_tolerance_m", 0.025)
        self.declare_parameter("default_heading_tolerance_rad", 0.05)

        self.declare_parameter("default_linear_speed_mps", 0.08)
        self.declare_parameter("default_reverse_speed_mps", 0.06)
        self.declare_parameter("default_angular_speed_radps", 0.35)

        self.declare_parameter("max_linear_speed_mps", 0.12)
        self.declare_parameter("max_angular_speed_radps", 0.50)

        self.declare_parameter("timeout_margin_s", 3.0)

        self.declare_parameter("scan_topic", "/ldlidar_node/scan")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("final_cmd_vel_topic", "/controller/cmd_vel")

        self.declare_parameter("front_sector_deg", 35.0)
        self.declare_parameter("front_stop_distance_m", 0.13)

        self.declare_parameter("require_scan_for_forward", True)
        self.declare_parameter("require_odom_for_result_check", False)

        self.declare_parameter("nav2_drive_action", "/drive_on_heading")
        self.declare_parameter("nav2_backup_action", "/backup")
        self.declare_parameter("nav2_spin_action", "/spin")

    def _scan_cb(self, msg: LaserScan):
        with self.lock:
            self.latest_scan = msg
            self.latest_scan_time = self.get_clock().now()

    def _odom_cb(self, msg: Odometry):
        with self.lock:
            self.latest_odom = msg
            self.latest_odom_time = self.get_clock().now()

    def goal_cb(self, goal_request):
        primitive = goal_request.primitive_type
        value = goal_request.value

        if primitive not in {
            ExecuteMotionPrimitive.Goal.DRIVE_FORWARD,
            ExecuteMotionPrimitive.Goal.DRIVE_BACKWARD,
            ExecuteMotionPrimitive.Goal.ROTATE_RELATIVE,
            ExecuteMotionPrimitive.Goal.STOP,
            ExecuteMotionPrimitive.Goal.WAIT,
            ExecuteMotionPrimitive.Goal.ADVANCE_CELL,
        }:
            self.get_logger().error(f"Rejecting unknown primitive_type={primitive}")
            return GoalResponse.REJECT

        if not math.isfinite(value):
            self.get_logger().error(f"Rejecting non-finite primitive value={value}")
            return GoalResponse.REJECT

        if primitive in {
            ExecuteMotionPrimitive.Goal.DRIVE_FORWARD,
            ExecuteMotionPrimitive.Goal.DRIVE_BACKWARD,
        } and abs(value) > 0.60:
            self.get_logger().error(f"Rejecting translation > 0.60 m: {value}")
            return GoalResponse.REJECT

        if primitive == ExecuteMotionPrimitive.Goal.ROTATE_RELATIVE and abs(value) > math.pi + 0.10:
            self.get_logger().error(f"Rejecting rotation > pi+0.10 rad: {value}")
            return GoalResponse.REJECT

        return GoalResponse.ACCEPT

    def cancel_cb(self, goal_handle):
        self.get_logger().warn("Cancel requested for motion primitive")
        self._cancel_active_nav_goal()
        self._publish_zero_for(0.30)
        return CancelResponse.ACCEPT

    def execute_cb(self, goal_handle):
        goal = goal_handle.request
        start_pose = self._current_pose()

        try:
            result = self._execute_goal(goal_handle, goal, start_pose)
        except Exception as exc:
            self.get_logger().exception(f"Internal error while executing primitive: {exc}")
            self._cancel_active_nav_goal()
            self._publish_zero_for(0.30)
            result = self._make_result(
                False,
                self.INTERNAL_ERROR,
                f"Internal error: {exc}",
            )

        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            self._publish_zero_for(0.30)
            return self._make_result(False, self.CANCELED, "Goal canceled by client")

        if result.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()

        self._publish_zero_for(0.30)
        return result

    def _execute_goal(self, goal_handle, goal, start_pose: Optional[Pose2D]):
        primitive = goal.primitive_type

        if primitive == ExecuteMotionPrimitive.Goal.STOP:
            self._cancel_active_nav_goal()
            self._publish_zero_for(0.30)
            return self._make_result(True, self.SUCCESS, "STOP succeeded")

        if primitive == ExecuteMotionPrimitive.Goal.WAIT:
            seconds = max(0.0, float(goal.value))
            return self._execute_wait(goal_handle, seconds)

        if primitive == ExecuteMotionPrimitive.Goal.ADVANCE_CELL:
            distance = float(self.cell_length_m)
            return self._execute_drive_forward(goal_handle, goal, distance, start_pose)

        if primitive == ExecuteMotionPrimitive.Goal.DRIVE_FORWARD:
            distance = abs(float(goal.value))
            return self._execute_drive_forward(goal_handle, goal, distance, start_pose)

        if primitive == ExecuteMotionPrimitive.Goal.DRIVE_BACKWARD:
            distance = abs(float(goal.value))
            return self._execute_drive_backward(goal_handle, goal, distance, start_pose)

        if primitive == ExecuteMotionPrimitive.Goal.ROTATE_RELATIVE:
            angle = float(goal.value)
            return self._execute_rotate(goal_handle, goal, angle, start_pose)

        return self._make_result(False, self.REJECTED_INVALID_GOAL, f"Unknown primitive_type={primitive}")

    def _execute_drive_forward(self, goal_handle, goal, distance: float, start_pose: Optional[Pose2D]):
        if distance <= 0.0:
            return self._make_result(False, self.REJECTED_INVALID_GOAL, "DRIVE_FORWARD distance must be > 0")

        if goal.collision_check_enabled:
            ok, message = self._forward_precheck()
            if not ok:
                return self._make_result(False, self.OBSTACLE_TOO_CLOSE, message)

        speed = self._resolved_linear_speed(goal.max_linear_x_mps)
        timeout_s = self._resolved_timeout(goal.timeout_s, distance / speed)
        feedback_name = "DRIVE_FORWARD"

        nav_goal = DriveOnHeading.Goal()
        nav_goal.target = Point(x=float(distance), y=0.0, z=0.0)
        nav_goal.speed = float(speed)
        nav_goal.time_allowance = duration_from_seconds(timeout_s)
        if hasattr(nav_goal, "disable_collision_checks"):
            nav_goal.disable_collision_checks = True

        nav_result = self._send_nav_goal_with_monitoring(
            goal_handle=goal_handle,
            client=self.drive_client,
            nav_goal=nav_goal,
            timeout_s=timeout_s,
            feedback_name=feedback_name,
            monitor_front=goal.collision_check_enabled,
            target_distance=distance,
            target_angle=0.0,
        )

        if not nav_result.success:
            return nav_result

        return self._check_final_translation(goal, start_pose, distance, "DRIVE_FORWARD")

    def _execute_drive_backward(self, goal_handle, goal, distance: float, start_pose: Optional[Pose2D]):
        if distance <= 0.0:
            return self._make_result(False, self.REJECTED_INVALID_GOAL, "DRIVE_BACKWARD distance must be > 0")

        speed = self._resolved_reverse_speed(goal.max_linear_x_mps)
        timeout_s = self._resolved_timeout(goal.timeout_s, distance / speed)

        nav_goal = BackUp.Goal()
        nav_goal.target = Point(x=-float(distance), y=0.0, z=0.0)
        nav_goal.speed = float(speed)
        nav_goal.time_allowance = duration_from_seconds(timeout_s)
        if hasattr(nav_goal, "disable_collision_checks"):
            nav_goal.disable_collision_checks = True

        nav_result = self._send_nav_goal_with_monitoring(
            goal_handle=goal_handle,
            client=self.backup_client,
            nav_goal=nav_goal,
            timeout_s=timeout_s,
            feedback_name="DRIVE_BACKWARD",
            monitor_front=False,
            target_distance=distance,
            target_angle=0.0,
        )

        if not nav_result.success:
            return nav_result

        return self._check_final_translation(goal, start_pose, distance, "DRIVE_BACKWARD")

    def _execute_rotate(self, goal_handle, goal, angle: float, start_pose: Optional[Pose2D]):
        if abs(angle) <= 1.0e-4:
            return self._make_result(True, self.SUCCESS, "ROTATE_RELATIVE skipped, angle approximately zero")

        speed = self._resolved_angular_speed(goal.max_angular_z_radps)
        timeout_s = self._resolved_timeout(goal.timeout_s, abs(angle) / speed)

        nav_goal = Spin.Goal()
        nav_goal.target_yaw = float(angle)
        nav_goal.time_allowance = duration_from_seconds(timeout_s)
        if hasattr(nav_goal, "disable_collision_checks"):
            nav_goal.disable_collision_checks = True

        nav_result = self._send_nav_goal_with_monitoring(
            goal_handle=goal_handle,
            client=self.spin_client,
            nav_goal=nav_goal,
            timeout_s=timeout_s,
            feedback_name="ROTATE_RELATIVE",
            monitor_front=False,
            target_distance=0.0,
            target_angle=angle,
        )

        if not nav_result.success:
            return nav_result

        return self._check_final_rotation(goal, start_pose, angle)

    def _execute_wait(self, goal_handle, seconds: float):
        start = time.monotonic()
        while time.monotonic() - start < seconds:
            if goal_handle.is_cancel_requested:
                self._cancel_active_nav_goal()
                self._publish_zero_for(0.30)
                return self._make_result(False, self.CANCELED, "WAIT canceled by client")

            elapsed = time.monotonic() - start
            feedback = ExecuteMotionPrimitive.Feedback()
            feedback.progress = clamp(elapsed / max(seconds, 1.0e-6), 0.0, 1.0)
            feedback.distance_remaining_m = 0.0
            feedback.heading_remaining_rad = 0.0
            feedback.front_clearance_m = self._front_clearance_or_negative()
            feedback.state = "WAIT"
            goal_handle.publish_feedback(feedback)

            self._publish_zero_once()
            time.sleep(0.05)

        return self._make_result(True, self.SUCCESS, f"WAIT succeeded after {seconds:.2f} s")

    def _send_nav_goal_with_monitoring(
        self,
        goal_handle,
        client,
        nav_goal,
        timeout_s: float,
        feedback_name: str,
        monitor_front: bool,
        target_distance: float,
        target_angle: float,
    ):
        if not client.wait_for_server(timeout_sec=3.0):
            return self._make_result(
                False,
                self.NAV2_SERVER_UNAVAILABLE,
                f"{feedback_name} failed: Nav2 action server unavailable",
            )

        send_done = threading.Event()
        result_done = threading.Event()
        data = {
            "goal_handle": None,
            "result": None,
            "exception": None,
        }

        def on_send_done(future):
            try:
                data["goal_handle"] = future.result()
            except Exception as exc:
                data["exception"] = exc
            finally:
                send_done.set()

        send_future = client.send_goal_async(nav_goal)
        send_future.add_done_callback(on_send_done)

        if not send_done.wait(timeout=3.0):
            return self._make_result(False, self.NAV2_SERVER_UNAVAILABLE, f"{feedback_name} send_goal timed out")

        if data["exception"] is not None:
            return self._make_result(False, self.INTERNAL_ERROR, f"{feedback_name} send_goal exception: {data['exception']}")

        nav_goal_handle = data["goal_handle"]
        if nav_goal_handle is None or not nav_goal_handle.accepted:
            return self._make_result(False, self.NAV2_GOAL_REJECTED, f"{feedback_name} goal rejected by Nav2")

        with self.lock:
            self.active_nav_goal_handle = nav_goal_handle

        def on_result_done(future):
            try:
                data["result"] = future.result()
            except Exception as exc:
                data["exception"] = exc
            finally:
                result_done.set()

        result_future = nav_goal_handle.get_result_async()
        result_future.add_done_callback(on_result_done)

        start_time = time.monotonic()

        while not result_done.is_set():
            elapsed = time.monotonic() - start_time

            if goal_handle.is_cancel_requested:
                self._cancel_active_nav_goal()
                return self._make_result(False, self.CANCELED, f"{feedback_name} canceled by client")

            if elapsed > timeout_s + 0.5:
                self._cancel_active_nav_goal()
                return self._make_result(False, self.TIMEOUT, f"{feedback_name} timed out after {elapsed:.2f} s")

            if monitor_front:
                clearance = self._front_clearance()
                if clearance is not None and clearance < self.front_stop_distance_m:
                    self._cancel_active_nav_goal()
                    self._publish_zero_for(0.30)
                    return self._make_result(
                        False,
                        self.OBSTACLE_TOO_CLOSE,
                        f"{feedback_name} stopped: front clearance {clearance:.3f} m < stop distance {self.front_stop_distance_m:.3f} m",
                    )

            feedback = ExecuteMotionPrimitive.Feedback()
            feedback.progress = clamp(elapsed / max(timeout_s, 1.0e-6), 0.0, 1.0)
            feedback.distance_remaining_m = max(0.0, target_distance * (1.0 - feedback.progress))
            feedback.heading_remaining_rad = abs(target_angle) * (1.0 - feedback.progress)
            feedback.front_clearance_m = self._front_clearance_or_negative()
            feedback.state = feedback_name
            goal_handle.publish_feedback(feedback)

            time.sleep(0.05)

        with self.lock:
            self.active_nav_goal_handle = None

        if data["exception"] is not None:
            return self._make_result(False, self.INTERNAL_ERROR, f"{feedback_name} result exception: {data['exception']}")

        nav_result = data["result"]
        if nav_result is None:
            return self._make_result(False, self.NAV2_FAILED, f"{feedback_name} produced no result")

        if getattr(nav_result, "status", None) != GoalStatus.STATUS_SUCCEEDED:
            status = getattr(nav_result, "status", None)
            return self._make_result(False, self.NAV2_FAILED, f"{feedback_name} failed with action status {status}")

        return self._make_result(True, self.SUCCESS, f"{feedback_name} Nav2 action succeeded")

    def _check_final_translation(self, goal, start_pose: Optional[Pose2D], target_distance: float, name: str):
        if start_pose is None:
            if self.require_odom_for_result_check:
                return self._make_result(False, self.INTERNAL_ERROR, f"{name} cannot verify final distance: no starting odom")
            return self._make_result(True, self.SUCCESS, f"{name} succeeded without odom verification")

        end_pose = self._current_pose()
        if end_pose is None:
            if self.require_odom_for_result_check:
                return self._make_result(False, self.INTERNAL_ERROR, f"{name} cannot verify final distance: no ending odom")
            return self._make_result(True, self.SUCCESS, f"{name} succeeded without ending odom verification")

        actual_distance = math.hypot(end_pose.x - start_pose.x, end_pose.y - start_pose.y)
        error = abs(actual_distance - target_distance)

        hard_fail_error_m = min(0.08, max(0.04, target_distance * 0.5))
        if error > hard_fail_error_m:
            return self._make_result(
                False,
                self.FINAL_ERROR_TOO_LARGE,
                f"{name} final distance error {error:.3f} m > hard fail threshold {hard_fail_error_m:.3f} m",
                final_position_error_m=error,
            )

        if error > self._resolved_position_tolerance(goal.position_tolerance_m):
            self.get_logger().warn(
                f"{name} final distance error {error:.3f} m exceeds requested tolerance, but accepting because odom is diagnostic"
            )

        return self._make_result(
            True,
            self.SUCCESS,
            f"{name} succeeded, final distance error {error:.3f} m",
            final_position_error_m=error,
        )

    def _check_final_rotation(self, goal, start_pose: Optional[Pose2D], target_angle: float):
        if start_pose is None:
            if self.require_odom_for_result_check:
                return self._make_result(False, self.INTERNAL_ERROR, "ROTATE_RELATIVE cannot verify final heading: no starting odom")
            return self._make_result(True, self.SUCCESS, "ROTATE_RELATIVE succeeded without odom verification")

        end_pose = self._current_pose()
        if end_pose is None:
            if self.require_odom_for_result_check:
                return self._make_result(False, self.INTERNAL_ERROR, "ROTATE_RELATIVE cannot verify final heading: no ending odom")
            return self._make_result(True, self.SUCCESS, "ROTATE_RELATIVE succeeded without ending odom verification")

        actual_delta = normalize_angle(end_pose.yaw - start_pose.yaw)
        error = abs(normalize_angle(actual_delta - target_angle))

        hard_fail_error_rad = 0.35
        if error > hard_fail_error_rad:
            return self._make_result(
                False,
                self.FINAL_ERROR_TOO_LARGE,
                f"ROTATE_RELATIVE final heading error {error:.3f} rad > hard fail threshold {hard_fail_error_rad:.3f} rad",
                final_heading_error_rad=error,
            )

        if error > self._resolved_heading_tolerance(goal.heading_tolerance_rad):
            self.get_logger().warn(
                f"ROTATE_RELATIVE final heading error {error:.3f} rad exceeds requested tolerance, but accepting because odom is diagnostic"
            )

        return self._make_result(
            True,
            self.SUCCESS,
            f"ROTATE_RELATIVE succeeded, final heading error {error:.3f} rad",
            final_heading_error_rad=error,
        )

    def _forward_precheck(self) -> Tuple[bool, str]:
        if self.require_scan_for_forward and not self._scan_is_fresh():
            return False, "DRIVE_FORWARD rejected: no fresh LiDAR scan"

        clearance = self._front_clearance()
        if clearance is None:
            if self.require_scan_for_forward:
                return False, "DRIVE_FORWARD rejected: front clearance unavailable"
            return True, "front clearance unavailable but scan not required"

        if clearance < self.front_stop_distance_m:
            return (
                False,
                f"DRIVE_FORWARD rejected: front clearance {clearance:.3f} m < stop distance {self.front_stop_distance_m:.3f} m",
            )

        return True, "front clearance ok"

    def _scan_is_fresh(self) -> bool:
        with self.lock:
            if self.latest_scan is None or self.latest_scan_time is None:
                return False
            age_ns = (self.get_clock().now() - self.latest_scan_time).nanoseconds
        return age_ns < int(0.75 * 1.0e9)

    def _front_clearance_or_negative(self) -> float:
        clearance = self._front_clearance()
        return -1.0 if clearance is None else float(clearance)

    def _front_clearance(self) -> Optional[float]:
        with self.lock:
            scan = self.latest_scan

        if scan is None:
            return None

        half_angle = math.radians(self.front_sector_deg)
        best = None

        for idx, distance in enumerate(scan.ranges):
            if not math.isfinite(distance):
                continue

            if distance < scan.range_min or distance > scan.range_max:
                continue

            angle = scan.angle_min + idx * scan.angle_increment
            if abs(normalize_angle(angle)) <= half_angle:
                if best is None or distance < best:
                    best = float(distance)

        return best

    def _current_pose(self) -> Optional[Pose2D]:
        with self.lock:
            odom = self.latest_odom

        if odom is None:
            return None

        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        return Pose2D(float(p.x), float(p.y), yaw_from_quat(q))

    def _cancel_active_nav_goal(self):
        with self.lock:
            handle = self.active_nav_goal_handle
            self.active_nav_goal_handle = None

        if handle is None:
            return

        try:
            future = handle.cancel_goal_async()
            done = threading.Event()
            future.add_done_callback(lambda _: done.set())
            done.wait(timeout=1.0)
        except Exception as exc:
            self.get_logger().warn(f"Failed to cancel active Nav2 goal: {exc}")

    def _publish_zero_once(self):
        self.zero_pub.publish(Twist())

    def _publish_zero_for(self, seconds: float):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self._publish_zero_once()
            time.sleep(0.05)
        self._publish_zero_once()

    def _resolved_position_tolerance(self, requested: float) -> float:
        if requested > 0.0 and math.isfinite(requested):
            return float(requested)
        return float(self.default_position_tolerance_m)

    def _resolved_heading_tolerance(self, requested: float) -> float:
        if requested > 0.0 and math.isfinite(requested):
            return float(requested)
        return float(self.default_heading_tolerance_rad)

    def _resolved_linear_speed(self, requested: float) -> float:
        if requested > 0.0 and math.isfinite(requested):
            return min(float(requested), float(self.max_linear_speed_mps))
        return min(float(self.default_linear_speed_mps), float(self.max_linear_speed_mps))

    def _resolved_reverse_speed(self, requested: float) -> float:
        if requested > 0.0 and math.isfinite(requested):
            return min(float(requested), float(self.max_linear_speed_mps))
        return min(float(self.default_reverse_speed_mps), float(self.max_linear_speed_mps))

    def _resolved_angular_speed(self, requested: float) -> float:
        if requested > 0.0 and math.isfinite(requested):
            return min(float(requested), float(self.max_angular_speed_radps))
        return min(float(self.default_angular_speed_radps), float(self.max_angular_speed_radps))

    def _resolved_timeout(self, requested: float, nominal_duration_s: float) -> float:
        if requested > 0.0 and math.isfinite(requested):
            return float(requested)
        return max(1.0, float(nominal_duration_s) + float(self.timeout_margin_s))

    def _make_result(
        self,
        success: bool,
        code: int,
        message: str,
        final_position_error_m: float = 0.0,
        final_heading_error_rad: float = 0.0,
    ):
        result = ExecuteMotionPrimitive.Result()
        result.success = bool(success)
        result.result_code = int(code)
        result.message = str(message)
        result.final_position_error_m = float(final_position_error_m)
        result.final_heading_error_rad = float(final_heading_error_rad)

        if success:
            self.get_logger().info(message)
        else:
            self.get_logger().error(message)

        return result


def yaw_from_quat(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def duration_from_seconds(seconds: float) -> Duration:
    seconds = max(0.0, float(seconds))
    whole = int(seconds)
    msg = Duration()
    msg.sec = whole
    msg.nanosec = int((seconds - whole) * 1.0e9)
    return msg


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def main(args=None):
    rclpy.init(args=args)
    node = MotionPrimitiveServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
