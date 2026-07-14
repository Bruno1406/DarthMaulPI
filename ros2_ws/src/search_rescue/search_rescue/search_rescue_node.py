from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

import rclpy
from apriltag_msgs.msg import AprilTagDetectionArray
from cv_bridge import CvBridge
from maze_interface.srv import GradeCubes
from maze_explorer_node import (
    Cell,
    DIR_NAME,
    OPEN,
    MazeExplorerNode,
    MotionStep,
    NEG_X,
    NEG_Y,
    POS_X,
    POS_Y,
    SparseMazeMap,
    neighbor,
    parse_bool,
)
from sensor_msgs.msg import CameraInfo, Image

from search_rescue.cube_tracker import (
    ADD_RESULT_ADDED,
    ADD_RESULT_CAPACITY,
    MAX_CUBES,
    CubeTracker,
)
from search_rescue.vision import detect_color


class SearchRescueMazeMap(SparseMazeMap):
    """The v87 maze map plus Task 3-only dynamic cube obstacles."""

    def __init__(
        self,
        confirm_hits: int = 1,
        conflict_override_hits: int = 3,
    ) -> None:
        super().__init__(
            confirm_hits,
            conflict_override_hits,
        )
        self.blocked_cells: set[Cell] = set()

    def block_cell(
        self,
        cell: Cell,
    ) -> bool:
        if cell in self.blocked_cells:
            return False

        self.blocked_cells.add(cell)
        return True

    def open_neighbors(
        self,
        cell: Cell,
    ) -> List[Tuple[int, Cell]]:
        return [
            (direction, next_cell)
            for direction, next_cell in super().open_neighbors(cell)
            if next_cell not in self.blocked_cells
        ]


class SearchRescueNode(MazeExplorerNode):
    """Task 3 perception and grading layered on the unchanged v87 explorer."""

    def __init__(self) -> None:
        super().__init__()

        # Task 3 submits cube tuples, not the Task 2 maze service.
        if self.submit_maze_to_grader:
            self.get_logger().warn(
                'Ignoring submit_maze_to_grader=true in Task 3. '
                'Only /grade_cubes will be submitted.'
            )
            self.submit_maze_to_grader = False

        # Prevent the base explorer from shutting down before the asynchronous
        # cube service request has completed.
        self.task3_shutdown_on_complete = bool(
            self.shutdown_on_complete
        )
        self.shutdown_on_complete = False

        self.declare_parameter(
            'rect_image_topic',
            '/ascamera/camera_publisher/rgb0/image_rect',
        )
        self.declare_parameter(
            'raw_image_topic',
            '/ascamera/camera_publisher/rgb0/image',
        )
        self.declare_parameter(
            'camera_info_topic',
            '/ascamera/camera_publisher/rgb0/camera_info',
        )
        self.declare_parameter(
            'apriltag_topic',
            '/apriltag_detections',
        )

        self.declare_parameter(
            'tag_size_m',
            0.0162,
        )
        self.declare_parameter(
            'tag_min_side_px',
            15.0,
        )
        self.declare_parameter(
            'tag_min_aspect_ratio',
            0.60,
        )
        self.declare_parameter(
            'tag_max_aspect_ratio',
            1.50,
        )
        self.declare_parameter(
            'cube_crop_scale',
            2.30,
        )
        self.declare_parameter(
            'camera_data_timeout_s',
            0.75,
        )

        # These are the newer physical corrections from v1.
        self.declare_parameter(
            'distance_scale',
            1.18,
        )
        self.declare_parameter(
            'distance_bias_cm',
            -3.19,
        )
        self.declare_parameter(
            'cube_center_offset_cm',
            1.50,
        )
        self.declare_parameter(
            'minimum_detection_distance_cm',
            8.0,
        )
        self.declare_parameter(
            'maximum_detection_distance_cm',
            35.0,
        )

        self.declare_parameter(
            'camera_forward_offset_cm',
            8.0,
        )
        self.declare_parameter(
            'camera_left_offset_cm',
            0.0,
        )
        self.declare_parameter(
            'cube_merge_distance_cm',
            20.0,
        )
        self.declare_parameter(
            'cube_cell_assignment_tolerance_cm',
            13.0,
        )
        self.declare_parameter(
            'block_cube_cells',
            True,
        )
        self.declare_parameter(
            'observation_hold_s',
            0.60,
        )

        self.declare_parameter(
            'submit_cubes_to_grader',
            True,
        )
        self.declare_parameter(
            'cube_grade_service_name',
            '/grade_cubes',
        )
        self.declare_parameter(
            'cube_grade_service_timeout_s',
            10.0,
        )
        self.declare_parameter(
            'cube_grade_result_required',
            False,
        )

        self.rect_image_topic = str(
            self.get_parameter(
                'rect_image_topic'
            ).value
        ).strip()

        self.raw_image_topic = str(
            self.get_parameter(
                'raw_image_topic'
            ).value
        ).strip()

        self.camera_info_topic = str(
            self.get_parameter(
                'camera_info_topic'
            ).value
        ).strip()

        self.apriltag_topic = str(
            self.get_parameter(
                'apriltag_topic'
            ).value
        ).strip()

        self.tag_size_m = float(
            self.get_parameter(
                'tag_size_m'
            ).value
        )

        self.tag_min_side_px = float(
            self.get_parameter(
                'tag_min_side_px'
            ).value
        )

        self.tag_min_aspect_ratio = float(
            self.get_parameter(
                'tag_min_aspect_ratio'
            ).value
        )

        self.tag_max_aspect_ratio = float(
            self.get_parameter(
                'tag_max_aspect_ratio'
            ).value
        )

        self.cube_crop_scale = float(
            self.get_parameter(
                'cube_crop_scale'
            ).value
        )

        self.camera_data_timeout_s = float(
            self.get_parameter(
                'camera_data_timeout_s'
            ).value
        )

        self.distance_scale = float(
            self.get_parameter(
                'distance_scale'
            ).value
        )

        self.distance_bias_cm = float(
            self.get_parameter(
                'distance_bias_cm'
            ).value
        )

        self.cube_center_offset_cm = float(
            self.get_parameter(
                'cube_center_offset_cm'
            ).value
        )

        self.minimum_detection_distance_cm = float(
            self.get_parameter(
                'minimum_detection_distance_cm'
            ).value
        )

        self.maximum_detection_distance_cm = float(
            self.get_parameter(
                'maximum_detection_distance_cm'
            ).value
        )

        self.camera_forward_offset_cm = float(
            self.get_parameter(
                'camera_forward_offset_cm'
            ).value
        )

        self.camera_left_offset_cm = float(
            self.get_parameter(
                'camera_left_offset_cm'
            ).value
        )

        self.cube_merge_distance_cm = float(
            self.get_parameter(
                'cube_merge_distance_cm'
            ).value
        )

        self.cube_cell_assignment_tolerance_cm = float(
            self.get_parameter(
                'cube_cell_assignment_tolerance_cm'
            ).value
        )

        self.block_cube_cells = parse_bool(
            self.get_parameter(
                'block_cube_cells'
            ).value
        )

        self.observation_hold_s = float(
            self.get_parameter(
                'observation_hold_s'
            ).value
        )

        self.submit_cubes_to_grader = parse_bool(
            self.get_parameter(
                'submit_cubes_to_grader'
            ).value
        )

        self.cube_grade_service_name = str(
            self.get_parameter(
                'cube_grade_service_name'
            ).value
        ).strip()

        self.cube_grade_service_timeout_s = float(
            self.get_parameter(
                'cube_grade_service_timeout_s'
            ).value
        )

        self.cube_grade_result_required = parse_bool(
            self.get_parameter(
                'cube_grade_result_required'
            ).value
        )

        self._validate_task3_parameters()

        # The base constructor has just created an empty SparseMazeMap.
        # Replace only that empty map with a Task 3-specific subclass.
        self.maze = SearchRescueMazeMap(
            confirm_hits=int(
                self.get_parameter(
                    'wall_confirm_hits'
                ).value
            ),
            conflict_override_hits=int(
                self.get_parameter(
                    'conflict_override_hits'
                ).value
            ),
        )

        self.tracker = CubeTracker(
            limit_distance_cm=self.cube_merge_distance_cm,
            max_cubes=MAX_CUBES,
        )

        self.bridge = CvBridge()

        self.latest_rect_image = None
        self.latest_raw_image = None
        self.latest_rect_image_monotonic = 0.0
        self.latest_raw_image_monotonic = 0.0
        self.camera_info: Optional[CameraInfo] = None

        # Allow camera observations before the first motion.
        self.observation_hold_until_monotonic = (
            time.monotonic()
            + self.observation_hold_s
        )

        self.rect_image_subscriber = self.create_subscription(
            Image,
            self.rect_image_topic,
            self.callback_rect_image,
            10,
        )

        self.raw_image_subscriber = self.create_subscription(
            Image,
            self.raw_image_topic,
            self.callback_raw_image,
            10,
        )

        self.camera_info_subscriber = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.callback_camera_info,
            10,
        )

        self.apriltag_subscriber = self.create_subscription(
            AprilTagDetectionArray,
            self.apriltag_topic,
            self.callback_apriltag,
            10,
        )

        self.cube_grade_client = self.create_client(
            GradeCubes,
            self.cube_grade_service_name,
        )

        self.cube_grade_future = None
        self.cube_grade_submit_time = None
        self.cube_submission_started = False
        self.cube_submission_finished = False

        self.get_logger().info(
            'Task 3 search and rescue initialized on v87 explorer: '
            f'max_cubes={MAX_CUBES}, '
            f'tag_size={self.tag_size_m:.4f} m, '
            f'distance={self.distance_scale:.3f}*measured'
            f'{self.distance_bias_cm:+.3f} cm, '
            f'cube_center_offset='
            f'{self.cube_center_offset_cm:.2f} cm'
        )

    def _validate_task3_parameters(self) -> None:
        errors = []

        topic_parameters = {
            'rect_image_topic': self.rect_image_topic,
            'raw_image_topic': self.raw_image_topic,
            'camera_info_topic': self.camera_info_topic,
            'apriltag_topic': self.apriltag_topic,
        }

        for name, value in topic_parameters.items():
            if not value:
                errors.append(
                    f'{name} must not be empty'
                )

        positive_parameters = {
            'tag_size_m': self.tag_size_m,
            'tag_min_side_px': self.tag_min_side_px,
            'cube_crop_scale': self.cube_crop_scale,
            'camera_data_timeout_s': (
                self.camera_data_timeout_s
            ),
            'minimum_detection_distance_cm': (
                self.minimum_detection_distance_cm
            ),
            'maximum_detection_distance_cm': (
                self.maximum_detection_distance_cm
            ),
            'cube_merge_distance_cm': (
                self.cube_merge_distance_cm
            ),
            'cube_cell_assignment_tolerance_cm': (
                self.cube_cell_assignment_tolerance_cm
            ),
            'cube_grade_service_timeout_s': (
                self.cube_grade_service_timeout_s
            ),
        }

        for name, value in positive_parameters.items():
            if (
                not math.isfinite(value)
                or value <= 0.0
            ):
                errors.append(
                    f'{name} must be finite and > 0'
                )

        if (
            not math.isfinite(self.distance_scale)
            or self.distance_scale <= 0.0
        ):
            errors.append(
                'distance_scale must be finite and > 0'
            )

        if not math.isfinite(
            self.distance_bias_cm
        ):
            errors.append(
                'distance_bias_cm must be finite'
            )

        if not math.isfinite(
            self.cube_center_offset_cm
        ):
            errors.append(
                'cube_center_offset_cm must be finite'
            )

        if not math.isfinite(
            self.camera_forward_offset_cm
        ):
            errors.append(
                'camera_forward_offset_cm must be finite'
            )

        if not math.isfinite(
            self.camera_left_offset_cm
        ):
            errors.append(
                'camera_left_offset_cm must be finite'
            )

        if (
            self.maximum_detection_distance_cm
            <= self.minimum_detection_distance_cm
        ):
            errors.append(
                'maximum_detection_distance_cm must be '
                'greater than minimum_detection_distance_cm'
            )

        if self.tag_min_aspect_ratio <= 0.0:
            errors.append(
                'tag_min_aspect_ratio must be > 0'
            )

        if (
            self.tag_max_aspect_ratio
            < self.tag_min_aspect_ratio
        ):
            errors.append(
                'tag_max_aspect_ratio must be '
                '>= tag_min_aspect_ratio'
            )

        if self.observation_hold_s < 0.0:
            errors.append(
                'observation_hold_s must be >= 0'
            )

        if (
            self.submit_cubes_to_grader
            and not self.cube_grade_service_name
        ):
            errors.append(
                'cube_grade_service_name must not be empty '
                'when submit_cubes_to_grader=true'
            )

        if errors:
            message = '; '.join(errors)
            self.get_logger().error(message)
            raise ValueError(message)

    def callback_rect_image(
        self,
        msg: Image,
    ) -> None:
        try:
            self.latest_rect_image = (
                self.bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding='bgr8',
                )
            )
        except Exception as exc:
            self.get_logger().warn(
                f'Failed to convert rectified image: {exc}',
                throttle_duration_sec=2.0,
            )
            return

        self.latest_rect_image_monotonic = (
            time.monotonic()
        )

    def callback_raw_image(
        self,
        msg: Image,
    ) -> None:
        try:
            self.latest_raw_image = (
                self.bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding='bgr8',
                )
            )
        except Exception as exc:
            self.get_logger().warn(
                f'Failed to convert raw image: {exc}',
                throttle_duration_sec=2.0,
            )
            return

        self.latest_raw_image_monotonic = (
            time.monotonic()
        )

    def callback_camera_info(
        self,
        msg: CameraInfo,
    ) -> None:
        if self.camera_info is not None:
            return

        self.camera_info = msg

        self.get_logger().info(
            f'Camera intrinsics received: '
            f'fx={float(msg.p[0]):.2f}, '
            f'cx={float(msg.p[2]):.2f}'
        )

    def callback_apriltag(
        self,
        msg: AprilTagDetectionArray,
    ) -> None:
        if (
            self.completed
            or self.motion_in_flight
            or not msg.detections
        ):
            return

        now = time.monotonic()

        if (
            self.latest_rect_image is None
            or (
                now
                - self.latest_rect_image_monotonic
                > self.camera_data_timeout_s
            )
        ):
            self.get_logger().debug(
                'Ignoring tag detection without '
                'a fresh rectified image.'
            )
            return

        if (
            self.latest_raw_image is None
            or (
                now
                - self.latest_raw_image_monotonic
                > self.camera_data_timeout_s
            )
        ):
            self.get_logger().debug(
                'Ignoring tag detection without '
                'a fresh raw image.'
            )
            return

        valid_detections = []

        for detection in msg.detections:
            if len(detection.corners) < 4:
                continue

            xs = [
                float(corner.x)
                for corner in detection.corners
            ]
            ys = [
                float(corner.y)
                for corner in detection.corners
            ]

            width_px = max(xs) - min(xs)
            height_px = max(ys) - min(ys)

            if (
                width_px < self.tag_min_side_px
                or height_px < self.tag_min_side_px
            ):
                continue

            aspect_ratio = (
                height_px / width_px
                if width_px > 0.0
                else 0.0
            )

            if (
                aspect_ratio
                < self.tag_min_aspect_ratio
                or aspect_ratio
                > self.tag_max_aspect_ratio
            ):
                continue

            valid_detections.append(
                (
                    detection,
                    width_px * height_px,
                )
            )

        if not valid_detections:
            return

        detection, _ = max(
            valid_detections,
            key=lambda item: item[1],
        )

        relative_position = (
            self.estimate_cube_relative_position(
                detection
            )
        )

        if relative_position is None:
            return

        forward_cm, left_cm = relative_position

        # Preserve the newer v1 physical calibration.
        forward_cm = (
            self.distance_scale * forward_cm
            + self.distance_bias_cm
        )

        # Convert tag/front-face distance to cube-center distance.
        forward_cm += self.cube_center_offset_cm

        if (
            forward_cm
            < self.minimum_detection_distance_cm
            or forward_cm
            > self.maximum_detection_distance_cm
        ):
            return

        local_x_cm, local_y_cm = (
            self.cube_camera_to_local(
                forward_cm,
                left_cm,
            )
        )

        # The AprilTag itself is enough to identify a cube obstacle.
        # Color detection is not required before avoiding its cell.
        if self.block_cube_cells:
            cube_cell = self.infer_cube_cell(
                local_x_cm,
                local_y_cm,
            )

            if (
                cube_cell is not None
                and cube_cell != self.current_cell
            ):
                if self.maze.block_cell(cube_cell):
                    # A previously queued route may pass through this cell.
                    self.motion_queue.clear()

                    self.get_logger().warn(
                        f'Cube obstacle added at '
                        f'cell={cube_cell}; '
                        'discarded queued route and will replan.'
                    )

                    self._publish_current_maze()

        raw_crop = self.crop_cube_from_detection(
            self.latest_raw_image,
            detection,
        )

        if raw_crop is None:
            return

        color_id = detect_color(raw_crop)

        if color_id is None:
            return

        add_result = self.tracker.add_cube(
            local_x_cm,
            local_y_cm,
            int(color_id),
        )

        if add_result == ADD_RESULT_ADDED:
            self.get_logger().info(
                f'New cube accepted: '
                f'count={self.tracker.get_cube_count()}'
                f'/{MAX_CUBES}, '
                f'local_x={local_x_cm:.1f} cm, '
                f'local_y={local_y_cm:.1f} cm, '
                f'color={int(color_id)}'
            )

        elif add_result == ADD_RESULT_CAPACITY:
            self.get_logger().warn(
                'Rejected a fifth distinct cube hypothesis. '
                'The Task 3 request is strictly capped '
                'at four cubes.',
                throttle_duration_sec=2.0,
            )

    def crop_cube_from_detection(
        self,
        image,
        detection,
    ):
        if (
            image is None
            or len(detection.corners) < 4
        ):
            return None

        height, width = image.shape[:2]

        center_x = int(detection.centre.x)
        center_y = int(detection.centre.y)

        xs = [
            float(corner.x)
            for corner in detection.corners
        ]
        ys = [
            float(corner.y)
            for corner in detection.corners
        ]

        tag_size_px = max(
            max(xs) - min(xs),
            max(ys) - min(ys),
        )

        if tag_size_px <= 0.0:
            return None

        crop_size = max(
            1,
            int(
                tag_size_px
                * self.cube_crop_scale
            ),
        )

        half_size = crop_size // 2

        x1 = max(
            0,
            center_x - half_size,
        )
        x2 = min(
            width,
            center_x + half_size,
        )
        y1 = max(
            0,
            center_y - half_size,
        )
        y2 = min(
            height,
            center_y + half_size,
        )

        if (
            x2 <= x1
            or y2 <= y1
        ):
            return None

        return image[y1:y2, x1:x2]

    def estimate_cube_relative_position(
        self,
        detection,
    ) -> Optional[Tuple[float, float]]:
        """
        Return camera-relative coordinates as:

            forward_cm
            left_cm
        """

        if (
            self.camera_info is None
            or len(detection.corners) < 4
        ):
            return None

        focal_x = float(
            self.camera_info.p[0]
        )
        principal_x = float(
            self.camera_info.p[2]
        )

        if (
            not math.isfinite(focal_x)
            or focal_x <= 0.0
        ):
            return None

        side_lengths = []
        corners = detection.corners

        for index in range(4):
            first = corners[index]
            second = corners[
                (index + 1) % 4
            ]

            dx = (
                float(first.x)
                - float(second.x)
            )
            dy = (
                float(first.y)
                - float(second.y)
            )

            side_lengths.append(
                math.hypot(dx, dy)
            )

        tag_width_px = max(side_lengths)

        if (
            not math.isfinite(tag_width_px)
            or tag_width_px <= 0.0
        ):
            return None

        distance_m = (
            self.tag_size_m
            * focal_x
            / tag_width_px
        )

        image_offset_x = (
            float(detection.centre.x)
            - principal_x
        )

        lateral_m = (
            image_offset_x
            * distance_m
            / focal_x
        )

        forward_cm = distance_m * 100.0

        # A positive image offset is camera-right.
        # The Task 3 local convention uses positive left.
        left_cm = -lateral_m * 100.0

        return forward_cm, left_cm

    def cube_camera_to_local(
        self,
        forward_cm: float,
        left_cm: float,
    ) -> Tuple[float, float]:
        """
        Convert camera-relative coordinates into the same local grid frame
        used by the v87 maze explorer.
        """

        cell_cm = (
            self.cell_length_m
            * 100.0
        )

        # The explorer's integer cell coordinate represents the cell center.
        robot_x_cm = (
            float(self.current_cell[0])
            * cell_cm
        )
        robot_y_cm = (
            float(self.current_cell[1])
            * cell_cm
        )

        forward_from_robot_cm = (
            float(forward_cm)
            + self.camera_forward_offset_cm
        )

        left_from_robot_cm = (
            float(left_cm)
            + self.camera_left_offset_cm
        )

        if self.heading == POS_X:
            dx_cm = forward_from_robot_cm
            dy_cm = left_from_robot_cm

        elif self.heading == POS_Y:
            dx_cm = -left_from_robot_cm
            dy_cm = forward_from_robot_cm

        elif self.heading == NEG_X:
            dx_cm = -forward_from_robot_cm
            dy_cm = -left_from_robot_cm

        elif self.heading == NEG_Y:
            dx_cm = left_from_robot_cm
            dy_cm = -forward_from_robot_cm

        else:
            raise ValueError(
                f'Invalid heading: {self.heading}'
            )

        return (
            robot_x_cm + dx_cm,
            robot_y_cm + dy_cm,
        )

    def infer_cube_cell(
        self,
        local_x_cm: float,
        local_y_cm: float,
    ) -> Optional[Cell]:
        cell_cm = (
            self.cell_length_m
            * 100.0
        )

        cell = (
            int(
                round(
                    float(local_x_cm)
                    / cell_cm
                )
            ),
            int(
                round(
                    float(local_y_cm)
                    / cell_cm
                )
            ),
        )

        center_x_cm = (
            float(cell[0])
            * cell_cm
        )
        center_y_cm = (
            float(cell[1])
            * cell_cm
        )

        center_error_cm = math.hypot(
            float(local_x_cm)
            - center_x_cm,
            float(local_y_cm)
            - center_y_cm,
        )

        if (
            center_error_cm
            > self.cube_cell_assignment_tolerance_cm
        ):
            self.get_logger().debug(
                f'Cube at '
                f'({local_x_cm:.1f}, '
                f'{local_y_cm:.1f}) cm is '
                f'{center_error_cm:.1f} cm from '
                'the nearest cell center; '
                'not using it as a planner obstacle.'
            )
            return None

        return cell

    def _choose_unvisited_open_neighbor(
        self,
        cell: Cell,
    ) -> Optional[int]:
        """
        The v87 legacy neighbor selection with one Task 3 addition:
        cube-occupied cells are excluded.
        """

        for direction in self._ordered_directions():
            if (
                self.maze.wall_state(
                    cell,
                    direction,
                )
                != OPEN
            ):
                continue

            next_cell = neighbor(
                cell,
                direction,
            )

            if next_cell in self.maze.blocked_cells:
                continue

            if not self.maze.is_visited(
                next_cell
            ):
                self.get_logger().info(
                    f'Choosing frontier: '
                    f'{cell} -> {next_cell} '
                    f'via {DIR_NAME[direction]}'
                )
                return direction

        unresolved = (
            self.maze.unresolved_directions(
                cell
            )
        )

        if unresolved:
            self.get_logger().warn(
                f'No open unvisited edge from {cell}; '
                'unresolved='
                + ','.join(
                    DIR_NAME[direction]
                    for direction in unresolved
                )
            )

        return None

    def _handle_motion_result(
        self,
        future,
    ) -> None:
        completed_step: Optional[MotionStep] = (
            self.active_step
        )

        # This performs all normal v87 result validation and pose updates.
        super()._handle_motion_result(
            future
        )

        if (
            completed_step is not None
            and not self.shutdown_requested
            and not self.motion_in_flight
            and self.active_step is None
        ):
            # Delay the next dispatch so the stationary camera receives several
            # image and AprilTag messages at the newly updated robot pose.
            self.observation_hold_until_monotonic = (
                time.monotonic()
                + self.observation_hold_s
            )

    def _tick(self) -> None:
        self._check_cube_grade_timeout()

        if (
            not self.completed
            and not self.motion_in_flight
            and (
                time.monotonic()
                < self.observation_hold_until_monotonic
            )
        ):
            return

        # All exploration, planning and motion dispatch remain v87 behavior.
        super()._tick()

        self._maybe_request_task3_shutdown()

    def _finish_exploration(
        self,
        reason: str,
    ) -> None:
        if self.completed:
            return

        # Export the final maze and preserve all normal v87 completion logging.
        super()._finish_exploration(
            reason
        )

        if not self.completed:
            return

        # Cube positions are translated to the final maze-origin frame only
        # after the explored maze bounds are known.
        self._submit_cubes_once()

        self._maybe_request_task3_shutdown()

    def _maze_origin_in_local_cm(
        self,
    ) -> Tuple[float, float]:
        if not self.maze.cells:
            self.maze.ensure_cell(
                self.start_cell
            )

        minimum_x = min(
            cell[0]
            for cell in self.maze.cells
        )
        minimum_y = min(
            cell[1]
            for cell in self.maze.cells
        )

        cell_cm = (
            self.cell_length_m
            * 100.0
        )

        # Integer coordinates represent cell centers. The origin lies half a
        # cell before the minimum-coordinate center on each maze axis.
        origin_x_cm = (
            float(minimum_x) - 0.5
        ) * cell_cm

        origin_y_cm = (
            float(minimum_y) - 0.5
        ) * cell_cm

        return (
            origin_x_cm,
            origin_y_cm,
        )

    def _cube_service_arrays(
        self,
    ) -> Tuple[
        int,
        List[int],
        List[int],
        List[int],
    ]:
        origin_x_cm, origin_y_cm = (
            self._maze_origin_in_local_cm()
        )

        # This slice is a second independent four-cube safety guard.
        cubes = (
            self.tracker.snapshot()[
                :MAX_CUBES
            ]
        )

        xs = [
            int(
                round(
                    cube[0]
                    - origin_x_cm
                )
            )
            for cube in cubes
        ]

        ys = [
            int(
                round(
                    cube[1]
                    - origin_y_cm
                )
            )
            for cube in cubes
        ]

        colors = [
            int(cube[2])
            for cube in cubes
        ]

        if not (
            len(xs)
            == len(ys)
            == len(colors)
        ):
            raise RuntimeError(
                'Cube service arrays have '
                'different lengths.'
            )

        if len(xs) > MAX_CUBES:
            raise RuntimeError(
                'Refusing to create a request '
                'with more than four cubes.'
            )

        for value in xs + ys:
            if (
                value < -32768
                or value > 32767
            ):
                raise ValueError(
                    f'Cube coordinate {value} '
                    'does not fit int16.'
                )

        for color_id in colors:
            if (
                color_id < 0
                or color_id > 255
            ):
                raise ValueError(
                    f'Cube color {color_id} '
                    'does not fit uint8.'
                )

        return (
            len(xs),
            xs,
            ys,
            colors,
        )

    def _submit_cubes_once(self) -> None:
        if self.cube_submission_started:
            return

        self.cube_submission_started = True

        if not self.submit_cubes_to_grader:
            self.get_logger().info(
                'Cube grader submission is disabled.'
            )
            self.cube_submission_finished = True
            return

        try:
            (
                cube_count,
                xs,
                ys,
                colors,
            ) = self._cube_service_arrays()

        except Exception as exc:
            self.get_logger().error(
                f'Cube request validation failed: {exc}'
            )

            if self.cube_grade_result_required:
                self.exit_code = 1

            self.cube_submission_finished = True
            return

        # Third and final hard guard. The outgoing service request can never
        # contain n > 4, even if another component is modified incorrectly.
        cube_count = min(
            int(cube_count),
            MAX_CUBES,
        )
        xs = xs[:MAX_CUBES]
        ys = ys[:MAX_CUBES]
        colors = colors[:MAX_CUBES]

        if not self.cube_grade_client.wait_for_service(
            timeout_sec=(
                self.cube_grade_service_timeout_s
            )
        ):
            self.get_logger().error(
                f'Cube grade service '
                f'{self.cube_grade_service_name} '
                'was not available after '
                f'{self.cube_grade_service_timeout_s:.1f} s.'
            )

            if self.cube_grade_result_required:
                self.exit_code = 1

            self.cube_submission_finished = True
            return

        request = GradeCubes.Request()

        request.n = int(
            cube_count
        )
        request.x = [
            int(value)
            for value in xs
        ]
        request.y = [
            int(value)
            for value in ys
        ]
        request.color = [
            int(value)
            for value in colors
        ]

        self.get_logger().info(
            f'Submitting Task 3 cubes: '
            f'n={request.n}, '
            f'x={list(request.x)}, '
            f'y={list(request.y)}, '
            f'color={list(request.color)}'
        )

        self.cube_grade_submit_time = (
            self.get_clock().now()
        )

        self.cube_grade_future = (
            self.cube_grade_client.call_async(
                request
            )
        )

        self.cube_grade_future.add_done_callback(
            self._cube_grade_response_callback
        )

    def _cube_grade_response_callback(
        self,
        future,
    ) -> None:
        if self.cube_submission_finished:
            return

        self.cube_grade_submit_time = None

        try:
            response = future.result()

        except Exception as exc:
            self.get_logger().error(
                f'Cube grade service call failed: {exc}'
            )

            if self.cube_grade_result_required:
                self.exit_code = 1

        else:
            self.get_logger().info(
                'Cube grade service response: '
                f'score={int(response.score)}'
            )

            print(
                f'TASK3_GRADE_SCORE='
                f'{int(response.score)}',
                flush=True,
            )

        self.cube_submission_finished = True
        self._maybe_request_task3_shutdown()

    def _check_cube_grade_timeout(
        self,
    ) -> None:
        if (
            self.cube_submission_finished
            or self.cube_grade_future is None
            or self.cube_grade_future.done()
            or self.cube_grade_submit_time is None
        ):
            return

        elapsed_s = (
            self.get_clock().now()
            - self.cube_grade_submit_time
        ).nanoseconds * 1.0e-9

        if (
            elapsed_s
            <= self.cube_grade_service_timeout_s
        ):
            return

        self.get_logger().error(
            'Cube grade service response timed out '
            f'after {elapsed_s:.1f} s; '
            f'service={self.cube_grade_service_name}'
        )

        try:
            self.cube_grade_future.cancel()
        except Exception:
            pass

        if self.cube_grade_result_required:
            self.exit_code = 1

        self.cube_grade_submit_time = None
        self.cube_submission_finished = True

        self._maybe_request_task3_shutdown()

    def _maybe_request_task3_shutdown(
        self,
    ) -> None:
        if (
            not self.completed
            or not self.task3_shutdown_on_complete
        ):
            return

        if self.cube_submission_finished:
            self.shutdown_requested = True


def main(args=None) -> None:
    rclpy.init(args=args)

    node: Optional[SearchRescueNode] = None
    exit_code = 0

    try:
        node = SearchRescueNode()

        while (
            rclpy.ok()
            and not node.shutdown_requested
        ):
            rclpy.spin_once(
                node,
                timeout_sec=0.1,
            )

        exit_code = node.exit_code

    except KeyboardInterrupt:
        exit_code = 130

    finally:
        if node is not None:
            try:
                node.destroy_node()
            except KeyboardInterrupt:
                exit_code = 130

        if rclpy.ok():
            try:
                rclpy.shutdown()
            except KeyboardInterrupt:
                exit_code = 130

    if exit_code:
        raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
