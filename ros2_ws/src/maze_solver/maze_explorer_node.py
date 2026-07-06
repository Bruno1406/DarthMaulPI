from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
from statistics import median
import time
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import rclpy
from darth_maul_control_interfaces.action import ExecuteMotionPrimitive
from maze_interface.msg import RosMaze
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


# Course maze encoding:
#   1 = +x, 2 = -x, 4 = +y, 8 = -y
POS_X = 1
NEG_X = 2
POS_Y = 4
NEG_Y = 8
DIRECTIONS = (POS_X, POS_Y, NEG_X, NEG_Y)

UNKNOWN = 0
OPEN = 1
WALL = 2

DIR_NAME = {POS_X: '+x', NEG_X: '-x', POS_Y: '+y', NEG_Y: '-y'}
STATE_NAME = {UNKNOWN: 'unknown', OPEN: 'open', WALL: 'wall'}

DIR_DELTAS = {
    POS_X: (1, 0),
    NEG_X: (-1, 0),
    POS_Y: (0, 1),
    NEG_Y: (0, -1),
}
OPPOSITE = {POS_X: NEG_X, NEG_X: POS_X, POS_Y: NEG_Y, NEG_Y: POS_Y}
LEFT_OF = {POS_X: POS_Y, POS_Y: NEG_X, NEG_X: NEG_Y, NEG_Y: POS_X}
RIGHT_OF = {POS_X: NEG_Y, NEG_Y: NEG_X, NEG_X: POS_Y, POS_Y: POS_X}

Cell = Tuple[int, int]


@dataclass(frozen=True)
class MotionStep:
    kind: str
    direction: int = 0
    value: float = 0.0
    target_cell: Optional[Cell] = None


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'true', '1', 'yes', 'y', 'on'}:
            return True
        if normalized in {'false', '0', 'no', 'n', 'off'}:
            return False
    raise ValueError(f'Cannot parse boolean value: {value!r}')


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def orientation_angle(orientation: int) -> float:
    if orientation == POS_X:
        return 0.0
    if orientation == POS_Y:
        return math.pi / 2.0
    if orientation == NEG_X:
        return math.pi
    if orientation == NEG_Y:
        return -math.pi / 2.0
    raise ValueError(f'Invalid orientation: {orientation}')


def turn_between(current: int, target: int) -> float:
    return normalize_angle(orientation_angle(target) - orientation_angle(current))


def neighbor(cell: Cell, direction: int) -> Cell:
    dx, dy = DIR_DELTAS[direction]
    return (cell[0] + dx, cell[1] + dy)


def direction_between(a: Cell, b: Cell) -> int:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    if (dx, dy) == (1, 0):
        return POS_X
    if (dx, dy) == (-1, 0):
        return NEG_X
    if (dx, dy) == (0, 1):
        return POS_Y
    if (dx, dy) == (0, -1):
        return NEG_Y
    raise ValueError(f'Cells are not adjacent: {a} -> {b}')


def valid_ranges_in_sector(scan: LaserScan, center_rad: float, width_deg: float) -> List[float]:
    if scan is None:
        return []

    half_width = math.radians(float(width_deg)) / 2.0
    center = normalize_angle(float(center_rad))
    angle = float(scan.angle_min)
    values: List[float] = []

    for raw in scan.ranges:
        value = float(raw)
        if abs(normalize_angle(angle - center)) <= half_width:
            if math.isfinite(value) and scan.range_min <= value <= scan.range_max:
                values.append(value)
        angle += float(scan.angle_increment)

    return values


class SparseMazeMap:
    def __init__(self, confirm_hits: int = 1, conflict_override_hits: int = 3) -> None:
        self.confirm_hits = max(1, int(confirm_hits))
        self.conflict_override_hits = max(self.confirm_hits + 1, int(conflict_override_hits))
        self.cells: Dict[Cell, Dict[int, int]] = {}
        self.visits: Dict[Cell, int] = {}
        self.evidence: Dict[Tuple[Cell, int], Dict[int, int]] = {}

    def ensure_cell(self, cell: Cell) -> None:
        if cell not in self.cells:
            self.cells[cell] = {direction: UNKNOWN for direction in DIRECTIONS}
            self.visits[cell] = 0

    def mark_visited(self, cell: Cell) -> None:
        self.ensure_cell(cell)
        self.visits[cell] += 1

    def is_visited(self, cell: Cell) -> bool:
        return self.visits.get(cell, 0) > 0

    def visited_count(self) -> int:
        return sum(1 for count in self.visits.values() if count > 0)

    def wall_state(self, cell: Cell, direction: int) -> int:
        self.ensure_cell(cell)
        return self.cells[cell].get(direction, UNKNOWN)

    def update_edge(self, cell: Cell, direction: int, observed_state: int) -> List[str]:
        if observed_state == UNKNOWN:
            return []

        self.ensure_cell(cell)
        messages = self._update_one_side(cell, direction, observed_state)

        other = neighbor(cell, direction)

        if observed_state == OPEN:
            self.ensure_cell(other)
            messages.extend(self._update_one_side(other, OPPOSITE[direction], OPEN))
        elif other in self.cells:
            messages.extend(self._update_one_side(other, OPPOSITE[direction], WALL))

        return messages

    def _update_one_side(self, cell: Cell, direction: int, observed_state: int) -> List[str]:
        key = (cell, direction)
        if key not in self.evidence:
            self.evidence[key] = {OPEN: 0, WALL: 0}

        self.evidence[key][observed_state] += 1
        votes = self.evidence[key]
        committed = self.cells[cell][direction]

        if committed == UNKNOWN:
            if votes[WALL] >= self.confirm_hits:
                self.cells[cell][direction] = WALL
            elif votes[OPEN] >= self.confirm_hits:
                self.cells[cell][direction] = OPEN
            return []

        if committed == observed_state:
            return []

        if votes[observed_state] >= self.conflict_override_hits:
            old = committed
            self.cells[cell][direction] = observed_state
            return [
                f'overrode {cell} {DIR_NAME[direction]} from '
                f'{STATE_NAME[old]} to {STATE_NAME[observed_state]}'
            ]

        return [
            f'kept {cell} {DIR_NAME[direction]}={STATE_NAME[committed]} '
            f'despite {STATE_NAME[observed_state]} evidence '
            f'({votes[observed_state]}/{self.conflict_override_hits})'
        ]

    def open_neighbors(self, cell: Cell) -> List[Tuple[int, Cell]]:
        self.ensure_cell(cell)
        result = []
        for direction in DIRECTIONS:
            if self.cells[cell].get(direction, UNKNOWN) == OPEN:
                result.append((direction, neighbor(cell, direction)))
        return result

    def has_unvisited_open_neighbor(self, cell: Cell) -> bool:
        for _, next_cell in self.open_neighbors(cell):
            if not self.is_visited(next_cell):
                return True
        return False

    def unresolved_directions(self, cell: Cell) -> List[int]:
        self.ensure_cell(cell)
        return [
            direction for direction in DIRECTIONS
            if self.cells[cell].get(direction, UNKNOWN) == UNKNOWN
        ]

    def nearest_frontier_path(self, start: Cell) -> Optional[List[Cell]]:
        if self.has_unvisited_open_neighbor(start):
            return [start]

        queue = deque([start])
        previous: Dict[Cell, Optional[Cell]] = {start: None}

        while queue:
            cell = queue.popleft()
            for _, next_cell in self.open_neighbors(cell):
                if next_cell in previous:
                    continue

                previous[next_cell] = cell

                if self.has_unvisited_open_neighbor(next_cell):
                    path = [next_cell]
                    cursor = cell
                    while cursor is not None:
                        path.append(cursor)
                        cursor = previous[cursor]
                    path.reverse()
                    return path

                queue.append(next_cell)

        return None

    def export_ros_maze(
        self,
        start_cell: Cell,
        current_cell: Cell,
        start_orientation: int,
        unknown_as_wall: bool,
    ) -> Tuple[RosMaze, Dict[str, int]]:
        if not self.cells:
            self.ensure_cell(start_cell)

        xs = [cell[0] for cell in self.cells]
        ys = [cell[1] for cell in self.cells]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        n = max_x - min_x + 1
        m = max_y - min_y + 1
        flattened = [15 for _ in range(n * m)]

        unknown_edges = 0
        open_edges = 0
        wall_edges = 0

        for cell, walls in self.cells.items():
            i = cell[0] - min_x + 1
            j = cell[1] - min_y + 1
            index = (j - 1) * n + i

            bits = 0
            for direction in DIRECTIONS:
                state = walls.get(direction, UNKNOWN)
                if state == WALL:
                    bits |= direction
                    wall_edges += 1
                elif state == OPEN:
                    open_edges += 1
                else:
                    unknown_edges += 1
                    if unknown_as_wall:
                        bits |= direction

            flattened[index - 1] = bits

        def to_index(cell: Cell) -> int:
            i = cell[0] - min_x + 1
            j = cell[1] - min_y + 1
            return (j - 1) * n + i

        msg = RosMaze()
        msg.n = int(n)
        msg.m = int(m)
        msg.start_idx = int(to_index(start_cell))
        msg.end_idx = int(to_index(current_cell))
        msg.start_orientation = int(start_orientation)
        msg.l = [int(value) for value in flattened]

        stats = {
            'n': n,
            'm': m,
            'cells': len(self.cells),
            'visited_cells': self.visited_count(),
            'start_idx': int(msg.start_idx),
            'end_idx': int(msg.end_idx),
            'unknown_edges': unknown_edges,
            'open_edges': open_edges,
            'wall_edges': wall_edges,
        }
        return msg, stats

    def ascii_summary(self) -> str:
        if not self.cells:
            return '<empty>'

        min_x = min(cell[0] for cell in self.cells)
        max_x = max(cell[0] for cell in self.cells)
        min_y = min(cell[1] for cell in self.cells)
        max_y = max(cell[1] for cell in self.cells)

        rows = []
        for y in range(max_y, min_y - 1, -1):
            chars = []
            for x in range(min_x, max_x + 1):
                cell = (x, y)
                if cell not in self.cells:
                    chars.append(' ')
                elif self.is_visited(cell):
                    chars.append('V')
                else:
                    chars.append('.')
            rows.append(''.join(chars))
        return '\n'.join(rows)


class MazeExplorerNode(Node):
    def __init__(self) -> None:
        super().__init__('maze_explorer_node')

        self.declare_parameter('scan_topic', '/ldlidar_node/scan')
        self.declare_parameter('motion_action_name', '/darth_maul_control/execute_motion_primitive')
        self.declare_parameter('cell_length_m', 0.254)
        self.declare_parameter('start_heading', POS_X)

        self.declare_parameter('sector_width_deg', 24.0)
        self.declare_parameter('sector_min_samples', 3)
        self.declare_parameter('wall_threshold_m', 0.18)
        self.declare_parameter('open_threshold_m', 0.30)
        self.declare_parameter('stale_scan_timeout_s', 0.8)

        self.declare_parameter('wall_confirm_hits', 1)
        self.declare_parameter('conflict_override_hits', 3)
        self.declare_parameter('unknown_as_wall_on_export', True)

        self.declare_parameter('max_cells_to_visit', 0)
        self.declare_parameter('shutdown_on_complete', True)
        self.declare_parameter('output_maze_file', '')

        self.declare_parameter('drive_max_linear_x_mps', 0.18)
        self.declare_parameter('reverse_max_linear_x_mps', 0.075)
        self.declare_parameter('reverse_backtracking_enabled', True)
        self.declare_parameter('reverse_position_tolerance_m', 0.018)
        self.declare_parameter('reverse_heading_tolerance_rad', 0.180)
        self.declare_parameter('rotate_max_angular_z_radps', 0.85)
        self.declare_parameter('motion_timeout_s', 0.0)
        self.declare_parameter('direction_priority', 'left_straight_right_back')

        self.scan_topic = str(self.get_parameter('scan_topic').value).strip()
        self.motion_action_name = str(self.get_parameter('motion_action_name').value).strip()
        self.cell_length_m = float(self.get_parameter('cell_length_m').value)
        self.start_heading = int(self.get_parameter('start_heading').value)

        self.sector_width_deg = float(self.get_parameter('sector_width_deg').value)
        self.sector_min_samples = int(self.get_parameter('sector_min_samples').value)
        self.wall_threshold_m = float(self.get_parameter('wall_threshold_m').value)
        self.open_threshold_m = float(self.get_parameter('open_threshold_m').value)
        self.stale_scan_timeout_s = float(self.get_parameter('stale_scan_timeout_s').value)

        self.unknown_as_wall_on_export = parse_bool(
            self.get_parameter('unknown_as_wall_on_export').value
        )
        self.max_cells_to_visit = int(self.get_parameter('max_cells_to_visit').value)
        self.shutdown_on_complete = parse_bool(self.get_parameter('shutdown_on_complete').value)
        self.output_maze_file = str(self.get_parameter('output_maze_file').value).strip()

        self.drive_max_linear_x_mps = float(self.get_parameter('drive_max_linear_x_mps').value)
        self.reverse_max_linear_x_mps = float(
            self.get_parameter('reverse_max_linear_x_mps').value
        )
        self.reverse_backtracking_enabled = parse_bool(
            self.get_parameter('reverse_backtracking_enabled').value
        )
        self.reverse_position_tolerance_m = float(
            self.get_parameter('reverse_position_tolerance_m').value
        )
        self.reverse_heading_tolerance_rad = float(
            self.get_parameter('reverse_heading_tolerance_rad').value
        )
        self.rotate_max_angular_z_radps = float(
            self.get_parameter('rotate_max_angular_z_radps').value
        )
        self.motion_timeout_s = float(self.get_parameter('motion_timeout_s').value)
        self.direction_priority = str(self.get_parameter('direction_priority').value).strip()

        self._validate_parameters()

        self.maze = SparseMazeMap(
            confirm_hits=int(self.get_parameter('wall_confirm_hits').value),
            conflict_override_hits=int(self.get_parameter('conflict_override_hits').value),
        )

        self.start_cell: Cell = (0, 0)
        self.current_cell: Cell = self.start_cell
        self.heading = int(self.start_heading)

        self.latest_scan: Optional[LaserScan] = None
        self.latest_scan_monotonic = 0.0

        self.motion_queue: Deque[MotionStep] = deque()
        self.motion_in_flight = False
        self.active_step: Optional[MotionStep] = None

        self.completed = False
        self.shutdown_requested = False
        self.exit_code = 0

        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self._scan_callback, 10)
        self.motion_client = ActionClient(self, ExecuteMotionPrimitive, self.motion_action_name)
        self.maze_pub = self.create_publisher(RosMaze, '/discovered_maze', 10)
        self.timer = self.create_timer(0.10, self._tick)

        self.get_logger().info(
            'Maze explorer initialized: '
            f'scan={self.scan_topic}, cell={self.cell_length_m:.3f} m, '
            f'heading={DIR_NAME[self.heading]}, '
            f'wall<={self.wall_threshold_m:.3f}, open>={self.open_threshold_m:.3f}'
        )

    def _validate_parameters(self) -> None:
        errors = []
        valid_priorities = {
            'left_straight_right_back',
            'straight_left_right_back',
            'right_straight_left_back',
            'straight_right_left_back',
        }

        if self.start_heading not in DIRECTIONS:
            errors.append(f'start_heading={self.start_heading} invalid; use 1, 2, 4, or 8')
        if not self.scan_topic:
            errors.append('scan_topic must not be empty')
        if not self.motion_action_name:
            errors.append('motion_action_name must not be empty')
        if not math.isfinite(self.cell_length_m) or self.cell_length_m <= 0.0:
            errors.append('cell_length_m must be finite and > 0')
        if self.wall_threshold_m <= 0.0:
            errors.append('wall_threshold_m must be > 0')
        if self.open_threshold_m <= self.wall_threshold_m:
            errors.append('open_threshold_m must be greater than wall_threshold_m')
        if self.sector_width_deg <= 0.0 or self.sector_width_deg > 90.0:
            errors.append('sector_width_deg must be in (0, 90]')
        if self.sector_min_samples < 1:
            errors.append('sector_min_samples must be >= 1')
        if self.max_cells_to_visit < 0:
            errors.append('max_cells_to_visit must be >= 0')
        if self.stale_scan_timeout_s <= 0.0:
            errors.append('stale_scan_timeout_s must be > 0')
        if self.drive_max_linear_x_mps <= 0.0:
            errors.append('drive_max_linear_x_mps must be > 0')
        if self.reverse_max_linear_x_mps <= 0.0:
            errors.append('reverse_max_linear_x_mps must be > 0')
        if self.reverse_position_tolerance_m <= 0.0:
            errors.append('reverse_position_tolerance_m must be > 0')
        if self.reverse_heading_tolerance_rad <= 0.0:
            errors.append('reverse_heading_tolerance_rad must be > 0')
        if self.rotate_max_angular_z_radps <= 0.0:
            errors.append('rotate_max_angular_z_radps must be > 0')
        if self.motion_timeout_s < 0.0:
            errors.append('motion_timeout_s must be >= 0')
        if self.direction_priority not in valid_priorities:
            errors.append(
                'direction_priority must be one of '
                + ', '.join(sorted(valid_priorities))
            )

        if errors:
            message = '; '.join(errors)
            self.get_logger().error(message)
            raise ValueError(message)

    def _scan_callback(self, msg: LaserScan) -> None:
        self.latest_scan = msg
        self.latest_scan_monotonic = time.monotonic()

    def _tick(self) -> None:
        if self.shutdown_requested or self.completed or self.motion_in_flight:
            return

        if self.motion_queue:
            self._send_next_motion_step()
            return

        if not self._scan_is_fresh():
            self.get_logger().warn('Waiting for fresh scan.', throttle_duration_sec=2.0)
            return

        if not self.motion_client.server_is_ready():
            if not self.motion_client.wait_for_server(timeout_sec=0.05):
                self.get_logger().warn(
                    f'Waiting for motion action server {self.motion_action_name}.',
                    throttle_duration_sec=2.0,
                )
                return

        self._exploration_step()

    def _scan_is_fresh(self) -> bool:
        return (
            self.latest_scan is not None
            and time.monotonic() - self.latest_scan_monotonic <= self.stale_scan_timeout_s
        )

    def _exploration_step(self) -> None:
        self._observe_and_update_current_cell()

        if (
            self.max_cells_to_visit > 0
            and self.maze.visited_count() >= self.max_cells_to_visit
        ):
            self._finish_exploration(f'max_cells_to_visit={self.max_cells_to_visit} reached')
            return

        direction = self._choose_unvisited_open_neighbor(self.current_cell)
        if direction is not None:
            self._enqueue_step(direction)
            return

        path = self.maze.nearest_frontier_path(self.current_cell)
        if path is None:
            self._finish_exploration('no reachable frontier remains')
            return

        if len(path) < 2:
            self._finish_exploration('frontier path unexpectedly empty')
            return

        self.get_logger().info(f'Backtracking to frontier through path: {path}')
        self._enqueue_path(path)

    def _observe_and_update_current_cell(self) -> None:
        if self.latest_scan is None:
            return

        self.maze.mark_visited(self.current_cell)
        observations = self._observe_walls(self.latest_scan)

        parts = []
        for direction, state, distance, reason in observations:
            for message in self.maze.update_edge(self.current_cell, direction, state):
                self.get_logger().warn(message)

            parts.append(
                f'{DIR_NAME[direction]}={STATE_NAME[state]}'
                f'({distance:.3f} m; {reason})'
            )

        self._publish_current_maze()

        self.get_logger().info(
            f'Cell={self.current_cell}, heading={DIR_NAME[self.heading]}, '
            f'visit={self.maze.visits[self.current_cell]}: '
            + '; '.join(parts)
        )

    def _observe_walls(self, scan: LaserScan) -> List[Tuple[int, int, float, str]]:
        specs = [
            ('front', 0.0, self.heading),
            ('left', math.pi / 2.0, LEFT_OF[self.heading]),
            ('right', -math.pi / 2.0, RIGHT_OF[self.heading]),
            ('rear', math.pi, OPPOSITE[self.heading]),
        ]

        observations = []
        for name, angle, direction in specs:
            values = valid_ranges_in_sector(scan, angle, self.sector_width_deg)
            state, distance, reason = self._classify_sector(name, values)
            observations.append((direction, state, distance, reason))

        return observations

    def _classify_sector(self, name: str, values: Sequence[float]) -> Tuple[int, float, str]:
        if len(values) < self.sector_min_samples:
            return UNKNOWN, float('nan'), f'{name}: samples {len(values)} < {self.sector_min_samples}'

        distance = float(median(values))

        if distance <= self.wall_threshold_m:
            return WALL, distance, f'{name}: median <= wall threshold'

        if distance >= self.open_threshold_m:
            return OPEN, distance, f'{name}: median >= open threshold'

        return UNKNOWN, distance, f'{name}: ambiguous between thresholds'

    def _choose_unvisited_open_neighbor(self, cell: Cell) -> Optional[int]:
        for direction in self._ordered_directions():
            if self.maze.wall_state(cell, direction) != OPEN:
                continue

            next_cell = neighbor(cell, direction)
            if not self.maze.is_visited(next_cell):
                self.get_logger().info(
                    f'Choosing frontier: {cell} -> {next_cell} via {DIR_NAME[direction]}'
                )
                return direction

        unresolved = self.maze.unresolved_directions(cell)
        if unresolved:
            self.get_logger().warn(
                f'No open unvisited edge from {cell}; unresolved='
                + ','.join(DIR_NAME[d] for d in unresolved)
            )

        return None

    def _ordered_directions(self) -> List[int]:
        if self.direction_priority == 'straight_left_right_back':
            return [self.heading, LEFT_OF[self.heading], RIGHT_OF[self.heading], OPPOSITE[self.heading]]
        if self.direction_priority == 'right_straight_left_back':
            return [RIGHT_OF[self.heading], self.heading, LEFT_OF[self.heading], OPPOSITE[self.heading]]
        if self.direction_priority == 'straight_right_left_back':
            return [self.heading, RIGHT_OF[self.heading], LEFT_OF[self.heading], OPPOSITE[self.heading]]
        return [LEFT_OF[self.heading], self.heading, RIGHT_OF[self.heading], OPPOSITE[self.heading]]

    def _enqueue_path(self, path: Sequence[Cell]) -> None:
        if len(path) < 2:
            self._fatal(f'Cannot enqueue path with fewer than 2 cells: {path}')
            return

        simulated_cell = self.current_cell
        simulated_heading = self.heading

        if path[0] != simulated_cell:
            self._fatal(
                f'Backtracking path does not start at current cell: '
                f'current={self.current_cell}, path={path}'
            )
            return

        for target_cell in path[1:]:
            direction = direction_between(simulated_cell, target_cell)
            simulated_heading = self._enqueue_segment(
                from_cell=simulated_cell,
                from_heading=simulated_heading,
                direction=direction,
                target_cell=target_cell,
            )
            simulated_cell = target_cell

    def _enqueue_step(self, direction: int) -> None:
        target_cell = neighbor(self.current_cell, direction)
        self._enqueue_segment(
            from_cell=self.current_cell,
            from_heading=self.heading,
            direction=direction,
            target_cell=target_cell,
        )

    def _enqueue_segment(
        self,
        *,
        from_cell: Cell,
        from_heading: int,
        direction: int,
        target_cell: Cell,
    ) -> int:
        expected_target = neighbor(from_cell, direction)
        if target_cell != expected_target:
            self._fatal(
                f'Invalid segment target: from={from_cell}, '
                f'direction={DIR_NAME[direction]}, '
                f'target={target_cell}, expected={expected_target}'
            )
            return from_heading

        should_reverse = bool(
            self.reverse_backtracking_enabled
            and direction == OPPOSITE[from_heading]
        )

        if should_reverse:
            self.motion_queue.append(
                MotionStep('drive_backward', direction, float(self.cell_length_m), target_cell)
            )

            self.get_logger().info(
                f'Queued reverse segment {from_cell} -> {target_cell} '
                f'via {DIR_NAME[direction]} '
                f'(heading stays {DIR_NAME[from_heading]}, '
                f'drive_backward={self.cell_length_m:.3f})'
            )

            return from_heading

        turn = turn_between(from_heading, direction)

        if abs(turn) > 1.0e-6:
            self.motion_queue.append(MotionStep('rotate', direction, turn, None))

        self.motion_queue.append(
            MotionStep('drive_forward', direction, float(self.cell_length_m), target_cell)
        )

        self.get_logger().info(
            f'Queued segment {from_cell} -> {target_cell} via {DIR_NAME[direction]} '
            f'(from_heading={DIR_NAME[from_heading]}, turn={turn:.3f}, '
            f'drive={self.cell_length_m:.3f})'
        )

        return direction

    def _send_next_motion_step(self) -> None:
        if self.motion_in_flight or not self.motion_queue:
            return

        step = self.motion_queue.popleft()
        self.active_step = step

        goal = ExecuteMotionPrimitive.Goal()

        if step.kind == 'rotate':
            goal.primitive_type = ExecuteMotionPrimitive.Goal.ROTATE_RELATIVE
            goal.value = float(step.value)
            goal.collision_check_enabled = False
            goal.max_linear_x_mps = 0.0
            goal.max_linear_y_mps = 0.0
            goal.max_angular_z_radps = float(self.rotate_max_angular_z_radps)
        elif step.kind == 'drive_forward':
            goal.primitive_type = ExecuteMotionPrimitive.Goal.DRIVE_FORWARD
            goal.value = float(step.value)
            goal.collision_check_enabled = True
            goal.max_linear_x_mps = float(self.drive_max_linear_x_mps)
            goal.max_linear_y_mps = 0.0
            goal.max_angular_z_radps = 0.0
        elif step.kind == 'drive_backward':
            goal.primitive_type = ExecuteMotionPrimitive.Goal.DRIVE_BACKWARD
            goal.value = float(step.value)
            goal.collision_check_enabled = False
            goal.max_linear_x_mps = float(self.reverse_max_linear_x_mps)
            goal.max_linear_y_mps = 0.0
            goal.max_angular_z_radps = 0.0
        else:
            self._fatal(f'Unknown motion step: {step}')
            return

        if step.kind == 'drive_backward':
            goal.position_tolerance_m = float(self.reverse_position_tolerance_m)
            goal.heading_tolerance_rad = float(self.reverse_heading_tolerance_rad)
        else:
            goal.position_tolerance_m = 0.0
            goal.heading_tolerance_rad = 0.0
        goal.timeout_s = float(self.motion_timeout_s)

        goal.grid_n = 0
        goal.grid_m = 0
        goal.grid_start_idx = 0
        goal.grid_heading = 0
        goal.grid_run_cells = 0
        goal.grid_l = []

        self.motion_in_flight = True

        self.get_logger().info(
            f'Sending {step.kind}: value={step.value:.3f}, '
            f'direction={DIR_NAME.get(step.direction, "none")}, '
            f'target={step.target_cell}'
        )

        future = self.motion_client.send_goal_async(
            goal,
            feedback_callback=self._handle_motion_feedback,
        )
        future.add_done_callback(self._handle_motion_goal_response)

    def _handle_motion_feedback(self, feedback_msg) -> None:
        feedback = feedback_msg.feedback
        self.get_logger().debug(
            f'Motion feedback: state={feedback.state}, progress={feedback.progress:.3f}, '
            f'remaining={feedback.distance_remaining_m:.3f}, '
            f'heading={feedback.heading_remaining_rad:.3f}, '
            f'front={feedback.front_clearance_m:.3f}'
        )

    def _handle_motion_goal_response(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self._fatal(f'Motion goal failed for {self.active_step}: {exc}')
            return

        if goal_handle is None:
            self._fatal(f'Motion goal returned no handle for {self.active_step}')
            return

        if not goal_handle.accepted:
            self._fatal(f'Motion goal rejected for {self.active_step}')
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._handle_motion_result)

    def _handle_motion_result(self, future) -> None:
        step = self.active_step

        try:
            wrapped_result = future.result()
            result = wrapped_result.result
        except Exception as exc:
            self._fatal(f'Motion result failed for {step}: {exc}')
            return

        self.get_logger().info(
            f'Motion result for {step}: success={result.success}, '
            f'code={result.result_code}, pos_error={result.final_position_error_m:.3f}, '
            f'heading_error={result.final_heading_error_rad:.3f}, message={result.message}'
        )

        if not result.success:
            self._fatal('Stopping exploration because motion failed.')
            return

        if step is not None:
            if step.kind == 'rotate':
                self.heading = int(step.direction)
            elif step.kind in ('drive_forward', 'drive_backward'):
                if step.target_cell is None:
                    self._fatal('Drive succeeded but target_cell is missing.')
                    return
                self.current_cell = step.target_cell

        self.active_step = None
        self.motion_in_flight = False

    def _publish_current_maze(self) -> None:
        msg, _ = self.maze.export_ros_maze(
            self.start_cell,
            self.current_cell,
            self.start_heading,
            self.unknown_as_wall_on_export,
        )
        self.maze_pub.publish(msg)

    def _finish_exploration(self, reason: str) -> None:
        if self.completed:
            return

        self.completed = True
        msg, stats = self.maze.export_ros_maze(
            self.start_cell,
            self.current_cell,
            self.start_heading,
            self.unknown_as_wall_on_export,
        )
        self.maze_pub.publish(msg)

        self.get_logger().info(
            'Exploration complete: '
            f'{reason}; n={stats["n"]}, m={stats["m"]}, cells={stats["cells"]}, '
            f'visited={stats["visited_cells"]}, start_idx={stats["start_idx"]}, '
            f'end_idx={stats["end_idx"]}, unknown_edges={stats["unknown_edges"]}, '
            f'open_edges={stats["open_edges"]}, wall_edges={stats["wall_edges"]}'
        )
        self.get_logger().info('Visited map:\n' + self.maze.ascii_summary())
        self.get_logger().info('Final L: ' + ','.join(str(int(v)) for v in msg.l))

        if self.output_maze_file:
            self._write_maze_file(Path(self.output_maze_file).expanduser(), msg)

        if self.shutdown_on_complete:
            self.shutdown_requested = True

    def _write_maze_file(self, path: Path, msg: RosMaze) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                str(int(msg.n)),
                str(int(msg.m)),
                str(int(msg.start_idx)),
                str(int(msg.end_idx)),
                str(int(msg.start_orientation)),
            ]
            lines.extend(str(int(value)) for value in msg.l)
            path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
            self.get_logger().info(f'Wrote discovered maze to {path}')
        except OSError as exc:
            self.get_logger().error(f'Failed to write discovered maze to {path}: {exc}')

    def _fatal(self, message: str) -> None:
        self.get_logger().error(message)
        self.motion_queue.clear()
        self.motion_in_flight = False
        self.active_step = None
        self.exit_code = 1
        self.shutdown_requested = True


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[MazeExplorerNode] = None
    exit_code = 0

    try:
        node = MazeExplorerNode()
        while rclpy.ok() and not node.shutdown_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
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
