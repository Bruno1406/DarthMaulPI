import math
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import rclpy
from darth_maul_control_interfaces.action import ExecuteMotionPrimitive
from maze_interface.srv import GetRosMaze
from rclpy.action import ActionClient
from rclpy.node import Node


@dataclass
class SearchNode:
    position: int
    parent: Optional['SearchNode'] = None
    g: float = 0.0
    h: float = 0.0

    @property
    def f(self) -> float:
        return self.g + self.h


VALID_ORIENTATIONS = {1, 2, 4, 8}


@dataclass(frozen=True)
class MotionCommand:
    name: str
    value: float
    start_idx: int = 0
    heading: int = 0
    run_cells: int = 0


def parse_bool_parameter(value: Any) -> bool:
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

    raise ValueError(f'Cannot parse boolean parameter value: {value!r}')


def validate_maze_fields(
    n: int,
    m: int,
    start_idx: int,
    end_idx: int,
    start_orientation: int,
    flattened_l: Sequence[int],
) -> Optional[str]:
    if n <= 0 or m <= 0:
        return f'invalid maze dimensions: n={n}, m={m}'

    expected_len = n * m
    if len(flattened_l) != expected_len:
        return (
            f'invalid maze length: len(l)={len(flattened_l)}, '
            f'expected n*m={expected_len}'
        )

    if start_idx < 1 or start_idx > expected_len:
        return (
            f'invalid start_idx={start_idx}; expected value in '
            f'[1, {expected_len}]'
        )

    if end_idx < 1 or end_idx > expected_len:
        return (
            f'invalid end_idx={end_idx}; expected value in '
            f'[1, {expected_len}]'
        )

    if start_orientation not in VALID_ORIENTATIONS:
        return (
            f'invalid start_orientation={start_orientation}; expected one of '
            f'{sorted(VALID_ORIENTATIONS)}'
        )

    for idx, value in enumerate(flattened_l, start=1):
        if value < 0 or value > 15:
            return f'invalid maze cell value at I={idx}: {value}; expected 0..15'

    if int(flattened_l[start_idx - 1]) == 15:
        return f'start_idx={start_idx} refers to a missing/unreachable cell'

    if int(flattened_l[end_idx - 1]) == 15:
        return f'end_idx={end_idx} refers to a missing/unreachable cell'

    return None


def validate_ros_maze(maze) -> Optional[str]:
    return validate_maze_fields(
        int(maze.n),
        int(maze.m),
        int(maze.start_idx),
        int(maze.end_idx),
        int(maze.start_orientation),
        [int(value) for value in maze.l],
    )


def validate_solver_parameters(
    maze_nr: int,
    cell_length_m: float,
    max_commands_to_execute: int,
    motion_server_timeout_s: float,
    maze_service_timeout_s: float,
    maze_service_name: str,
) -> Optional[str]:
    if maze_nr < -128 or maze_nr > 127:
        return f'maze_nr={maze_nr} is outside int8 range [-128, 127]'

    if not math.isfinite(cell_length_m) or cell_length_m <= 0.0:
        return (
            f'cell_length_m={cell_length_m!r} is invalid; '
            'expected a finite value > 0.0'
        )

    if max_commands_to_execute < 0:
        return (
            f'max_commands_to_execute={max_commands_to_execute} is invalid; '
            'expected an integer >= 0'
        )

    if (
        not math.isfinite(motion_server_timeout_s)
        or motion_server_timeout_s <= 0.0
    ):
        return (
            f'motion_server_timeout_s={motion_server_timeout_s!r} is invalid; '
            'expected a finite value > 0.0'
        )

    if (
        not math.isfinite(maze_service_timeout_s)
        or maze_service_timeout_s <= 0.0
    ):
        return (
            f'maze_service_timeout_s={maze_service_timeout_s!r} is invalid; '
            'expected a finite value > 0.0'
        )

    if not maze_service_name.strip():
        return 'maze_service_name must not be empty'

    if not maze_service_name.startswith('/'):
        return (
            f'maze_service_name={maze_service_name!r} is invalid for exam use; '
            'expected an absolute service name such as /get_ros_maze'
        )

    return None


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def orientation_angle(orientation: int) -> float:
    angles = {
        1: 0.0,
        4: math.pi / 2.0,
        2: math.pi,
        8: -math.pi / 2.0,
    }
    if orientation not in angles:
        raise ValueError(f'Invalid orientation: {orientation}')
    return angles[orientation]


def turn_between_orientations(current: int, target: int) -> float:
    return normalize_angle(orientation_angle(target) - orientation_angle(current))


def build_motion_commands(
    start_orientation: int,
    orientations: List[int],
    cell_length_m: float,
    path: Optional[List[int]] = None,
) -> List[MotionCommand]:
    """Build known-maze motion commands using maximal straight-run compression.

    If path is provided, drive_forward commands include grid metadata:
    start_idx, heading, and run_cells.
    """
    commands: List[MotionCommand] = []

    if not orientations:
        return commands

    if path is not None and len(path) != len(orientations) + 1:
        raise ValueError(
            f'path length must equal len(orientations)+1; '
            f'got len(path)={len(path)}, len(orientations)={len(orientations)}'
        )

    current_orientation = int(start_orientation)
    index = 0

    while index < len(orientations):
        run_orientation = int(orientations[index])
        run_length = 1

        while (
            index + run_length < len(orientations)
            and int(orientations[index + run_length]) == run_orientation
        ):
            run_length += 1

        turn = turn_between_orientations(current_orientation, run_orientation)
        if abs(turn) > 1.0e-6:
            rotate_start_idx = int(path[index]) if path is not None else 0
            commands.append(
                MotionCommand(
                    'rotate',
                    float(turn),
                    start_idx=rotate_start_idx,
                    heading=run_orientation,
                    run_cells=1,
                )
            )

        distance_m = float(run_length) * float(cell_length_m)
        start_idx = int(path[index]) if path is not None else 0
        commands.append(
            MotionCommand(
                'drive_forward',
                distance_m,
                start_idx=start_idx,
                heading=run_orientation,
                run_cells=run_length,
            )
        )

        current_orientation = run_orientation
        index += run_length

    return commands


class MazeSolverNode(Node):
    def __init__(self):
        super().__init__('maze_solver_node')
        self.shutdown_requested = False
        self.exit_code = 0

        self.declare_parameter('maze_nr', 1)
        self.declare_parameter('cell_length_m', 0.254)
        self.declare_parameter('max_commands_to_execute', 0)
        self.declare_parameter('execute_motions', True)
        self.declare_parameter('motion_server_timeout_s', 5.0)
        self.declare_parameter('maze_service_name', '/get_ros_maze')
        self.declare_parameter('maze_service_timeout_s', 15.0)
        self.declare_parameter('shutdown_on_fatal_error', True)

        self.maze_nr = int(self.get_parameter('maze_nr').value)
        self.cell_length_m = float(self.get_parameter('cell_length_m').value)
        self.max_commands_to_execute = int(
            self.get_parameter('max_commands_to_execute').value
        )
        self.execute_motions = parse_bool_parameter(
            self.get_parameter('execute_motions').value
        )
        self.motion_server_timeout_s = float(
            self.get_parameter('motion_server_timeout_s').value
        )
        self.maze_service_name = str(
            self.get_parameter('maze_service_name').value
        ).strip()

        self.maze_service_timeout_s = float(
            self.get_parameter('maze_service_timeout_s').value
        )
        self.shutdown_on_fatal_error = parse_bool_parameter(
            self.get_parameter('shutdown_on_fatal_error').value
        )

        parameter_error = validate_solver_parameters(
            maze_nr=self.maze_nr,
            cell_length_m=self.cell_length_m,
            max_commands_to_execute=self.max_commands_to_execute,
            motion_server_timeout_s=self.motion_server_timeout_s,
            maze_service_timeout_s=self.maze_service_timeout_s,
            maze_service_name=self.maze_service_name,
        )
        if parameter_error is not None:
            raise ValueError(parameter_error)

        self.get_logger().info(
            f'maze_nr={self.maze_nr}, '
            f'cell_length_m={self.cell_length_m:.3f}, '
            f'max_commands_to_execute={self.max_commands_to_execute}, '
            f'execute_motions={self.execute_motions}, '
            f'maze_service_name={self.maze_service_name}, '
            f'maze_service_timeout_s={self.maze_service_timeout_s:.1f}, '
            f'shutdown_on_fatal_error={self.shutdown_on_fatal_error}'
        )

        self.maze_client = self.create_client(GetRosMaze, self.maze_service_name)
        self.motion_client = ActionClient(
            self,
            ExecuteMotionPrimitive,
            '/darth_maul_control/execute_motion_primitive',
        )

        self.command_queue: List[MotionCommand] = []
        self.current_command: Optional[MotionCommand] = None
        self.current_maze_n = 0
        self.current_maze_m = 0
        self.current_maze_l = []

        self._request_maze()

    def _fatal(self, message: str) -> None:
        self.get_logger().error(message)

        if self.shutdown_on_fatal_error:
            self.exit_code = 1
            self.shutdown_requested = True

    def _finish_successfully(self, message: str) -> None:
        self.get_logger().info(message)
        self.exit_code = 0
        self.shutdown_requested = True

    def _request_maze(self) -> None:
        deadline = None
        if self.maze_service_timeout_s > 0.0:
            deadline = time.monotonic() + self.maze_service_timeout_s

        while not self.maze_client.wait_for_service(0.5):
            if deadline is not None and time.monotonic() >= deadline:
                self._fatal(
                    f'Maze service {self.maze_service_name!r} was not available '
                    f'within {self.maze_service_timeout_s:.1f} s. '
                    'Check that the teacher-provided maze server is running, '
                    'that ROS_DOMAIN_ID matches, and that this node is not '
                    'namespaced away from /get_ros_maze.'
                )
                return

            self.get_logger().info(
                f'Waiting for maze service {self.maze_service_name!r}...'
            )

        request = GetRosMaze.Request()
        request.maze_nr = self.maze_nr

        self.get_logger().info(
            f'Requesting maze_nr={self.maze_nr} from {self.maze_service_name!r}'
        )

        future = self.maze_client.call_async(request)
        future.add_done_callback(self._handle_maze_response)

    def _handle_maze_response(self, future) -> None:
        try:
            response = future.result()
        except Exception as exc:
            self._fatal(f'Failed to call {self.maze_service_name!r}: {exc}')
            return

        maze = response.maze

        self.get_logger().info(f'n: {maze.n}')
        self.get_logger().info(f'm: {maze.m}')
        self.get_logger().info(f'start_idx: {maze.start_idx}')
        self.get_logger().info(f'end_idx: {maze.end_idx}')
        self.get_logger().info(f'start_orientation: {maze.start_orientation}')
        self.get_logger().info(f'l length: {len(maze.l)}')

        validation_error = validate_ros_maze(maze)
        if validation_error is not None:
            self._fatal(
                f'Received invalid maze from {self.maze_service_name!r}: '
                f'{validation_error}'
            )
            return

        self.get_logger().info('Received valid maze.')
        self.current_maze_n = int(maze.n)
        self.current_maze_m = int(maze.m)
        self.current_maze_l = [int(value) for value in maze.l]

        grid = self._ros_maze_to_matrix(maze)
        graph = self._build_graph(grid)

        path = self._astar_path(graph, int(maze.start_idx), int(maze.end_idx), int(maze.n))
        self.get_logger().info(f'Path: {path}')

        if not path:
            self._fatal(
                f'No path found from {maze.start_idx} to {maze.end_idx}.'
            )
            return

        try:
            orientations = self._path_to_orientations(path, int(maze.n))
        except ValueError as exc:
            self._fatal(str(exc))
            return

        self.get_logger().info(f'Orientations: {orientations}')

        try:
            commands = self._orientations_to_commands(
                int(maze.start_orientation),
                orientations,
                path,
            )
        except ValueError as exc:
            self._fatal(str(exc))
            return

        if self.max_commands_to_execute > 0:
            original_count = len(commands)
            commands = commands[:self.max_commands_to_execute]
            self.get_logger().warn(
                f'max_commands_to_execute={self.max_commands_to_execute}; '
                f'truncated command list from {original_count} to '
                f'{len(commands)} commands.'
            )

        self.get_logger().info(f'Path cells: {len(path)}')
        self.get_logger().info(f'Path edges: {len(orientations)}')
        self.get_logger().info(
            f'Built {len(commands)} motion commands using maximal straight-run compression'
        )
        for command_index, motion_command in enumerate(commands, start=1):
            self.get_logger().info(
                f'Motion command {command_index}/{len(commands)}: '
                f'{motion_command.name} {motion_command.value:.3f}, '
                f'start_idx={motion_command.start_idx}, '
                f'heading={motion_command.heading}, '
                f'run_cells={motion_command.run_cells}'
            )

        if not self.execute_motions:
            self._finish_successfully('execute_motions=false; dry run complete.')
            return

        self._execute_commands(commands)

    def _ros_maze_to_matrix(self, maze):
        grid = []
        for i in range(int(maze.n)):
            row = []
            for j in range(int(maze.m)):
                row.append(int(maze.l[j * int(maze.n) + i]))
            grid.append(row)
        return grid

    def _build_graph(self, grid):
        # Local implementation to avoid package-layout ambiguity.
        # Maze wall bits:
        # 1 = +x wall
        # 2 = -x wall
        # 4 = +y wall
        # 8 = -y wall
        # 15 = missing/unreachable cell
        n = len(grid)
        m = len(grid[0]) if n > 0 else 0
        graph = {}

        def idx(i, j):
            return j * n + i + 1

        for i in range(n):
            for j in range(m):
                cell = grid[i][j]
                if cell == 15:
                    continue

                current = idx(i, j)
                neighbors = []

                if not (cell & 1) and i + 1 < n and grid[i + 1][j] != 15:
                    neighbors.append(idx(i + 1, j))

                if not (cell & 2) and i - 1 >= 0 and grid[i - 1][j] != 15:
                    neighbors.append(idx(i - 1, j))

                if not (cell & 4) and j + 1 < m and grid[i][j + 1] != 15:
                    neighbors.append(idx(i, j + 1))

                if not (cell & 8) and j - 1 >= 0 and grid[i][j - 1] != 15:
                    neighbors.append(idx(i, j - 1))

                graph[current] = neighbors

        return graph

    def _astar_path(self, graph, start_idx, end_idx, n):
        start = SearchNode(position=start_idx)
        open_list = [start]
        closed = set()
        best_g = {start_idx: 0.0}

        while open_list:
            current = min(open_list, key=lambda node: node.f)
            open_list.remove(current)

            if current.position == end_idx:
                path = []
                node = current
                while node is not None:
                    path.append(node.position)
                    node = node.parent
                return list(reversed(path))

            closed.add(current.position)

            for neighbor in graph.get(current.position, []):
                if neighbor in closed:
                    continue

                tentative_g = current.g + 1.0
                if tentative_g >= best_g.get(neighbor, float('inf')):
                    continue

                best_g[neighbor] = tentative_g
                child = SearchNode(
                    position=neighbor,
                    parent=current,
                    g=tentative_g,
                    h=self._manhattan_distance(neighbor, end_idx, n),
                )
                open_list.append(child)

        return []

    def _manhattan_distance(self, current_idx, target_idx, n):
        current_i, current_j = self._I_to_ij(current_idx, n)
        target_i, target_j = self._I_to_ij(target_idx, n)
        return abs(current_i - target_i) + abs(current_j - target_j)

    @staticmethod
    def _I_to_ij(index, n):
        # Returns zero-based i, j.
        i = (index - 1) % n
        j = (index - 1) // n
        return i, j

    def _path_to_orientations(self, path, n):
        orientations = []

        for current, nxt in zip(path, path[1:]):
            diff = nxt - current

            if diff == 1:
                orientations.append(1)   # +x
            elif diff == -1:
                orientations.append(2)   # -x
            elif diff == n:
                orientations.append(4)   # +y
            elif diff == -n:
                orientations.append(8)   # -y
            else:
                raise ValueError(f'{current} and {nxt} are not neighboring cells')

        return orientations

    def _orientations_to_commands(self, start_orientation, orientations, path):
        return build_motion_commands(
            int(start_orientation),
            list(orientations),
            self.cell_length_m,
            path=list(path),
        )

    def _turn_between_orientations(self, current, target):
        return turn_between_orientations(int(current), int(target))

    def _execute_commands(self, commands):
        self.command_queue = list(commands)

        if not self.command_queue:
            self._finish_successfully('No motion commands to execute.')
            return

        self.get_logger().info(f'Executing {len(self.command_queue)} motion commands.')

        if not self.motion_client.wait_for_server(timeout_sec=self.motion_server_timeout_s):
            self._fatal(
                'Motion action server /darth_maul_control/execute_motion_primitive '
                f'was not available within {self.motion_server_timeout_s:.1f} s.'
            )
            return

        self._send_next_command()

    def _send_next_command(self):
        if not self.command_queue:
            self._finish_successfully('All motion commands finished.')
            return

        motion_command = self.command_queue.pop(0)
        command = motion_command.name
        value = motion_command.value
        self.current_command = motion_command

        goal = ExecuteMotionPrimitive.Goal()

        if command == 'drive_forward':
            goal.primitive_type = ExecuteMotionPrimitive.Goal.DRIVE_FORWARD
            goal.value = float(value)
            goal.collision_check_enabled = True
            goal.grid_n = int(self.current_maze_n)
            goal.grid_m = int(self.current_maze_m)
            goal.grid_start_idx = int(motion_command.start_idx)
            goal.grid_heading = int(motion_command.heading)
            goal.grid_robot_heading = int(motion_command.heading)
            goal.grid_run_cells = int(motion_command.run_cells)
            goal.grid_l = list(self.current_maze_l)
        elif command == 'rotate':
            goal.primitive_type = ExecuteMotionPrimitive.Goal.ROTATE_RELATIVE
            goal.value = float(value)
            goal.collision_check_enabled = False
            if (
                self.current_maze_n > 0
                and self.current_maze_m > 0
                and motion_command.start_idx > 0
                and motion_command.heading in VALID_ORIENTATIONS
                and self.current_maze_l
            ):
                goal.grid_n = int(self.current_maze_n)
                goal.grid_m = int(self.current_maze_m)
                goal.grid_start_idx = int(motion_command.start_idx)
                goal.grid_heading = int(motion_command.heading)
                goal.grid_robot_heading = int(motion_command.heading)
                goal.grid_run_cells = 1
                goal.grid_l = list(self.current_maze_l)
        else:
            self._fatal(f'Unknown motion command: {command}')
            return

        goal.position_tolerance_m = 0.0
        goal.heading_tolerance_rad = 0.0
        goal.max_linear_x_mps = 0.0
        goal.max_linear_y_mps = 0.0
        goal.max_angular_z_radps = 0.0
        goal.timeout_s = 0.0

        self.get_logger().info(
            f'Sending motion command: {command}, {value:.3f}, '
            f'start_idx={motion_command.start_idx}, '
            f'heading={motion_command.heading}, '
            f'run_cells={motion_command.run_cells}'
        )
        future = self.motion_client.send_goal_async(
            goal,
            feedback_callback=self._handle_motion_feedback,
        )
        future.add_done_callback(self._handle_motion_goal_response)

    def _handle_motion_feedback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().debug(
            f'Motion feedback: state={feedback.state}, '
            f'progress={feedback.progress:.2f}, '
            f'remaining={feedback.distance_remaining_m:.3f}, '
            f'heading={feedback.heading_remaining_rad:.3f}, '
            f'front={feedback.front_clearance_m:.3f}'
        )

    def _handle_motion_goal_response(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self._fatal(
                f'Motion goal request failed for {self.current_command}: {exc}'
            )
            return

        if goal_handle is None:
            self._fatal(
                f'Motion goal request returned no goal handle for {self.current_command}'
            )
            return

        if not goal_handle.accepted:
            self._fatal(f'Motion goal rejected: {self.current_command}')
            return

        self.get_logger().info(f'Motion goal accepted: {self.current_command}')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._handle_motion_result)

    def _handle_motion_result(self, future):
        try:
            wrapped_result = future.result()
            result = wrapped_result.result
        except Exception as exc:
            self._fatal(
                f'Motion result failed for {self.current_command}: {exc}'
            )
            return

        self.get_logger().info(
            f'Motion result for {self.current_command}: '
            f'success={result.success}, '
            f'code={result.result_code}, '
            f'message={result.message}, '
            f'pos_error={result.final_position_error_m:.3f}, '
            f'heading_error={result.final_heading_error_rad:.3f}'
        )

        if not result.success:
            self._fatal('Stopping command queue because motion failed.')
            return

        self._send_next_command()


def main(args=None):
    rclpy.init(args=args)
    node = None
    exit_code = 0

    try:
        node = MazeSolverNode()

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
