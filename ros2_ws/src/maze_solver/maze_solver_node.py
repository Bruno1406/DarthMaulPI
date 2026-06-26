import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

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
    max_cells_per_drive: int,
) -> List[Tuple[str, float]]:
    max_cells_per_drive = max(1, int(max_cells_per_drive))
    commands: List[Tuple[str, float]] = []

    if not orientations:
        return commands

    current_orientation = int(start_orientation)
    i = 0

    while i < len(orientations):
        run_orientation = int(orientations[i])
        run_length = 1

        while (
            i + run_length < len(orientations)
            and int(orientations[i + run_length]) == run_orientation
        ):
            run_length += 1

        turn = turn_between_orientations(current_orientation, run_orientation)
        if abs(turn) > 1.0e-6:
            commands.append(('rotate', turn))

        remaining_cells = run_length
        while remaining_cells > 0:
            cells_this_command = min(remaining_cells, max_cells_per_drive)
            distance_m = float(cells_this_command) * float(cell_length_m)
            commands.append(('drive_forward', distance_m))
            remaining_cells -= cells_this_command

        current_orientation = run_orientation
        i += run_length

    return commands


class MazeSolverNode(Node):
    def __init__(self):
        super().__init__('maze_solver_node')

        self.declare_parameter('maze_nr', 1)
        self.declare_parameter('cell_length_m', 0.254)
        self.declare_parameter('max_cells_per_drive', 2)
        self.declare_parameter('execute_motions', True)
        self.declare_parameter('motion_server_timeout_s', 5.0)

        self.maze_nr = int(self.get_parameter('maze_nr').value)
        self.cell_length_m = float(self.get_parameter('cell_length_m').value)
        self.max_cells_per_drive = max(
            1,
            int(self.get_parameter('max_cells_per_drive').value),
        )
        self.execute_motions = bool(self.get_parameter('execute_motions').value)
        self.motion_server_timeout_s = float(
            self.get_parameter('motion_server_timeout_s').value
        )

        self.get_logger().info(
            f'maze_nr={self.maze_nr}, '
            f'cell_length_m={self.cell_length_m:.3f}, '
            f'max_cells_per_drive={self.max_cells_per_drive}, '
            f'execute_motions={self.execute_motions}'
        )

        self.maze_client = self.create_client(GetRosMaze, 'get_ros_maze')
        self.motion_client = ActionClient(
            self,
            ExecuteMotionPrimitive,
            '/darth_maul_control/execute_motion_primitive',
        )

        self.command_queue: List[Tuple[str, float]] = []
        self.current_command: Optional[Tuple[str, float]] = None

        self._request_maze()

    def _request_maze(self) -> None:
        while not self.maze_client.wait_for_service(1.0):
            self.get_logger().info('Waiting for get_ros_maze service...')

        request = GetRosMaze.Request()
        request.maze_nr = self.maze_nr

        future = self.maze_client.call_async(request)
        future.add_done_callback(self._handle_maze_response)

    def _handle_maze_response(self, future) -> None:
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(f'get_ros_maze failed: {exc}')
            return

        maze = response.maze

        self.get_logger().info(f'n: {maze.n}')
        self.get_logger().info(f'm: {maze.m}')
        self.get_logger().info(f'start_idx: {maze.start_idx}')
        self.get_logger().info(f'end_idx: {maze.end_idx}')
        self.get_logger().info(f'start_orientation: {maze.start_orientation}')
        self.get_logger().info(f'l length: {len(maze.l)}')

        if maze.n == 0 or maze.m == 0 or not maze.l:
            self.get_logger().error('Received empty maze.')
            return

        grid = self._ros_maze_to_matrix(maze)
        graph = self._build_graph(grid)

        path = self._astar_path(graph, int(maze.start_idx), int(maze.end_idx), int(maze.n))
        self.get_logger().info(f'Path: {path}')

        if not path:
            self.get_logger().error('No path found.')
            return

        orientations = self._path_to_orientations(path, int(maze.n))
        self.get_logger().info(f'Orientations: {orientations}')

        commands = self._orientations_to_commands(int(maze.start_orientation), orientations)
        self.get_logger().info(f'Commands: {commands}')
        self.get_logger().info(f'Path cells: {len(path)}')
        self.get_logger().info(f'Path edges: {len(orientations)}')
        self.get_logger().info(f'Motion commands after compression: {len(commands)}')
        for idx, command in enumerate(commands, start=1):
            self.get_logger().info(f'Command {idx}/{len(commands)}: {command}')

        if not self.execute_motions:
            self.get_logger().warn(
                'execute_motions is false; command list generated but not executed.'
            )
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

    def _orientations_to_commands(self, start_orientation, orientations):
        return build_motion_commands(
            int(start_orientation),
            list(orientations),
            self.cell_length_m,
            self.max_cells_per_drive,
        )

    def _turn_between_orientations(self, current, target):
        return turn_between_orientations(int(current), int(target))

    def _execute_commands(self, commands):
        self.command_queue = list(commands)

        if not self.command_queue:
            self.get_logger().info('No motion commands to execute.')
            return

        self.get_logger().info(f'Executing {len(self.command_queue)} motion commands.')

        if not self.motion_client.wait_for_server(timeout_sec=self.motion_server_timeout_s):
            self.get_logger().error('Motion action server is not available.')
            return

        self._send_next_command()

    def _send_next_command(self):
        if not self.command_queue:
            self.get_logger().info('All motion commands finished.')
            return

        command, value = self.command_queue.pop(0)
        self.current_command = (command, value)

        goal = ExecuteMotionPrimitive.Goal()

        if command == 'drive_forward':
            goal.primitive_type = ExecuteMotionPrimitive.Goal.DRIVE_FORWARD
            goal.value = float(value)
            goal.collision_check_enabled = True
        elif command == 'rotate':
            goal.primitive_type = ExecuteMotionPrimitive.Goal.ROTATE_RELATIVE
            goal.value = float(value)
            goal.collision_check_enabled = False
        else:
            self.get_logger().error(f'Unknown motion command: {command}')
            return

        goal.position_tolerance_m = 0.0
        goal.heading_tolerance_rad = 0.0
        goal.max_linear_x_mps = 0.0
        goal.max_linear_y_mps = 0.0
        goal.max_angular_z_radps = 0.0
        goal.timeout_s = 0.0

        self.get_logger().info(f'Sending motion command: {command}, {value}')
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
            self.get_logger().error(
                f'Motion goal request failed for {self.current_command}: {exc}'
            )
            return

        if goal_handle is None:
            self.get_logger().error(
                f'Motion goal request returned no goal handle for {self.current_command}'
            )
            return

        if not goal_handle.accepted:
            self.get_logger().error(f'Motion goal rejected: {self.current_command}')
            return

        self.get_logger().info(f'Motion goal accepted: {self.current_command}')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._handle_motion_result)

    def _handle_motion_result(self, future):
        try:
            wrapped_result = future.result()
            result = wrapped_result.result
        except Exception as exc:
            self.get_logger().error(
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
            self.get_logger().error('Stopping command queue because motion failed.')
            return

        self._send_next_command()


def main(args=None):
    rclpy.init(args=args)
    node = MazeSolverNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
