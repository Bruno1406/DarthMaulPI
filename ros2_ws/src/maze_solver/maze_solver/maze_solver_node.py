import math

import rclpy
from darth_maul_control_interfaces.action import ExecuteMotionPrimitive
from maze_interface.srv import GetRosMaze
from maze_solver.maze_utils import I_to_ij, build_graph
from rclpy.action import ActionClient
from rclpy.node import Node


class MazeNode:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


class MazeSolverNode(Node):
    def __init__(self):
        super().__init__("maze_solver_node")

        self.declare_parameter("maze_nr", 1)
        self.declare_parameter("cell_length_m", 0.254)
        self.declare_parameter("position_tolerance_m", 0.025)
        self.declare_parameter("heading_tolerance_rad", 0.05)
        self.declare_parameter("max_linear_x_mps", 0.08)
        self.declare_parameter("max_angular_z_radps", 0.35)
        self.declare_parameter("collision_check_enabled", True)

        self.maze_nr = int(self.get_parameter("maze_nr").value)
        self.cell_length_m = float(self.get_parameter("cell_length_m").value)
        self.position_tolerance_m = float(self.get_parameter("position_tolerance_m").value)
        self.heading_tolerance_rad = float(self.get_parameter("heading_tolerance_rad").value)
        self.max_linear_x_mps = float(self.get_parameter("max_linear_x_mps").value)
        self.max_angular_z_radps = float(self.get_parameter("max_angular_z_radps").value)
        self.collision_check_enabled = bool(self.get_parameter("collision_check_enabled").value)

        self.maze_client = self.create_client(GetRosMaze, "get_ros_maze")
        self.motion_client = ActionClient(
            self,
            ExecuteMotionPrimitive,
            "/darth_maul_control/execute_motion_primitive",
        )

        self.command_queue = []
        self.current_command = None

        self._request_maze()

    def _request_maze(self):
        while not self.maze_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for get_ros_maze service...")

        request = GetRosMaze.Request()
        request.maze_nr = self.maze_nr

        future = self.maze_client.call_async(request)
        future.add_done_callback(self.handle_maze_response)

    def handle_maze_response(self, future):
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(f"get_ros_maze failed: {exc}")
            return

        maze = response.maze

        self.get_logger().info(f"n: {maze.n}")
        self.get_logger().info(f"m: {maze.m}")
        self.get_logger().info(f"start_idx: {maze.start_idx}")
        self.get_logger().info(f"end_idx: {maze.end_idx}")
        self.get_logger().info(f"start_orientation: {maze.start_orientation}")
        self.get_logger().info(f"l length: {len(maze.l)}")

        if maze.n == 0 or maze.m == 0 or not maze.l:
            self.get_logger().error("Received empty maze. Aborting.")
            return

        L = self.ros_maze_to_matrix(maze)
        graph = build_graph(L)

        path = self.astar_path(graph, maze.start_idx, maze.end_idx, maze.n)
        if not path:
            self.get_logger().error("No path found. Aborting.")
            return

        self.get_logger().info(f"Path: {path}")

        orientations = self.path_to_orientations(path, maze.n)
        self.get_logger().info(f"Orientations: {orientations}")

        commands = self.orientations_to_commands(maze.start_orientation, orientations)
        self.get_logger().info(f"Commands: {commands}")

        self.execute_commands(commands)

    def ros_maze_to_matrix(self, maze):
        L = []
        for i in range(maze.n):
            row = []
            for j in range(maze.m):
                row.append(maze.l[j * maze.n + i])
            L.append(row)
        return L

    def astar_path(self, graph, start_idx, end_idx, n):
        start_node = MazeNode(None, start_idx)
        end_node = MazeNode(None, end_idx)

        open_list = [start_node]
        closed_list = []

        while open_list:
            current_node = open_list[0]
            current_index = 0

            for index, item in enumerate(open_list):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index

            open_list.pop(current_index)
            closed_list.append(current_node)

            if current_node == end_node:
                path = []
                current = current_node

                while current is not None:
                    path.append(current.position)
                    current = current.parent

                return path[::-1]

            for neighbor_idx in graph.get(current_node.position, []):
                child = MazeNode(current_node, neighbor_idx)

                if child in closed_list:
                    continue

                child.g = current_node.g + 1
                child.h = self.manhattan_distance(child.position, end_node.position, n)
                child.f = child.g + child.h

                replaced = False
                for open_node in open_list:
                    if child == open_node:
                        replaced = True
                        if child.g < open_node.g:
                            open_list.remove(open_node)
                            open_list.append(child)
                        break

                if not replaced:
                    open_list.append(child)

        return []

    def manhattan_distance(self, current_idx, target_idx, n):
        current_i, current_j = I_to_ij(current_idx, n)
        target_i, target_j = I_to_ij(target_idx, n)
        return abs(current_i - target_i) + abs(current_j - target_j)

    def path_to_orientations(self, path, n):
        orientations = []

        for current, nxt in zip(path, path[1:]):
            diff = nxt - current

            if diff == 1:
                orientations.append(1)
            elif diff == -1:
                orientations.append(2)
            elif diff == n:
                orientations.append(4)
            elif diff == -n:
                orientations.append(8)
            else:
                raise ValueError(f"{current} and {nxt} are not neighboring cells")

        return orientations

    def orientations_to_commands(self, start_orientation, orientations):
        commands = []
        current_orientation = start_orientation

        for target_orientation in orientations:
            turn = self.turn_between_orientations(current_orientation, target_orientation)

            if abs(turn) > 1.0e-4:
                commands.append(("rotation", turn))

            commands.append(("drive_forward", self.cell_length_m))
            current_orientation = target_orientation

        return commands

    def turn_between_orientations(self, current, target):
        angles = {
            1: 0.0,
            4: math.pi / 2.0,
            2: math.pi,
            8: -math.pi / 2.0,
        }

        diff = angles[target] - angles[current]

        while diff > math.pi:
            diff -= 2.0 * math.pi

        while diff < -math.pi:
            diff += 2.0 * math.pi

        return diff

    def execute_commands(self, commands):
        self.command_queue = list(commands)

        if not self.command_queue:
            self.get_logger().info("No motion commands to execute.")
            return

        self.get_logger().info(f"Executing {len(self.command_queue)} motion commands.")
        self.send_next_command()

    def send_next_command(self):
        if not self.command_queue:
            self.get_logger().info("All motion commands finished.")
            return

        command, value = self.command_queue.pop(0)
        self.current_command = (command, value)

        if not self.motion_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Motion action server is not available.")
            return

        goal = ExecuteMotionPrimitive.Goal()

        if command == "drive_forward":
            goal.primitive_type = ExecuteMotionPrimitive.Goal.DRIVE_FORWARD
            goal.value = float(value)
            goal.max_linear_x_mps = self.max_linear_x_mps
            goal.max_angular_z_radps = 0.0
        elif command == "rotation":
            goal.primitive_type = ExecuteMotionPrimitive.Goal.ROTATE_RELATIVE
            goal.value = float(value)
            goal.max_linear_x_mps = 0.0
            goal.max_angular_z_radps = self.max_angular_z_radps
        else:
            self.get_logger().error(f"Unknown motion command: {command}")
            return

        goal.position_tolerance_m = self.position_tolerance_m
        goal.heading_tolerance_rad = self.heading_tolerance_rad
        goal.max_linear_y_mps = 0.0
        goal.timeout_s = 0.0
        goal.collision_check_enabled = self.collision_check_enabled

        self.get_logger().info(f"Sending motion command: {command}, {value}")
        future = self.motion_client.send_goal_async(goal)
        future.add_done_callback(self.handle_motion_goal_response)

    def handle_motion_goal_response(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"Motion goal send failed: {exc}")
            return

        if not goal_handle.accepted:
            self.get_logger().error(f"Motion goal rejected: {self.current_command}")
            return

        self.get_logger().info(f"Motion goal accepted: {self.current_command}")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.handle_motion_result)

    def handle_motion_result(self, future):
        try:
            result = future.result().result
        except Exception as exc:
            self.get_logger().error(f"Motion result failed: {exc}")
            return

        self.get_logger().info(
            f"Motion result for {self.current_command}: "
            f"success={result.success}, "
            f"code={result.result_code}, "
            f"message={result.message}"
        )

        if not result.success:
            self.get_logger().error("Stopping command queue because motion failed.")
            return

        self.send_next_command()


def main(args=None):
    rclpy.init(args=args)
    node = MazeSolverNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
