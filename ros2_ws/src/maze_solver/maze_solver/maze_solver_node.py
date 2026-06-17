import rclpy
import math
from rclpy.node import Node
from maze_interface.srv import GetRosMaze
from maze_solver.maze_utils import build_graph,I_to_ij
from rclpy.action import ActionClient
from darth_maul_control_interfaces.action import ExecuteMotionPrimitive

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
        super().__init__('maze_solver_node')
        self.client = self.create_client(GetRosMaze, 'get_ros_maze')  

        self.motion_client = ActionClient(
            self,
            ExecuteMotionPrimitive,
            '/darth_maul_control/execute_motion_primitive'
        )

        self.command_queue = []
        self.current_command = None


        while not self.client.wait_for_service(1.0):
            self.get_logger().info('Waiting for get_ros_maze service...')

        request = GetRosMaze.Request()
        request.maze_nr = 1

        future = self.client.call_async(request)
        future.add_done_callback(self.handle_maze_response)

    def handle_maze_response(self,future):
        response = future.result()
        maze = response.maze

        self.get_logger().info(f'n: {maze.n}')  
        self.get_logger().info(f'm: {maze.m}')
        self.get_logger().info(f'start_idx: {maze.start_idx}')
        self.get_logger().info(f'end_idx: {maze.end_idx}')
        self.get_logger().info(f'start_orientation: {maze.start_orientation}')
        self.get_logger().info(f'l length: {len(maze.l)}')

        L = self.ros_maze_to_matrix(maze)  #见下文
        self.get_logger().info(f'First row of L: {L[0]}')

        graph = build_graph(L)  #maze_util.py

        path = self.astar_path(graph, maze.start_idx, maze.end_idx, maze.n)
        self.get_logger().info(f'Path: {path}')

        orientations = self.path_to_orientations(path, maze.n)
        self.get_logger().info(f'Orientations: {orientations}')

        commands = self.orientations_to_commands(maze.start_orientation, orientations)
        self.get_logger().info(f'Commands: {commands}')

        self.execute_commands(commands)

    def ros_maze_to_matrix(self,maze):
        L=[]
        for i in range(maze.n):
            row=[]
            for j in range(maze.m):
                row.append(maze.l[j*maze.n+i])
            L.append(row)
        return L
    
    def astar_path(self, graph, start_idx, end_idx, n):
        start_node = MazeNode(None, start_idx)
        start_node.g = start_node.h = start_node.f = 0

        end_node = MazeNode(None, end_idx)
        end_node.g = end_node.h = end_node.f = 0

        open_list = []
        closed_list = []

        open_list.append(start_node)

        while len(open_list) > 0:
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

            children = []
            for neighbor_idx in graph.get(current_node.position, []):
                new_node = MazeNode(current_node, neighbor_idx)
                children.append(new_node)

            for child in children:
                if child in closed_list:
                    continue

                child.g = current_node.g + 1
                child.h = self.manhattan_distance(child.position, end_node.position, n)
                child.f = child.g + child.h

                is_in_open = False
                for open_node in open_list:
                    if child == open_node:
                        is_in_open = True
                        
                        if child.g < open_node.g:
                            open_list.remove(open_node)
                            open_list.append(child)
                        break      

                if not is_in_open:
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
    
    def orientations_to_commands(self, start_orientation, orientations):
        commands = []
        current_orientation = start_orientation

        for target_orientation in orientations:
            turn = self.turn_between_orientations(current_orientation,target_orientation)

            if turn != 0:
                commands.append(('ratation',turn))

            commands.append(('drive_forward',0.254))
            current_orientation = target_orientation
        
        return commands
    
    def turn_between_orientations(self, current, target):
        angles = {
            1: 0.0,
            4: math.pi/2.0,
            2: math.pi,
            8: -math.pi/2.0
        }

        diff = angles[target]-angles[current]
        while diff > math.pi:
            diff = diff-2.0*math.pi
        while diff < -math.pi:
            diff = diff+2.0*math.pi
        
        return diff
    
    def execute_commands(self, commands):
        self.command_queue = list(commands)

        if not self.command_queue:
            self.get_logger().info('No motion commands to execute.')
            return

        self.get_logger().info(f'Executing {len(self.command_queue)} motion commands.')
        self.send_next_command()

    def send_next_command(self):
        if not self.command_queue:
            self.get_logger().info('All motion commands finished.')
            return
        
        command, value = self.command_queue.pop(0)
        self.current_command = (command, value)

        if not self.motion_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('Motion action server is not available.')
            return

        goal = ExecuteMotionPrimitive.Goal()

        if command == 'drive_forward':
            goal.primitive_type = ExecuteMotionPrimitive.Goal.DRIVE_FORWARD
            goal.value = float(value)
        elif command == 'rotation':
            goal.primitive_type = ExecuteMotionPrimitive.Goal.ROTATE_RELATIVE
            goal.value = float(value)
        else:
            self.get_logger().error(f'Unknown motion command: {command}')
            return

        goal.position_tolerance_m = 0.0
        goal.heading_tolerance_rad = 0.0
        goal.max_linear_x_mps = 0.0
        goal.max_linear_y_mps = 0.0
        goal.max_angular_z_radps = 0.0

        self.get_logger().info(f'Sending motion command: {command}, {value}')
        future = self.motion_client.send_goal_async(goal)
        future.add_done_callback(self.handle_motion_goal_response)

    def handle_motion_goal_response(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error(f'Motion goal rejected: {self.current_command}')
            return

        self.get_logger().info(f'Motion goal accepted: {self.current_command}')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.handle_motion_result)

    def handle_motion_result(self, future):
        result = future.result().result

        self.get_logger().info(
            f'Motion result for {self.current_command}: '
            f'success={result.success}, '
            f'code={result.result_code}, '
            f'message={result.message}'
        )

        if not result.success:
            self.get_logger().error('Stopping command queue because motion failed.')
            return

        self.send_next_command()


    
def main(args=None):
    rclpy.init(args=args)
    node = MazeSolverNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
