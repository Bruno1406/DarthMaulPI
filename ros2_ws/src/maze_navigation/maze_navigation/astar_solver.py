import rclpy
from rclpy.node import Node as RosNode
from maze_interface.srv import GetRosMaze

# --- NODE CLASS FOR A* (A-STAR) ALGORITHM ---
class MazeNode:
    def __init__(self, parent=None, position=None):
        self.parent = parent      # Where did we come from? (To retrace the path back)
        self.position = position  # Coordinate in (row, column) format
        
        self.g = 0  # Real step cost from the start node to this node
        self.h = 0  # Heuristic cost (Manhattan Distance) from this node to the target
        self.f = 0  # Total cost (f = g + h)

    def __eq__(self, other):
        # Treat nodes with the same coordinates as the same node
        return self.position == other.position

# --- MANHATTAN DISTANCE FUNCTION ---
def heuristic(current_pos, target_pos):
    # Absolute sum of X and Y differences since diagonal movement is not allowed
    return abs(current_pos[0] - target_pos[0]) + abs(current_pos[1] - target_pos[1])

# --- MAIN ROS 2 SOLVER NODE ---
class MazeSolverNode(RosNode):
    def __init__(self):
        super().__init__('maze_solver_node')
        self.client = self.create_client(GetRosMaze, 'get_maze') 
        
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Searching for the Maze server...')
        
        self.send_request()

    def send_request(self):
        request = GetRosMaze.Request()
        request.maze_nr = 1  # We are requesting maze number 1
        self.future = self.client.call_async(request)
        self.future.add_done_callback(self.maze_response_callback)

    def maze_response_callback(self, future):
        try:
            response = future.result()
            maze = response.maze
            self.get_logger().info('--- MAZE DATA SUCCESSFULLY RECEIVED ---')
            
            # Dimensions of the maze
            cols = maze.m
            
            # Helper function to convert 1D Index to 2D (Row, Column) Matrix Coordinate
            def idx_to_coord(idx):
                idx_0 = idx - 1  # Shifting index to 0-based if they start from 1
                r = idx_0 // cols
                c = idx_0 % cols
                return (r, c)
                
            start_pos = idx_to_coord(maze.start_idx)
            end_pos = idx_to_coord(maze.end_idx)
            
            self.get_logger().info(f'Start (Row, Column): {start_pos}')
            self.get_logger().info(f'Target (Row, Column): {end_pos}')
            
            # --- A* (A-STAR) ALGORITHM STARTS HERE ---
            
            # 1. Initialize Open and Closed lists
            open_list = []
            closed_list = []

            # 2. Create Start and End nodes
            start_node = MazeNode(None, start_pos)
            start_node.g = start_node.h = start_node.f = 0
            
            end_node = MazeNode(None, end_pos)
            end_node.g = end_node.h = end_node.f = 0

            # Add the start node to the open list
            open_list.append(start_node)

            # Helper function: Convert 2D coordinate to 1D index to read wall data
            def coord_to_idx(r, c):
                return r * cols + c

            # Movement array: (Row change, Col change, Wall bit to check)
            # Assuming standard 4-bit encoding: Top=1, Right=2, Bottom=4, Left=8
            directions = [
                (-1, 0, 1),  # Move UP (Check TOP wall of current cell)
                (0, 1, 2),   # Move RIGHT (Check RIGHT wall of current cell)
                (1, 0, 4),   # Move DOWN (Check BOTTOM wall of current cell)
                (0, -1, 8)   # Move LEFT (Check LEFT wall of current cell)
            ]

            self.get_logger().info('Calculating the shortest path...')

            # 3. Main Loop
            while len(open_list) > 0:
                # Get the current node (the one with the lowest f cost)
                current_node = open_list[0]
                current_index = 0
                for index, item in enumerate(open_list):
                    if item.f < current_node.f:
                        current_node = item
                        current_index = index

                # Pop current node from open list, add to closed list
                open_list.pop(current_index)
                closed_list.append(current_node)

                # 4. Check if we reached the Target
                if current_node == end_node:
                    path = []
                    current = current_node
                    while current is not None:
                        path.append(current.position)
                        current = current.parent
                    
                    path = path[::-1]  # Reverse the path to get Start -> End
                    self.get_logger().info('--- PATH SUCCESSFULLY FOUND! ---')
                    self.get_logger().info(f'Total Steps: {len(path) - 1}')
                    self.get_logger().info(f'Path Coordinates: {path}')
                    return  # Algorithm finishes here

                # 5. Generate Children (Adjacent cells)
                children = []
                current_r, current_c = current_node.position
                
                # Get the wall data for the current cell
                current_1d_idx = coord_to_idx(current_r, current_c)
                current_walls = maze.walls[current_1d_idx]

                for row_offset, col_offset, wall_bit in directions:
                    node_position = (current_r + row_offset, current_c + col_offset)

                    # 5.a Make sure the new position is within maze boundaries
                    if (node_position[0] > (maze.n - 1) or node_position[0] < 0 or 
                        node_position[1] > (maze.m - 1) or node_position[1] < 0):
                        continue

                    # 5.b Wall Check (Bitwise AND)
                    # If the result is NOT 0, it means there is a wall blocking the way!
                    if (current_walls & wall_bit) != 0:
                        continue

                    # Valid move! Create new node and add to children
                    new_node = MazeNode(current_node, node_position)
                    children.append(new_node)

                # 6. Loop through children and evaluate costs
                for child in children:
                    # If child is already in the closed list, skip it
                    if child in closed_list:
                        continue

                    # Calculate g, h, and f values
                    child.g = current_node.g + 1
                    child.h = heuristic(child.position, end_node.position)
                    child.f = child.g + child.h

                    # If child is already in the open list with a lower cost, skip it
                    is_in_open = False
                    for open_node in open_list:
                        if child == open_node and child.g >= open_node.g:
                            is_in_open = True
                            break
                    
                    if not is_in_open:
                        open_list.append(child)
            
            # If the loop finishes and we haven't returned, the target is unreachable
            self.get_logger().warn('No valid path could be found!')
            
        except Exception as e:
            self.get_logger().error(f'An error occurred: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = MazeSolverNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()