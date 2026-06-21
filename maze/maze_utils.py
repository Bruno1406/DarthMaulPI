L = [
    [10, 6, 14],
    [8, 0, 4],
    [11, 5, 13]
]
def ij_to_I(i, j, n):
    """
    Convert matrix coordinates (i, j) to the maze segment index I.

    Important:
    - In this Python code, i and j are 0-based indices.
    - In the project documentation, I starts at 1.
    - The maze is numbered column by column.

    Formula from documentation:
        I = (j - 1) * n + i

    Python 0-based version:
        I = j * n + i + 1
    """
    return j * n + i + 1


def I_to_ij(I, n):
    """
    Convert a maze segment index I back to matrix coordinates (i, j).

    Important:
    - I is 1-based, as used in the task description.
    - The returned i and j are 0-based Python list indices.
    """
    i = (I - 1) % n
    j = (I - 1) // n
    return i, j


def get_neighbors(i, j, L):
    """
    Return all reachable neighboring cells of cell (i, j).

    Parameters:
        i, j:
            0-based matrix coordinates.
            Example: L[i][j]

        L:
            Maze matrix. Each cell contains a 4-bit wall encoding.

    Wall encoding:
        1 = wall in +x direction, i.e. down
        2 = wall in -x direction, i.e. up
        4 = wall in +y direction, i.e. right
        8 = wall in -y direction, i.e. left
        15 = cell is not part of the maze

    Returns:
        A list of neighboring cells as 0-based coordinate tuples:
            [(i1, j1), (i2, j2), ...]
    """
    cell = L[i][j]
    neighbors = []

# If there is no wall in +x direction, we can move down.
    if not (cell & 1):
        neighbors.append((i + 1, j))

    # If there is no wall in -x direction, we can move up.
    if not (cell & 2):
        neighbors.append((i - 1, j))

    # If there is no wall in +y direction, we can move right.
    if not (cell & 4):
        neighbors.append((i, j + 1))

    # If there is no wall in -y direction, we can move left.
    if not (cell & 8):
        neighbors.append((i, j - 1))

    return neighbors


def build_graph(L):
    """
    Convert the maze matrix L into a graph.

    The graph uses the segment index I as node ID.
    This is useful because start_idx and end_idx from RosMaze.msg
    are also given as segment indices.

    Example output:
        {
            1: [2, 4],
            2: [1, 3],
            4: [1, 5],
        }

    Meaning:
        Cell 1 can reach cells 2 and 4.
        Cell 2 can reach cells 1 and 3.
        Cell 4 can reach cells 1 and 5.
    """
    graph = {}

    n = len(L)
    m = len(L[0])

    for i in range(n):
        for j in range(m):
            # Skip cells that are not part of the maze.
            if L[i][j] == 15:
                continue

            # Convert current cell coordinates to segment index I.
            current = ij_to_I(i, j, n)

            # Get reachable neighboring cells as matrix coordinates.
            neighbors = get_neighbors(i, j, L)

            # Convert neighbor coordinates to segment indices as well.
            neighbor_ids = []
            
            for ni, nj in neighbors:
                neighbor_id = ij_to_I(ni, nj, n)
                neighbor_ids.append(neighbor_id)

            graph[current] = neighbor_ids

    return graph


#test
'''if __name__ == "__main__":
    L = [
        [10, 6, 14],
        [8, 0, 4],
        [11, 5, 13]
    ]

    graph = build_graph(L)
    print(graph)'''

def cell_to_world(I,n):
    i,j = I_to_ij(I,n)
    x = (i+0.5)*0.254
    y = (j+0.5)*0.254
    return x,y

def world_to_cell(x,y,n):
    i = int(x//0.254)
    j = int(y//0.254)
    I = ij_to_I(i,j,n)
    return I
