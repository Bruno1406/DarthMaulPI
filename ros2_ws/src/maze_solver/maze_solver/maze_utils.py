CELL_LENGTH_M = 0.254


def ij_to_I(i, j, n):
    return j * n + i + 1


def I_to_ij(I, n):
    i = (I - 1) % n
    j = (I - 1) // n
    return i, j


def _inside(i, j, L):
    return 0 <= i < len(L) and 0 <= j < len(L[0])


def get_neighbors(i, j, L):
    cell = L[i][j]
    if cell == 15:
        return []

    candidates = []

    if not (cell & 1):
        candidates.append((i + 1, j))

    if not (cell & 2):
        candidates.append((i - 1, j))

    if not (cell & 4):
        candidates.append((i, j + 1))

    if not (cell & 8):
        candidates.append((i, j - 1))

    neighbors = []
    for ni, nj in candidates:
        if not _inside(ni, nj, L):
            continue
        if L[ni][nj] == 15:
            continue
        neighbors.append((ni, nj))

    return neighbors


def build_graph(L):
    graph = {}

    n = len(L)
    if n == 0:
        return graph

    m = len(L[0])
    if m == 0:
        return graph

    for i in range(n):
        for j in range(m):
            if L[i][j] == 15:
                continue

            current = ij_to_I(i, j, n)
            neighbor_ids = []

            for ni, nj in get_neighbors(i, j, L):
                neighbor_ids.append(ij_to_I(ni, nj, n))

            graph[current] = neighbor_ids

    return graph


def cell_to_world(I, n, cell_length_m=CELL_LENGTH_M):
    i, j = I_to_ij(I, n)
    x = (i + 0.5) * cell_length_m
    y = (j + 0.5) * cell_length_m
    return x, y


def world_to_cell(x, y, n, cell_length_m=CELL_LENGTH_M):
    if x < 0.0 or y < 0.0:
        raise ValueError("world coordinates must be non-negative")

    i = int(x // cell_length_m)
    j = int(y // cell_length_m)
    return ij_to_I(i, j, n)
