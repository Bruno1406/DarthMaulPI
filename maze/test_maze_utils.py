import maze_utils

L = [
    [10, 6, 14],
    [8, 0, 4],
    [11, 5, 13]
]

graph = maze_utils.build_graph(L)
print("Graph:")
print(graph)

n = len(L)

print("Cell 1 world position:")
print(maze_utils.cell_to_world(1, n))

print("Cell 5 world position:")
print(maze_utils.cell_to_world(5, n))

print("World position (0.127, 0.127) to cell:")
print(maze_utils.world_to_cell(0.127, 0.127, n))

print("World position (0.381, 0.381) to cell:")
print(maze_utils.world_to_cell(0.381, 0.381, n))