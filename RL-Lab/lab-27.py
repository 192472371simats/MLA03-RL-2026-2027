import heapq
# Road network
graph = {
    "A": [("B", 2), ("C", 4)],
    "B": [("A", 2), ("D", 3)],
    "C": [("A", 4), ("D", 2)],
    "D": [("B", 3), ("C", 2), ("E", 3)],
    "E": []
}
start = "A"
goal = "E"
queue = [(0, start, [start])]
visited = set()
while queue:
    cost, node, path = heapq.heappop(queue)
    if node in visited:
        continue
    visited.add(node)
    if node == goal:
        print("Safe Route:", " -> ".join(path))
        print("Total Travel Cost:", cost)
        break
    for next_node, road_cost in graph[node]:
        if next_node not in visited:
            heapq.heappush(
                queue,
                (cost + road_cost,
                 next_node,
                 path + [next_node])
            )