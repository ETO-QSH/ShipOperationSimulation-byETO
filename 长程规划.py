from 平滑操作 import smooth_path
from heapq import heappush, heappop
from 地图生成 import np, plt, load_island_map

def dilate_obstacles(grid, dilation_radius=1):
    """障碍物（陆地）膨胀处理"""
    dilated = grid.copy()
    h, w = grid.shape
    for y in range(h):
        for x in range(w):
            if grid[y, x] == 1:  # 陆地作为障碍物
                for dy in range(-dilation_radius, dilation_radius + 1):
                    for dx in range(-dilation_radius, dilation_radius + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            dilated[ny, nx] = 1  # 将周围标记为障碍
    return dilated

def a_star_marine(grid, start, end, safe_margin=16):
    """
    带宽度要求的A*算法实现
    :param grid: 原始地图 (1: 陆地, 0: 海洋)
    :param start: 起点 (y, x)
    :param end: 终点 (y, x)
    :param safe_margin: 要求离陆地的最小安全距离（单位：格）
    :return: 路径坐标列表
    """
    # 根据安全距离计算膨胀半径
    dilated_radius = safe_margin
    safe_grid = dilate_obstacles(grid, dilated_radius)

    # 检查起终点是否在安全区域
    if safe_grid[start] == 1 or safe_grid[end] == 1:
        raise ValueError("起点或终点位于危险区域！")

    # A*算法实现（16方向移动）
    h, w = grid.shape
    open_heap, came_from, cost = [], {}, {start: 0}
    heappush(open_heap, (0, start))

    # 16个移动方向及其代价
    sq2, sq5 = np.sqrt(2), np.sqrt(5)
    directions = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, sq2), (-1, 1, sq2), (1, -1, sq2), (1, 1, sq2),
        (-2, -1, sq5), (-2, 1, sq5), (2, -1, sq5), (2, 1, sq5),
        (-1, -2, sq5), (-1, 2, sq5), (1, -2, sq5), (1, 2, sq5)
    ]

    # 启发式函数（欧几里得距离）
    def heuristic(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    while open_heap:
        current = heappop(open_heap)[1]
        if current == end:
            break

        for dy, dx, move_cost in directions:
            ny, nx = current[0] + dy, current[1] + dx
            # 检查是否在安全航道上
            if 0 <= ny < h and 0 <= nx < w and safe_grid[ny, nx] == 0:
                new_cost = cost[current] + move_cost
                if (ny, nx) not in cost or new_cost < cost[(ny, nx)]:
                    cost[(ny, nx)] = new_cost
                    priority = new_cost + heuristic(end, (ny, nx))
                    heappush(open_heap, (priority, (ny, nx)))
                    came_from[(ny, nx)] = current

    # 路径回溯
    path, current = [], end
    while current != start:
        path.append(current)
        current = came_from.get(current, None)
        if current is None:
            return []
    path.append(start)
    return path[::-1]

def find_base_path(island_map, start, end, safe_margin=16):
    ly, lx = len(island_map), len(island_map[0])
    p5 = [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]
    path = a_star_marine(island_map, start, end, safe_margin=safe_margin)  # 查找安全路径
    if path:
        smoothed_path = smooth_path(path, base_epsilon=2.5, base_alpha=0.25)
        for point in smoothed_path:
            # 将浮点坐标四舍五入并转为整数
            y, x = int(point[0]+0.5), int(point[1]+0.5)
            for dy, dx in p5:  # 标记路径点及其周围区域
                ny = np.clip(y + dy, 0, ly - 1)
                nx = np.clip(x + dx, 0, lx - 1)
                island_map[ny, nx] = 2  # 使用2表示路径
        return island_map
    else:
        return None


if __name__ == "__main__":
    island_map = load_island_map()
    start, end = (0, 1023), (1023, 0)
    island_map = find_base_path(island_map, start, end, 16)
    if type(island_map) == np.ndarray:
        np.save(island_map, "res\\island_map_A_Star")
        plt.imshow(island_map, cmap='terrain')
        plt.scatter(start[1], start[0], c='red', s=200, marker='*', label='Start')
        plt.scatter(end[1], end[0], c='pink', s=200, marker='*', label='End')
        plt.title('A Star Path with Safety Margin and Smoothness', pad=15, fontweight='bold')
        plt.show()
    else:
        print("未找到安全路径！")
