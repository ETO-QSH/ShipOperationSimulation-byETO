import numpy as np
import matplotlib.pyplot as plt

def load_island_map(filename="res\\island_map.npy"):
    """从文件加载岛屿地图"""
    island_map = np.load(filename)
    print(f"岛屿地图已从 {filename.split('\\')[1]} 加载")
    return island_map

def generate_perlin_noise_2d(shape, scale=100.0, octaves=2, persistence=0.5, lacunarity=2.0, seed=None):
    """生成二维柏林噪声"""
    if seed:
        np.random.seed(seed)
    noise_map = np.zeros(shape)

    def leap(a, b, x):
        return a + x * (b - a)

    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def gradient(h, x, y):
        vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        return vectors[h % 4].dot(np.array([x, y]))

    for i in range(shape[0]):
        for j in range(shape[1]):
            x, y = j / scale, i / scale
            amplitude, frequency, noise_value = 1.0, 1.0, 0.0
            for _ in range(octaves):
                x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
                x1, y1 = x0 + 1, y0 + 1
                sx, sy = fade(x - x0), fade(y - y0)

                np.random.seed(int(x0) + int(y0) * 10000 + seed)
                n00 = gradient(np.random.randint(0, 4), x - x0, y - y0)
                np.random.seed(int(x0) + int(y1) * 10000 + seed)
                n01 = gradient(np.random.randint(0, 4), x - x0, y - y1)
                np.random.seed(int(x1) + int(y0) * 10000 + seed)
                n10 = gradient(np.random.randint(0, 4), x - x1, y - y0)
                np.random.seed(int(x1) + int(y1) * 10000 + seed)
                n11 = gradient(np.random.randint(0, 4), x - x1, y - y1)

                i0, i1 = leap(n00, n10, sx), leap(n01, n11, sx)
                noise_value += leap(i0, i1, sy) * amplitude

                amplitude *= persistence
                frequency *= lacunarity
                x *= frequency
                y *= frequency

            noise_map[i][j] = noise_value

    return noise_map

def filter_islands(island_map, min_size):
    """过滤掉小于最小尺寸的岛屿"""
    height, width = island_map.shape
    visited = np.zeros((height, width), dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i in range(height):
        for j in range(width):
            if island_map[i][j] == 1 and not visited[i][j]:
                stack = [(i, j)]
                visited[i][j] = True
                size, region = 0, []
                while stack:
                    x, y = stack.pop()
                    size += 1
                    region.append((x, y))
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < height and 0 <= ny < width:
                            if island_map[nx][ny] == 1 and not visited[nx][ny]:
                                visited[nx][ny] = True
                                stack.append((nx, ny))
                if size < min_size:
                    for x, y in region:
                        island_map[x][y] = 0

    return island_map

def make_perlin_map(width=1024, height=1024, scale=100.0, octaves=2, persistence=0.5, circularity=2.0, seed=42, island_threshold=0.60, min_island_size=300):
    """
    生成一个随机地图，建议复用 (❀｣╹□╹)｣*･
    :return: island_map

    @ 设置地图尺寸
    :param width: 设置地图的宽度
    :param height: 设置地图的高度

    @ 设置柏林噪声的参数
    :param scale: 控制噪声的密度, 值越小生成的岛屿越密集
    :param octaves: 增加分形层数以生成更复杂的地形, 从而避免生成大片的岛屿
    :param persistence: 控制噪声的平滑度, 值越大岛屿边缘越破碎
    :param circularity: 控制噪声的频率, 岛屿的分布更加分散
    :param seed: 指定随机种子以便复现

    @ 设置陆海比及筛选岛屿
    :param island_threshold: 设置岛屿的阈值，控制岛屿面积占比
    :param min_island_size: 过滤掉小于最小尺寸的岛屿
    """

    # 生成柏林噪声矩阵并归一化
    world = generate_perlin_noise_2d((height, width), scale, octaves, persistence, circularity, seed)
    world = (world - world.min()) / (world.max() - world.min())

    # 设置陆海比例和最小岛屿
    island_map = np.where(world > island_threshold, 1, 0).astype(int)
    island_map = filter_islands(island_map, min_island_size)

    # 统计岛屿面积
    print(f"岛屿面积占比: {np.sum(island_map) / (width * height):.2%}")
    return island_map


if __name__ == "__main__":
    # 生成并保存岛屿地图
    island_map = make_perlin_map()
    np.save(island_map, "res\\island_map.npy")

    # 可视化海图
    plt.imshow(island_map, cmap='terrain')
    plt.title('Generated Island Map', pad=15, fontweight='bold')
    plt.show()
