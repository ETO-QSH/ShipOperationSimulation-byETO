import numpy as np


def compute_curvature_change(path):
    """计算路径的平均曲率变化（使用 NumPy）"""
    path = np.array(path)
    if len(path) < 3:
        return 0

    # 计算相邻点之间的向量
    vectors = np.diff(path, axis=0)

    # 计算向量的方向（角度）
    angles = np.arctan2(vectors[:, 0], vectors[:, 1])

    # 计算相邻方向之间的差异
    angle_diffs = np.abs(np.diff(angles))
    # 处理角度 wrap-around (pi to -pi)
    angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)

    # 计算平均曲率变化
    return np.mean(angle_diffs)


def compute_curvature_radius_three_points(p1, p2, p3):
    """
    根据三个点计算曲率半径
    :param p1: 第一个点坐标 (y, x)
    :param p2: 第二个点坐标 (y, x)
    :param p3: 第三个点坐标 (y, x)
    :return: 曲率半径
    """
    # 将点转换为NumPy数组
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    # 计算向量
    v1 = p1 - p2
    v2 = p3 - p2

    # 计算向量叉积的模长
    cross_product = np.linalg.norm(np.cross(v1, v2))

    # 计算向量的模长
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    # 计算曲率半径
    if cross_product == 0 or v1_norm == 0 or v2_norm == 0:
        return np.inf  # 三点共线，曲率半径为无穷大

    curvature_radius = (v1_norm * v2_norm * np.sqrt(v1_norm**2 + v2_norm**2 - cross_product**2)) / (2 * cross_product)
    return curvature_radius

def evaluate_smoothness_total_reciprocal(path, window_size=5, filter=300):
    """
    评估路径的平滑度，使用每 window_size 个点计算曲率半径的倒数之和
    :param path: 路径坐标列表，每个元素为 (y, x)
    :param window_size: 计算窗口大小，默认为5个点
    :return: 总平滑度（曲率半径倒数之和）
    """
    total_smoothness = 0.0
    n = len(path)

    for i in range(n - window_size + 1):
        # 取当前窗口的点
        window_points = path[i:i+window_size]

        # 提取中间3个点用于计算曲率半径
        p1 = window_points[0]
        p2 = window_points[2]
        p3 = window_points[4]

        # 计算曲率半径
        radius = compute_curvature_radius_three_points(p1, p2, p3)

        # 累加曲率半径的倒数
        if radius != 0 and radius != np.inf and radius < filter:
            # print(radius)
            total_smoothness += 1.0 / radius

    return total_smoothness

def evaluate_smoothness(path, smoothed_path, window_size=5, filter=300):
    """评估原始路径和平滑路径的平滑度"""
    original_smoothness = evaluate_smoothness_total_reciprocal(path, window_size, filter)
    smoothed_smoothness = evaluate_smoothness_total_reciprocal(smoothed_path, window_size, filter)

    evaluation = {
        "original": original_smoothness,
        "smoothed": smoothed_smoothness
    }

    return evaluation
