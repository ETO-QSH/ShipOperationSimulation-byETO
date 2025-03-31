import math
from 地图生成 import np
from scipy.integrate import quad

def douglas_pucker(points, epsilon):
    """道格拉斯-普克降采样算法"""
    if len(points) <= 2:
        return points.copy()

    # 寻找最大距离点
    max_d, index = 0, 0
    line_start, line_end = points[0], points[-1]

    for i in range(1, len(points) - 1):
        d = perpendicular_distance(points[i], line_start, line_end)
        if d > max_d:
            max_d, index = d, i
    if max_d > epsilon:
        left = douglas_pucker(points[:index+1], epsilon)
        right = douglas_pucker(points[index:], epsilon)
        return left[:-1] + right
    else:
        return [line_start, line_end]

def perpendicular_distance(point, line_start, line_end):
    """计算点到直线的垂直距离"""
    (x, y), (x1, y1), (x2, y2) = point, line_start, line_end

    if (x1 == x2) and (y1 == y2):
        return math.hypot(x - x1, y - y1)

    a, b, c = y2 - y1, x1 - x2, x2 * y1 - x1 * y2
    return abs(a * x + b * y + c) / math.sqrt(a ** 2 + b ** 2)

def calculate_circle_center(A, B, C):
    """计算三点共圆的圆心"""
    (ax, ay), (bx, by), (cx, cy) = A, B, C

    # 解线性方程组
    a1, b1 = bx - ax, by - ay
    c1 = (bx ** 2 + by ** 2 - ax ** 2 - ay ** 2) / 2
    a2, b2 = cx - bx, cy - by
    c2 = (cx ** 2 + cy ** 2 - bx ** 2 - by ** 2) / 2

    det = a1 * b2 - a2 * b1
    if det == 0:
        return None  # 三点共线
    return (b2 * c1 - b1 * c2) / det, (a1 * c2 - a2 * c1) / det

def compute_tangent_directions(key_points):
    """计算每个关键点的切线方向"""
    directions, n = [], len(key_points)

    for i in range(n):
        if i == 0 or i == n - 1:
            # 起点和终点的处理
            if i == 0:
                dx = key_points[i + 1][0] - key_points[i][0]
                dy = key_points[i + 1][1] - key_points[i][1]
            else:
                dx = key_points[i][0] - key_points[i - 1][0]
                dy = key_points[i][1] - key_points[i - 1][1]
        else:
            # 中间点计算切线方向
            prev_p = key_points[i - 1]
            current_p = key_points[i]
            next_p = key_points[i + 1]

            # 计算圆心
            center = calculate_circle_center(prev_p, current_p, next_p)
            if center is None:
                dx, dy = next_p[0] - prev_p[0], next_p[1] - prev_p[1]  # 三点共线，取前后方向
            else:
                vec_x, vec_y = current_p[0] - center[0], current_p[1] - center[1]  # 计算半径向量

                # 计算转弯方向
                v1 = (current_p[0] - prev_p[0], current_p[1] - prev_p[1])
                v2 = (next_p[0] - current_p[0], next_p[1] - current_p[1])
                cross = v1[0] * v2[1] - v1[1] * v2[0]  # 叉积z分量
                dx_base, dy_base = -vec_y, vec_x  # 确定切线方向

                # 根据转弯方向调整切线方向
                if cross > 0:  # 左转
                    dx, dy = dx_base, dy_base
                else:  # 右转
                    dx, dy = -dx_base, -dy_base

                # 保持方向与路径前进一致
                forward_vec = (next_p[0] - prev_p[0], next_p[1] - prev_p[1])
                dot_product = dx * forward_vec[0] + dy * forward_vec[1]
                if dot_product < 0:
                    dx, dy = -dx, -dy

        # 归一化方向向量
        length = math.hypot(dx, dy)
        if length > 0:
            directions.append((dx / length, dy / length))
        else:
            directions.append((0.0, 0.0))

    return directions

def bezier_curve_length(p0, p1, p2, p3):
    """计算三阶贝塞尔曲线的长度"""
    def integrand(t):
        dx_dt = 3 * (1 - t) ** 2 * (p1[0] - p0[0]) + 6 * (1 - t) * t * (p2[0] - p1[0]) + 3 * t ** 2 * (p3[0] - p2[0])
        dy_dt = 3 * (1 - t) ** 2 * (p1[1] - p0[1]) + 6 * (1 - t) * t * (p2[1] - p1[1]) + 3 * t ** 2 * (p3[1] - p2[1])
        return np.sqrt(dx_dt ** 2 + dy_dt ** 2)
    return quad(integrand, 0, 1)[0]

def bezier_curve(p0, p1, p2, p3, num_points=None):
    """生成三阶贝塞尔曲线点集，若num_points为None则根据估算长度设置"""
    if num_points is None:
        length = bezier_curve_length(p0, p1, p2, p3)  # 计算贝塞尔曲线的长度
        num_points = int(length * 4)  # 将num_points设置为估算长度的4倍
    curve = []
    for t in np.linspace(0, 1, num_points):
        x = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0]
        y = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1]
        curve.append((x, y))
    return curve

def generate_bezier_path(key_points, directions, base_alpha=0.3):
    """生成贝塞尔曲线路径（带动态参数调整）"""
    smooth_points = []
    for i in range(len(key_points)-1):
        p0, p3 = key_points[i], key_points[i+1]
        dir0, dir3 = directions[i], directions[i+1]

        # 动态alpha调整
        if i > 0:
            prev_dir = directions[i-1]
            alpha = dynamic_alpha(prev_dir, dir0, base_alpha)
        else:
            alpha = base_alpha

        # 自动调整控制点长度
        dx, dy = p3[0] - p0[0], p3[1] - p0[1]
        segment_length = math.hypot(dx, dy)
        k = segment_length * alpha

        # 控制点边界保护
        p1 = (p0[0] + dir0[0] * min(k, segment_length * 0.8), p0[1] + dir0[1] * min(k, segment_length * 0.8))
        p2 = (p3[0] - dir3[0] * min(k, segment_length * 0.8), p3[1] - dir3[1] * min(k, segment_length * 0.8))

        # 生成曲线段
        segment = bezier_curve(p0, p1, p2, p3)
        smooth_points.extend(segment)
    return smooth_points

def angle_between(v1, v2):
    """计算两个方向向量之间的夹角（弧度）"""
    dot, det = v1[0] * v2[0] + v1[1] * v2[1], v1[0] * v2[1] - v1[1] * v2[0]
    return math.atan2(det, dot)  # 范围在[-π, π]

def dynamic_alpha(prev_dir, current_dir, base_alpha):
    """根据方向变化动态调整alpha参数"""
    angle_change = abs(angle_between(prev_dir, current_dir))
    return base_alpha * (1 - angle_change / math.pi)

def calculate_local_curvature(points, window=3):
    """计算路径的局部曲率"""
    curvatures = []
    n = len(points)
    for i in range(window//2, n - window//2):
        segment = points[i-window//2: i+window//2+1]
        total_angle = 0
        for j in range(1, len(segment)-1):
            v1 = (segment[j][0]-segment[j-1][0], segment[j][1]-segment[j-1][1])
            v2 = (segment[j+1][0]-segment[j][0], segment[j+1][1]-segment[j][1])
            total_angle += abs(angle_between(v1, v2))
        curvatures.append(total_angle / (len(segment)-1))
    return np.mean(curvatures) if curvatures else 0

def adaptive_epsilon(path, base_epsilon):
    """根据路径曲率自适应调整epsilon参数"""
    curvature = calculate_local_curvature(path)
    return base_epsilon * (1 - 0.5 * curvature/math.pi)

def smooth_path(raw_path, base_epsilon=2.5, base_alpha=0.25):
    """增强版路径平滑流程"""
    if len(raw_path) < 2:
        return raw_path

    # 自适应epsilon调整
    epsilon = adaptive_epsilon(raw_path, base_epsilon)
    key_points = douglas_pucker(raw_path, epsilon)

    # 处理短路径情况
    if len(key_points) < 3:
        if len(key_points) == 2:
            return bezier_curve(key_points[0], *key_points, key_points[1])
        return key_points

    # 方向场计算
    directions = compute_tangent_directions(key_points)
    return generate_bezier_path(key_points, directions, base_alpha)
