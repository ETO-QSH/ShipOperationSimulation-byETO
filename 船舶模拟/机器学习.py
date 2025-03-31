import torch
import random
import torch.nn as nn
import torch.optim as optim
from pygame import Vector2
from 地图生成 import np, plt
from collections import deque

# 强化学习参数
MEMORY_CAPACITY = 10000
UPDATE_INTERVAL = 100
BATCH_SIZE = 128
GAMMA = 0.98


class DQN(nn.Module):
    """深度Q网络，处理雷达网格和船舶状态"""
    def __init__(self, grid_size=32, state_dim=4):
        super(DQN, self).__init__()

        # 雷达特征提取（修正卷积参数）
        self.grid_size = grid_size
        self.conv = nn.Sequential(
            # 第一层：32x32 → 16x16
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 第二层：16x16 → 16x16（修正padding为2）
            nn.Conv2d(16, 32, kernel_size=3, padding=2, dilation=2),  # 关键修正点
            nn.ReLU(),

            # 第三层：16x16 → 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 状态特征提取
        self.state_fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU()
        )

        # 决策层（修正输入维度）
        self.fc = nn.Sequential(
            nn.Linear(8 * 8 * 64 + 64, 256),  # 正确维度计算
            nn.ReLU(),
            nn.Linear(256, 5)
        )

    def forward(self, grid, state):
        # 处理雷达输入
        grid_feat = self.conv(grid)
        grid_feat = grid_feat.view(grid.size(0), -1)  # 形状变为 [batch, 8*8*64]

        # 处理状态输入
        state_feat = self.state_fc(state)  # 形状变为 [batch, 64]

        # 合并特征
        combined = torch.cat([grid_feat, state_feat], dim=1)  # 合并后维度4160

        return self.fc(combined)


class ReplayMemory:
    """优先经验回放"""
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def train():
    """训练主循环"""
    simulator = NavigationSimulator()
    ship = RLNavigator()
    simulator.ship = ship  # 替换为RL控制的船

    episode_rewards = []
    for episode in range(1000):
        # 初始化环境
        simulator.island_map = IslandMap()
        state_grid, state_vec = ship.get_state(simulator.island_map)
        ship.map_pos = Vector2(ship.config['start_pos'])
        ship.velocity = 0.0
        ship.heading = 0.0
        total_reward = 0
        done = False

        while not done and simulator.running:
            # 选择并执行动作
            action = ship.choose_action(state_grid, state_vec, epsilon=0.2)
            execute_action(ship, action)

            # 更新环境
            dt = simulator.clock.tick(60) / 1000.0
            ship.update(dt)
            simulator.check_collision()

            # 获取新状态
            next_grid, next_vec = ship.get_state(simulator.island_map)

            # 计算奖励
            reward = calculate_reward(ship, simulator.island_map, action)
            total_reward += reward

            # 记录经验
            done = ship.is_colliding
            ship.memory.push((state_grid, state_vec, action, reward, next_grid, next_vec, done))

            # 训练模型
            ship.update_model()

            state_grid, state_vec = next_grid, next_vec

        episode_rewards.append(total_reward)
        print(f"Episode {episode}, Reward: {total_reward:.1f}")

        # 定期保存模型
        if episode % 100 == 0:
            torch.save(ship.policy_net.state_dict(), f"dqn\\ship_{episode:03}.pth")

    # 保存最终模型
    torch.save(ship.policy_net.state_dict(), f"res\\ship_dqn.pth")

    # 训练结束后绘制趋势图
    plt.figure(figsize=(12, 6))

    # 自动计算坐标轴范围, 动态取整
    max_reward, min_reward = np.max(episode_rewards), np.min(episode_rewards)
    max_y, min_y = np.ceil(max_reward / 50) * 50, np.floor(min_reward / 50) * 50

    # 创建柱状图
    bars = plt.bar(range(len(episode_rewards)), episode_rewards, color='#4CAF50', edgecolor='grey', alpha=0.8)

    # 标注最大值和最小值
    max_idx = np.argmax(episode_rewards)
    min_idx = np.argmin(episode_rewards)
    bars[max_idx].set_color('#FF5722')
    bars[min_idx].set_color('#2196F3')

    # 设置坐标轴
    plt.ylim(min_y, max_y)
    plt.xticks(range(0, len(episode_rewards), max(1, len(episode_rewards) // 20)))
    plt.ylabel('Episode Reward')
    plt.xlabel('Episode Number')
    plt.title('Training Progress - Reward Trend', pad=15, fontweight='bold')

    # 自动调整布局
    plt.tight_layout()

    # 保存并显示图表
    plt.savefig('res\\training_reward_trend.png', dpi=150)
    plt.show()

def execute_action(ship, action):
    """将动作编号转换为控制指令"""
    # 动作定义：{0: 左转, 1: 右转, 2: 加速, 3: 减速, 4: 保持}
    if action == 0:
        ship.rudder_angle = max(-30, ship.rudder_angle - 5)
    elif action == 1:
        ship.rudder_angle = min(30, ship.rudder_angle + 5)
    elif action == 2 and ship.gear < 4:
        ship.gear += 1
    elif action == 3 and ship.gear > -2:
        ship.gear -= 1

def calculate_reward(ship, map_instance, action):
    """基于新奖励方案的奖励函数"""

    # 基础分计算 ----------------------------------------------
    def get_zone_score(center_points, radius, max_score, weight_func):
        """计算指定区域的路径得分（添加除零保护）"""
        total = 0
        for point in center_points:
            map_x = int(point.x / TILE_SCALE)
            map_y = int(point.y / TILE_SCALE)
            local_path_mask, weights = generate_detection_grid(
                map_instance,
                center=(map_x, map_y),
                radius=radius,
                weight_func=weight_func
            )

            # 安全校验
            sum_weights = np.sum(weights)
            if sum_weights < 1e-3:  # 浮点数精度保护
                score = 0.0
            else:
                score = np.sum(local_path_mask * weights) / sum_weights

            # 添加区域有效性校验
            if np.sum(local_path_mask) == 0:  # 无路径区域
                score *= 0.8  # 轻微惩罚

            total += score * max_score
        return total

    # 各区域得分计算
    head_score = get_zone_score(
        center_points=[ship.collision_global[0]],
        radius=8,
        max_score=3,
        weight_func=lambda d: np.exp(-0.5 * (d / 2.5) ** 2)  # 高斯衰减
    )

    heart_score = get_zone_score(
        center_points=[ship.map_pos],
        radius=12,
        max_score=1.5,
        weight_func=lambda d: 1 - d / 12  # 线性衰减
    )

    tail_score = get_zone_score(
        center_points=get_rectangular_points(ship, density=0.8),
        radius=4,
        max_score=0.5,
        weight_func=lambda d: np.exp(-d ** 2 / 8)  # 高斯衰减
    )

    # 速度档位倍率
    gear_rates = {-2: -0.25, -1: 0, 0: 0.25, 1: 0.75, 2: 1.50, 3: 2.50, 4: 3.75}
    gear_rate = gear_rates.get(ship.gear, 0)

    # 舵角倍率 (1 - |angle|/60)
    rudder_rate = 1 - abs(ship.rudder_angle) / 60

    # 基础分
    base_score = (head_score + heart_score + tail_score) * gear_rate * rudder_rate

    # 修正分计算 ----------------------------------------------
    # 1. 路径偏离修正
    correction = 1.0
    if head_score < 0.1 or heart_score < 0.1:  # 关键区域无路径
        # 获取船头左右两点连线方向
        bow_left = ship.collision_global[1]
        bow_right = ship.collision_global[2]
        path_center = find_nearest_path_center(map_instance, ship.map_pos)

        # 计算偏离方向
        offset = (bow_left - path_center).cross(bow_right - path_center)
        if offset > 0:  # 路径在右侧
            if action == 0:
                correction *= 0.33  # 左转惩罚
            elif action == 1:
                correction *= 3.0  # 右转奖励
        else:  # 路径在左侧
            if action == 0:
                correction *= 3.0
            elif action == 1:
                correction *= 0.33

    # 2. 路径走向贴合修正
    sector_points = get_sector_path_points(map_instance, ship, radius=30)
    if len(sector_points) > 0:
        # 计算平均方向向量
        avg_vec = Vector2(0, 0)
        for p in sector_points:
            avg_vec += (p - ship.map_pos).normalize()
        avg_dir = avg_vec.angle_to(ship.collision_global[0] - ship.map_pos)

        # 角度差处理
        angle_diff = abs(avg_dir)
        if angle_diff <= 15:
            direction_bonus = 3.0
            turn_penalty = 0.5
        elif angle_diff <= 30:
            direction_bonus = 1.5 + 1.5 * (30 - angle_diff) / 15
            turn_penalty = 0.75
        else:
            direction_bonus = 0.2
            turn_penalty = 1.0

        # 转向修正
        if avg_dir > 0:  # 需要左转
            if action == 0:
                correction *= turn_penalty
            elif action == 1:
                correction /= turn_penalty
        else:  # 需要右转
            if action == 0:
                correction /= turn_penalty
            elif action == 1:
                correction *= turn_penalty

        base_score *= direction_bonus

    # 最终得分整合
    total_reward = base_score * correction

    # 碰撞惩罚
    if ship.is_colliding:
        total_reward -= 256

    return total_reward

def find_nearest_path_center(map_instance, position):
    """寻找最近的路径区域中心"""
    map_x = int(position.x / TILE_SCALE)
    map_y = int(position.y / TILE_SCALE)

    # 搜索半径逐步扩大
    for r in range(10, 100, 10):
        x_min = max(0, map_x - r)
        x_max = min(ORIGINAL_SIZE, map_x + r)
        y_min = max(0, map_y - r)
        y_max = min(ORIGINAL_SIZE, map_y + r)

        # 寻找路径点
        path_points = np.argwhere(map_instance.raw_map[y_min:y_max, x_min:x_max] == 2)
        if len(path_points) > 0:
            avg_y = np.mean(path_points[:, 0]) + y_min
            avg_x = np.mean(path_points[:, 1]) + x_min
            return Vector2(avg_x * TILE_SCALE, avg_y * TILE_SCALE)

    return position  # 未找到时返回当前位置

def get_sector_path_points(map_instance, ship, radius=30):
    """获取船头扇形区域内的路径点"""
    sector_points = []
    bow_dir = (ship.collision_global[0] - ship.map_pos).normalize()
    for angle in range(-30, 31, 5):
        for r in range(5, radius + 1, 5):
            dir_vec = bow_dir.rotate(angle) * r
            check_pos = ship.map_pos + dir_vec
            # 转换为地图坐标检查
            map_x = int(check_pos.x / TILE_SCALE)
            map_y = int(check_pos.y / TILE_SCALE)
            if 0 <= map_x < ORIGINAL_SIZE and 0 <= map_y < ORIGINAL_SIZE:
                if map_instance.raw_map[map_y, map_x] == 2:
                    sector_points.append(check_pos)
    return sector_points

def generate_detection_grid(map_instance, center, radius, weight_func):
    """返回局部路径掩码和权重矩阵"""
    x_min = max(0, center[0] - radius)
    x_max = min(map_instance.raw_map.shape[1], center[0] + radius + 1)
    y_min = max(0, center[1] - radius)
    y_max = min(map_instance.raw_map.shape[0], center[1] + radius + 1)

    if x_min >= x_max or y_min >= y_max:
        return np.zeros((0, 0), dtype=float), 0.0

    # 生成局部地图切片
    local_map = map_instance.raw_map[y_min:y_max, x_min:x_max]
    path_mask = (local_map == 2).astype(float)

    # 生成权重矩阵
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max) - center[0],
        np.arange(y_min, y_max) - center[1]
    )
    distances = np.sqrt(xx ** 2 + yy ** 2)
    weights = np.where(distances <= radius, weight_func(distances), 0)

    return path_mask, weights

def get_rectangular_points(ship, density=0.5):
    """生成船尾矩形检测点"""
    p1, p2 = ship.collision_global[1], ship.collision_global[2]  # 获取船尾左右两点
    length, width = 32, 2  # 计算矩形参数

    # 生成采样点
    points = []
    for t in np.arange(0, 1, density / length):
        base_point = p1.lerp(p2, t)
        direction = ship.collision_global[0] - ship.map_pos  # 船头方向
        perp_direction = Vector2(-direction.y, direction.x).normalize()

        for w in np.arange(-width / 2, width / 2, density):
            points.append(base_point + w * perp_direction)

    return points


if __name__ == "__main__":
    from 主程序 import Ship, IslandMap, NavigationSimulator, ORIGINAL_SIZE, TILE_SCALE

    class RLNavigator(Ship):
        """强化学习船舶控制"""
        def __init__(self):
            super().__init__()
            self.grid_size = 32  # 雷达观测网格尺寸
            self.policy_net = DQN(grid_size=self.grid_size)
            self.target_net = DQN(grid_size=self.grid_size)
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
            self.memory = ReplayMemory(MEMORY_CAPACITY)
            self.step_count = 0

        def get_state(self, map_instance):
            """获取当前状态"""
            # 获取雷达网格
            center_x = int(self.map_pos.x / 2)
            center_y = int(self.map_pos.y / 2)
            grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            half = self.grid_size // 2
            for i in range(-half, half):
                for j in range(-half, half):
                    x = center_x + i
                    y = center_y + j
                    if 0 <= x < 1024 and 0 <= y < 1024:
                        grid[j + half, i + half] = map_instance.raw_map[y, x]

            # 标准化船舶状态
            state_vec = np.array([
                self.velocity / 60.0,
                self.heading / 360.0,
                self.rudder_angle / 30.0,
                self.gear / 4.0
            ], dtype=np.float32)

            return grid, state_vec

        def choose_action(self, grid, state, epsilon=0.1):
            """ε-greedy策略选择动作"""
            if random.random() < epsilon:
                return random.randint(0, 4)
            else:
                with torch.no_grad():
                    grid_tensor = torch.FloatTensor(grid).unsqueeze(0).unsqueeze(0)
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = self.policy_net(grid_tensor, state_tensor)
                return q_values.argmax().item()

        def update_model(self):
            """训练网络"""
            if len(self.memory) < BATCH_SIZE:
                return

            # 从记忆库采样
            transitions = self.memory.sample(BATCH_SIZE)
            batch = list(zip(*transitions))

            # 转换为张量
            grid_batch = torch.FloatTensor(np.stack(batch[0])).unsqueeze(1)
            state_batch = torch.FloatTensor(np.stack(batch[1]))
            action_batch = torch.LongTensor(batch[2])
            reward_batch = torch.FloatTensor(batch[3])
            next_grid_batch = torch.FloatTensor(np.stack(batch[4])).unsqueeze(1)
            next_state_batch = torch.FloatTensor(np.stack(batch[5]))
            done_batch = torch.FloatTensor(batch[6])

            # 计算当前Q值
            current_q = self.policy_net(grid_batch, state_batch).gather(1, action_batch.unsqueeze(1))

            # 计算目标Q值
            with torch.no_grad():
                next_q = self.target_net(next_grid_batch, next_state_batch).max(1)[0]
                target_q = reward_batch + (1 - done_batch) * GAMMA * next_q

            # 计算损失
            loss = nn.MSELoss()(current_q.squeeze(), target_q)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 定期更新目标网络
            if self.step_count % UPDATE_INTERVAL == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            self.step_count += 1

    train()
