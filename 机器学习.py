import torch
import torch.nn as nn
import torch.optim as optim

import random
from pygame import Vector2
from collections import deque

from 地图生成 import np, plt
from 主程序 import Ship, IslandMap, NavigationSimulator, ORIGINAL_SIZE, TILE_SCALE

# 强化学习参数
MEMORY_CAPACITY = 10000
UPDATE_INTERVAL = 100
BATCH_SIZE = 128
GAMMA = 0.99


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
            reward = calculate_reward(ship, simulator.island_map)
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
    torch.save(ship.policy_net.state_dict(), f"ship_dqn.pth")
    
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
    plt.savefig('training_reward_trend.png', dpi=150)
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

def calculate_reward(ship, map_instance):
    """改进的奖励函数，考虑周围区域航道存在性"""
    reward = 0
    detect_radius = 8  # 检测半径

    # 获取当前船舶所在网格坐标
    center_x = int(ship.map_pos.x / TILE_SCALE)
    center_y = int(ship.map_pos.y / TILE_SCALE)

    # 边界保护
    x_min = max(0, center_x - detect_radius)
    x_max = min(ORIGINAL_SIZE, center_x + detect_radius + 1)
    y_min = max(0, center_y - detect_radius)
    y_max = min(ORIGINAL_SIZE, center_y + detect_radius + 1)

    # 创建检测区域网格
    x = np.arange(x_min, x_max)
    y = np.arange(y_min, y_max)
    xx, yy = np.meshgrid(x, y)

    # 计算圆形掩膜（边界自动裁剪）
    distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    circle_mask = distance <= detect_radius

    # 获取区域内的地图数据
    area_data = map_instance.raw_map[y_min:y_max, x_min:x_max]

    # 计算航道存在比例
    channel_pixels = np.sum((area_data == 2) & circle_mask)
    total_pixels = np.sum(circle_mask)
    channel_ratio = channel_pixels / total_pixels if total_pixels > 0 else 0

    # 航道奖励（连续值）
    reward += 3.0 * channel_ratio  # 最大奖励3.0当完全在航道内

    # 碰撞惩罚
    if ship.is_colliding:
        reward -= 25.0

    # 速度奖励（鼓励保持中高速）
    optimal_speed = ship.physics_config["max_speed_forward"]
    speed_diff = abs(ship.velocity - optimal_speed)
    reward += 0.5 * np.exp(-0.1 * speed_diff)  # 峰值奖励0.5

    # 航向稳定性奖励（减少无谓转向）
    reward += 0.02 * (30.0 - abs(ship.rudder_angle))  # 零舵角时获得最大奖励

    return reward


if __name__ == "__main__":
    train()
