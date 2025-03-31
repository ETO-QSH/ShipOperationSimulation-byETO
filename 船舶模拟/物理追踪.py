import queue
import itertools
import threading
from tqdm import tqdm
from pygame import Vector2
from 地图生成 import np, plt
from sklearn.neighbors import KDTree
from 主程序 import Ship, IslandMap, TILE_SCALE, ORIGINAL_SIZE

# 配置参数
THREADS = 8  # 并行线程数
PLAN_DEPTH = 6  # 操作序列深度
PREDICT_STEPS = 9  # 预测步长
TOTAL_FRAMES = 600  # 总帧数
BATCH_SIZE = 60  # CSV行存储帧数


def _apply_action(ship, action):
    """执行优化后的动作处理"""
    if action == 0:  # 左转
        ship.rudder_angle = max(-30, ship.rudder_angle - 5)
    elif action == 1:  # 右转
        ship.rudder_angle = min(30, ship.rudder_angle + 5)
    elif action == 2:  # 加速
        ship.gear = min(4, ship.gear + 1)
    elif action == 3:  # 减速
        ship.gear = max(-2, ship.gear - 1)
    # 保持动作（4）无需处理

def _get_available_actions(current_gear):
    """获取可用动作集合（带剪枝）"""
    base_actions = [0, 1, 4]  # 左转、右转、保持
    if current_gear < 4:
        base_actions.append(2)  # 加速
    if current_gear > -2:
        base_actions.append(3)  # 减速
    return base_actions

def _generate_action_sequences(current_gear):
    """生成优化后的动作序列组合"""
    first_actions = _get_available_actions(current_gear)  # 第一个动作包含变速选项
    follow_actions = [0, 1, 4]  # 后续动作仅转向和保持

    sequences = []
    for first in first_actions:
        for seq in itertools.product(follow_actions, repeat=PLAN_DEPTH - 1):
            sequences.append((first,) + seq)
    return sequences


class OptimizedPlanner:
    def __init__(self):
        self.island_map = IslandMap()
        self.raw_map = self.island_map.raw_map
        self.ship_template = Ship()
        self.vis_map = self._init_vis_map()
        self.kdtree = self._build_kdtree()

        # 多线程组件
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.lock = threading.Lock()
        self.progress = 0
        self.action_buffer = []

    def _init_vis_map(self):
        """初始化可视化地图"""
        vis_map = self.raw_map.copy()
        vis_map[vis_map == 2] = 0
        return vis_map

    def _build_kdtree(self):
        """构建路径点的KD树（物理坐标系）"""
        y_coords, x_coords = np.where(self.raw_map == 2)
        physical_points = np.column_stack((x_coords * TILE_SCALE + TILE_SCALE // 2, y_coords * TILE_SCALE + TILE_SCALE // 2))
        return KDTree(physical_points) if len(physical_points) > 0 else None

    def _calculate_cost(self, position):
        """计算综合代价（距离+速度）"""
        if self.kdtree is None:
            return 0
        distance, _ = self.kdtree.query([position], k=1)
        speed_cost = 5.0 * (1 - abs(self.current_gear) / 4)
        return distance[0][0] + speed_cost

    def _simulate_sequence(self, ship_state, actions):
        """模拟单个操作序列"""
        ship = self._clone_ship(ship_state)
        total_cost = 0

        # 执行规划动作
        for action in actions:
            _apply_action(ship, action)
            ship.update(1 / 60)
            total_cost += self._calculate_cost(ship.map_pos)

        # 执行预测动作
        for _ in range(PREDICT_STEPS):
            ship.update(1 / 60)
            total_cost += self._calculate_cost(ship.map_pos)

        return total_cost

    def _worker(self):
        """工作线程函数"""
        while True:
            try:
                frame_data = self.task_queue.get(timeout=1)
                frame_idx, ship_state, sequences = frame_data
                min_cost, best_action = float('inf'), 4

                for seq in sequences:
                    cost = self._simulate_sequence(ship_state, seq)
                    if cost < min_cost:
                        min_cost = cost
                        best_action = seq[0]

                self.result_queue.put((frame_idx, best_action, min_cost))
                with self.lock:
                    self.progress += 1

            except queue.Empty:
                break

    def _clone_ship(self, ship):
        """深度克隆船舶状态（包含物理配置）"""
        new_ship = Ship()
        new_ship.map_pos = Vector2(ship.map_pos)
        new_ship.velocity = ship.velocity
        new_ship.heading = ship.heading
        new_ship.rudder_angle = ship.rudder_angle
        new_ship.gear = ship.gear
        new_ship.physics_config = ship.physics_config.copy()
        self.current_gear = new_ship.gear
        return new_ship

    def _update_path(self, position):
        """更新路径可视化"""
        y = int(position.y / TILE_SCALE + 0.5)
        x = int(position.x / TILE_SCALE + 0.5)
        for dy, dx in [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]:
            ny = np.clip(y + dy, 0, ORIGINAL_SIZE - 1)
            nx = np.clip(x + dx, 0, ORIGINAL_SIZE - 1)
            self.vis_map[ny, nx] = 3

    def run(self):
        """主运行函数"""
        threads = [threading.Thread(target=self._worker) for _ in range(THREADS)]  # 初始化线程池
        for t in threads:
            t.start()

        ship_state = self._clone_ship(self.ship_template)  # 初始化船舶状态
        pbar = tqdm(total=TOTAL_FRAMES, desc="Optimized Precomputing")  # 进度条初始化

        for frame in range(TOTAL_FRAMES):
            sequences = _generate_action_sequences(ship_state.gear)  # 生成当前可用序列
            seq_batches = np.array_split(list(set(sequences)), THREADS)  # 分割任务
            for batch in seq_batches:
                if len(batch) > 0:
                    self.task_queue.put((frame, ship_state, batch.tolist()))

            # 收集结果
            best_actions = []
            for _ in range(THREADS):
                if not self.task_queue.empty():
                    result = self.result_queue.get()
                    best_actions.append(result[1:])

            final_action = sorted(best_actions, key=lambda x: x[1])[0][0] if best_actions else 4  # 选择最优动作
            _apply_action(ship_state, final_action)  # 应用动作并
            ship_state.update(1 / 60)

            self.action_buffer.append(final_action)
            self._update_path(ship_state.map_pos)  # 更新状态

            if len(self.action_buffer) >= BATCH_SIZE:  # 每60帧保存一次
                self._save_batch()
                self.action_buffer = []

            pbar.update(1)
            pbar.set_postfix_str(f"Gear: {ship_state.gear}, Action: {final_action}")

        # 保存剩余数据
        if len(self.action_buffer) > 0:
            self._save_batch()

        pbar.close()

        # 结束线程
        for t in threads:
            t.join()

        self._save_visualization()

    def _save_batch(self):
        """优化后的保存方法确保格式正确"""
        if len(self.action_buffer) < BATCH_SIZE:
            return

        # 提取60个动作并转换为字符串
        actions = self.action_buffer[:BATCH_SIZE]
        action_str = ','.join(map(str, actions))

        # 直接写入CSV（追加模式）
        with open('res\\opt_cmds.csv', 'a') as f:
            f.write(action_str + '\n')

        self.action_buffer = self.action_buffer[BATCH_SIZE:]  # 更新缓冲区
        np.save('res\\island_map_lattice.npy', self.vis_map)  # 保存地图数据

    def _save_visualization(self):
        """生成路径可视化"""
        plt.imshow(self.vis_map, cmap='terrain')
        plt.title('Optimized Navigation Path', pad=15, fontweight='bold')
        plt.show()


if __name__ == "__main__":
    planner = OptimizedPlanner()
    planner.run()
