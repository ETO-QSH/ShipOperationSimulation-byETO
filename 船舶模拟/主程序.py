import pygame
from 机器学习 import DQN, torch
from 地图生成 import np, load_island_map

# 配置参数
ORIGINAL_SIZE = 1024  # 原始地图尺寸
TILE_SCALE = 2  # 每个逻辑单元放大倍数
FULL_SIZE = ORIGINAL_SIZE * TILE_SCALE  # 2048×2048
SCREEN_SIZE = (768, 768)  # 显示窗口尺寸
COLORS = {'sea': (31, 159, 255), 'land': (31, 159, 31), 'road': (255, 0, 0), 'ship': (255, 127, 191), 'orz': (0, 255, 0)}


class IslandMap:
    def __init__(self):
        """地图信息初始化"""
        # 加载地图数据
        self.raw_map = load_island_map("res\\island_map_A_Star.npy")
        self.texture = self._create_full_texture()
        self.display_texture = pygame.transform.smoothscale(self.texture, SCREEN_SIZE)

        # 当前帧已探索区域记录
        self.current_explored = np.zeros_like(self.raw_map, dtype=int)

    def is_path(self, position):
        x = int(position.x / TILE_SCALE)
        y = int(position.y / TILE_SCALE)
        return self.raw_map[y, x] == 2

    def _create_full_texture(self):
        """创建2048×2048的完整纹理"""
        texture = pygame.Surface((FULL_SIZE, FULL_SIZE))
        transposed_map = self.raw_map.T  # 交换行列索引, 不然莫名其妙是反的浪费我两个小时（怒）

        # 使用numpy向量化操作加速纹理生成
        land_mask = np.repeat(np.repeat(transposed_map, TILE_SCALE, axis=0), TILE_SCALE, axis=1)
        # 颜色矩阵映射
        color_palette = np.array([COLORS['sea'], COLORS['land'], COLORS['road'], COLORS['orz']], dtype=np.uint8)
        # 使用NumPy高级索引直接映射颜色
        color_array = color_palette[land_mask]
        # 转换为Pygame Surface
        pygame.surfarray.blit_array(texture, color_array)
        return texture

    def draw_current_explored(self, screen):
        """在屏幕上绘制当前帧探索到的区域（向量化版本）"""
        # 创建透明图层
        explored_texture = pygame.Surface(SCREEN_SIZE, pygame.SRCALPHA)
        # 获取需要绘制的坐标点
        y_coords, x_coords = np.where(self.current_explored)

        # 转换为屏幕坐标系（使用向量化计算）
        screen_x = (x_coords * TILE_SCALE * (SCREEN_SIZE[0] / FULL_SIZE)).astype(int)
        screen_y = (y_coords * TILE_SCALE * (SCREEN_SIZE[1] / FULL_SIZE)).astype(int)

        # 过滤超出屏幕范围的坐标
        valid = (screen_x >= 0) & (screen_x < SCREEN_SIZE[0]) & (screen_y >= 0) & (screen_y < SCREEN_SIZE[1])
        screen_x, screen_y = screen_x[valid], screen_y[valid]

        # 直接操作像素数组
        pixels = pygame.surfarray.pixels3d(explored_texture)
        alpha = pygame.surfarray.pixels_alpha(explored_texture)

        # 设置颜色为半透明白色
        pixels[screen_x, screen_y] = [255, 255, 255]  # RGB
        alpha[screen_x, screen_y] = 128  # Alpha

        # 释放像素数组锁
        del pixels, alpha
        # 绘制到屏幕
        screen.blit(explored_texture, (0, 0))


class Ship:
    def __init__(self, model_path=None):
        """船舶模拟类，负责处理船舶物理特性、运动状态和碰撞检测"""
        # 图形和碰撞配置
        self.config = {
            'start_pos': (47, 2015),  # 初始地图坐标 (像素)
            'ship_shape': [(0, -24), (-12, 24), (12, 24)],  # 船体三角形顶点 (局部坐标)
            'collision_points': [(0, -24), (-12, 24), (12, 24)],  # 碰撞检测点 (局部坐标)
            'collision_color': (255, 0, 255),  # 正常碰撞框颜色 (BGR)
            'collision_alert': (0, 255, 255),  # 碰撞警告颜色 (BGR)
            'bow_vector_color': (255, 127, 0),  # 船头方向向量颜色 (RGB)
            'rudder_vector_color': (192, 192, 0),  # 舵向方向向量颜色 (RGB)
            'collision_width': 1,  # 碰撞框线宽 (像素)
            'collision_radius': 2,  # 监测点半径 (像素)
            'vector_width': 3,  # 向量线线宽 (像素)
            'vector_length': 72  # 向量线线长 (像素)
        }

        # 物理参数配置（所有单位基于像素和秒）
        self.physics_config = {
            'mass': 2000.0,              # 船舶质量 (kg)
            'max_rudder_angle': 30.0,    # 最大舵角 (度)
            'rudder_rate': 30.0,         # 舵角变化速率 (度/秒)
            'rudder_efficiency': 0.005,  # 转向效率系数 (1/像素)
            'rudder_return': 1.5,        # 自动回舵系数
            'propulsion_force': 6000.0,  # 最大推进力 (N)
            'water_resistance': 45.0,    # 线性水阻系数 (N·s/pixel)
            'hull_drag': 0.75,           # 二次水阻系数 (N·s²/pixel²)
            'min_steering_speed': 5.0,   # 最小有效转向速度 (pixel/s)
            'brake_force': 4000.0,       # 刹车力度 (N)
            'side_resistance': 3.2,      # 侧舷转向阻力系数 (N·s²/pixel²)
            'max_gear_forward': 4,       # 最大前进档位
            'max_gear_reverse': 2,       # 最大后退档位
            'neutral_drag': 0.9,         # 分离阻力系数 (N·s²/pixel²)
            'max_speed_forward': 60.0,   # 最大前进速度 (pixel/s, 原1像素/帧)
            'max_speed_reverse': 30.0    # 最大倒车速度 (pixel/s, 原0.5像素/帧)
        }

        # 动态状态变量
        self.map_pos = pygame.Vector2(self.config['start_pos'])  # 地图坐标 (像素)
        self.velocity = 0.0                # 当前速度 (像素/秒)，正数为前进
        self.heading = 0.0                 # 航向角 (度)，0度指向正右方
        self.rudder_angle = 0.0            # 当前舵角 (度)，左负右正
        self.gear = 0                      # 档位：0=空档，1-4=前进档，-1-2=后退档
        self.collision_global = []         # 全局坐标系的碰撞检测点
        self.collision_screen_points = []  # 屏幕坐标的碰撞标记点
        self.is_colliding = False          # 当前碰撞状态
        self.rl_model = None               # 自动控制状态
        if model_path:
            self.load_model(model_path)

    @property
    def screen_scale(self):
        """地图尺寸和模拟尺寸映射"""
        return SCREEN_SIZE[0] / FULL_SIZE

    def update(self, dt):
        """更新船舶状态"""
        # 推进力计算
        propulsion = self._calculate_propulsion()
        # 阻力计算（包含侧舷转向带来的速度损失）
        resistance = self._calculate_resistance()
        # 物理计算
        acceleration = (propulsion - resistance) / self.physics_config['mass']
        self.velocity += acceleration * dt
        # 速度限制
        self.velocity = np.clip(self.velocity, -self.physics_config['max_speed_reverse'], self.physics_config['max_speed_forward'])
        # 转向更新
        self._update_steering(dt)
        # 位置更新
        self._update_position(dt)
        self._update_collision_points()

    def _calculate_propulsion(self):
        """根据档位计算推进力"""
        if self.gear == 0:
            return 0.0
        elif self.gear > 0:
            ratio = self.gear / self.physics_config['max_gear_forward']
            return ratio**2 * self.physics_config['propulsion_force']
        else:
            ratio = abs(self.gear) / self.physics_config['max_gear_reverse']
            return -ratio**2 * self.physics_config['brake_force']

    def _calculate_resistance(self):
        """计算综合阻力（含侧舷转向带来的速度损失）"""
        speed = abs(self.velocity)
        sign = np.sign(self.velocity)

        # 基础阻力 = 线性项 + 二次项
        base_res = self.physics_config['water_resistance'] * speed + self.physics_config['hull_drag'] * speed ** 2

        # 反向附加二次阻力（模拟流体分离效应）
        if self.gear * self.velocity <= 0:
            base_res += self.physics_config['neutral_drag'] * speed ** 2

        # 侧舷转向阻力（与舵角成正比，速度二次方相关）
        steering_res = (abs(self.rudder_angle) / self.physics_config['max_rudder_angle']) * self.physics_config['side_resistance'] * speed ** 2
        return (base_res + steering_res) * sign

    def _update_steering(self, dt):
        """基于流体动力学的转向模型"""
        if abs(self.velocity) < self.physics_config['min_steering_speed']:
            return  # 速度过低无法转向

        # 有效舵角计算（考虑舵效损失）
        max_angle = self.physics_config['max_rudder_angle']
        effective_angle = self.rudder_angle * (1 - 0.5 * (abs(self.rudder_angle) / max_angle) ** 2)

        # 转向角速度 = 舵角 * 舵效系数 * 速度（与速度平方根相关）
        turn_rate = effective_angle * self.physics_config['rudder_efficiency'] * np.sqrt(abs(self.velocity)) * 60

        self.heading += turn_rate * dt
        self.heading %= 360  # 标准化航向角

    def _update_position(self, dt):
        """对状态微分进行叠加"""
        rad = np.deg2rad(self.heading)
        dx, dy = self.velocity * np.cos(rad) * dt, -self.velocity * np.sin(rad) * dt
        self.map_pos += pygame.Vector2(dx, dy)
        self.map_pos.x, self.map_pos.y = max(0.0, min(self.map_pos.x, FULL_SIZE)), max(0.0, min(self.map_pos.y, FULL_SIZE))

    def _update_collision_points(self):
        """更新碰撞检测点全局坐标"""
        angle = np.deg2rad(-self.heading + 90)  # 转换为Pygame旋转方向
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        self.collision_global = []

        for x, y in self.config['collision_points']:
            rot_x, rot_y = x * cos_a - y * sin_a, x * sin_a + y * cos_a  # 应用旋转矩阵
            global_pos = pygame.Vector2(self.map_pos.x + rot_x, self.map_pos.y + rot_y)  # 转换为全局坐标
            self.collision_global.append(global_pos)

    def draw(self, screen):
        """绘制船舶和碰撞检测元素及向量"""
        center_screen = self.map_pos.x * self.screen_scale, self.map_pos.y * self.screen_scale

        # 船头方向向量
        head_angle = np.deg2rad(self.heading)
        head_end = (center_screen[0] + self.config['vector_length'] * np.cos(head_angle) * self.screen_scale,
                    center_screen[1] - self.config['vector_length'] * np.sin(head_angle) * self.screen_scale)
        pygame.draw.line(screen, self.config['bow_vector_color'], center_screen, head_end, self.config['vector_width'])

        # 舵角方向向量
        rudder_angle = np.deg2rad(self.heading + self.rudder_angle)
        rudder_end = (center_screen[0] + self.config['vector_length'] * np.cos(rudder_angle) * self.screen_scale,
                      center_screen[1] - self.config['vector_length'] * np.sin(rudder_angle) * self.screen_scale)
        pygame.draw.line(screen, self.config['rudder_vector_color'], center_screen, rudder_end, self.config['vector_width'])

        # 绘制船体碰撞框
        self._draw_ship(screen)
        self._draw_collision(screen)

        # 绘制碰撞点标记
        if self.collision_screen_points:
            for p in self.collision_screen_points:
                pygame.draw.circle(screen, (0, 255, 0), p, self.config['collision_radius'])

        rad = np.deg2rad(self.heading + 180)  # 正后方方向
        text_offset = 4 * self.screen_scale  # 偏移量
        text_pos = self.map_pos.x * self.screen_scale + np.cos(rad) * text_offset, self.map_pos.y * self.screen_scale - np.sin(rad) * text_offset

        # 创建旋转字体对象
        font = pygame.font.SysFont('Arial', 12, bold=True)
        text = f"D{self.gear}" if self.gear > 0 else f"R{-self.gear}" if self.gear < 0 else "N0"
        color = (0, 191, 0) if self.gear > 0 else (255, 0, 0) if self.gear < 0 else (31, 31, 31)

        # 渲染旋转文本（保持文字方向与船头一致）
        text_surface = font.render(text, True, color)

        # 旋转文本表面（保持文字与船体同向）
        rotated_surface = pygame.transform.rotate(text_surface, self.heading)
        text_rect = rotated_surface.get_rect(center=(int(text_pos[0]), int(text_pos[1])))

        # 绘制文字到屏幕
        screen.blit(rotated_surface, text_rect)

    def _draw_ship(self, screen):
        """绘制船体三角形"""
        screen_points = []
        for x, y in self.config['ship_shape']:
            # 坐标旋转转换
            rot_x, rot_y = self._rotate_point(x, y)
            screen_x, screen_y = (self.map_pos.x + rot_x) * self.screen_scale, (self.map_pos.y + rot_y) * self.screen_scale
            screen_points.append((screen_x, screen_y))
        pygame.draw.polygon(screen, COLORS['ship'], screen_points)

    def _draw_collision(self, screen):
        """绘制碰撞检测框"""
        color = self.config['collision_alert'] if self.is_colliding else self.config['collision_color']
        screen_points = [(p.x * self.screen_scale, p.y * self.screen_scale) for p in self.collision_global]

        # 绘制多边形轮廓
        pygame.draw.polygon(screen, color, screen_points, self.config['collision_width'])
        # 绘制顶点标记
        for p in screen_points:
            pygame.draw.circle(screen, (255, 255, 255), (int(p[0]), int(p[1])), self.config['collision_radius'])

    def _rotate_point(self, x, y):
        """坐标旋转辅助函数"""
        angle = np.deg2rad(-self.heading + 90)
        return x * np.cos(angle) - y * np.sin(angle), x * np.sin(angle) + y * np.cos(angle)

    def radar_detect(self, map_instance, radius):
        """检测指定半径范围内的地图区域，并更新已探索区域"""
        # 每帧清空当前帧已探索区域记录
        map_instance.current_explored = np.zeros_like(map_instance.raw_map, dtype=bool)

        # 计算以船为中心的圆形区域的边界
        center_x, center_y = int(self.map_pos.x / TILE_SCALE), int(self.map_pos.y / TILE_SCALE)
        left, right = max(0, center_x - radius), min(ORIGINAL_SIZE, center_x + radius)
        top, bottom = max(0, center_y - radius), min(ORIGINAL_SIZE, center_y + radius)

        # 创建圆形遮罩
        y, x = np.ogrid[top:bottom, left:right]
        circle_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2

        # 找到圆形区域内的陆地像素
        land_pixels = np.where(np.logical_and(circle_mask, map_instance.raw_map[top:bottom, left:right]))
        # 更新当前帧已探索区域
        map_instance.current_explored[top:bottom, left:right][land_pixels] = True

    def load_model(self, path):
        """加载训练好的模型（兼容性修正）"""
        self.rl_model = DQN(grid_size=32)  # 必须与训练时的网格尺寸一致
        self.rl_model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))  # 确保兼容不同设备
        self.rl_model.eval()

    def get_state(self, map_instance):
        """获取当前状态（从RLNavigator移植）"""
        grid_size = 32  # 必须与训练时的网格尺寸一致

        # 获取雷达网格
        center_x = int(self.map_pos.x / TILE_SCALE)
        center_y = int(self.map_pos.y / TILE_SCALE)
        grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        half = grid_size // 2

        for i in range(-half, half):
            for j in range(-half, half):
                x = center_x + i
                y = center_y + j
                if 0 <= x < ORIGINAL_SIZE and 0 <= y < ORIGINAL_SIZE:
                    grid[j + half, i + half] = map_instance.raw_map[y, x]

        # 标准化船舶状态
        state_vec = np.array([
            self.velocity / self.physics_config['max_speed_forward'],
            self.heading / 360.0,
            self.rudder_angle / self.physics_config['max_rudder_angle'],
            self.gear / self.physics_config['max_gear_forward']
        ], dtype=np.float32)

        return grid, state_vec

    def auto_control(self, map_instance):
        """使用模型自动控制（完整实现）"""
        # 获取当前状态
        grid, state = self.get_state(map_instance)

        # 模型推理
        with torch.no_grad():
            grid_tensor = torch.FloatTensor(grid).unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = self.rl_model(grid_tensor, state_tensor).argmax().item()

        # 执行动作
        self._execute_rl_action(action)

    def _execute_rl_action(self, action):
        """将动作编号转换为控制指令"""
        # 动作定义：{0: 左转, 1: 右转, 2: 加速, 3: 减速, 4: 保持}
        rudder_step = 5  # 每次舵角变化量

        if action == 0:  # 左转
            self.rudder_angle = max(-self.physics_config['max_rudder_angle'], self.rudder_angle - rudder_step)
        elif action == 1:  # 右转
            self.rudder_angle = min(self.physics_config['max_rudder_angle'], self.rudder_angle + rudder_step)
        elif action == 2:  # 加速
            if self.gear < self.physics_config['max_gear_forward']:
                self.gear += 1
        elif action == 3:  # 减速
            if self.gear > -self.physics_config['max_gear_reverse']:
                self.gear -= 1


class NavigationSimulator:
    def __init__(self):
        """窗口对象初始化"""
        pygame.init()
        pygame.display.set_caption("高级船舶模拟器")
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        self.clock = pygame.time.Clock()
        self.island_map = IslandMap()
        self.ship = Ship()
        self.running = True

    def handle_events(self, dt):
        """处理键盘事件"""
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self._handle_gear_change(event)

        # 持续按键检测（每帧执行）
        if keys[pygame.K_SPACE]:  # 按住空格持续刹车
            if self.ship.velocity < self.ship.physics_config['min_steering_speed']:
                self.ship.velocity = 0  # 抛锚抛锚
            self._apply_brake(dt)
        self._handle_rudder_control(keys, dt)

    def _handle_gear_change(self, event):
        """处理档位切换"""
        if event.key == pygame.K_w:  # 升档
            if self.ship.gear < self.ship.physics_config['max_gear_forward']:
                self.ship.gear += 1
        elif event.key == pygame.K_s:  # 降档
            if self.ship.gear > -self.ship.physics_config['max_gear_reverse']:
                self.ship.gear -= 1

    def _handle_rudder_control(self, keys, dt):
        """舵角控制"""
        rudder_input = keys[pygame.K_d] - keys[pygame.K_a]
        max_rate = self.ship.physics_config['rudder_rate']

        if rudder_input != 0:
            delta = rudder_input * max_rate * dt
            self.ship.rudder_angle += delta
        else:
            return_speed = max_rate * self.ship.physics_config['rudder_return'] * dt
            current_angle = abs(self.ship.rudder_angle)  # 自动回舵（阻尼效应）
            if current_angle > return_speed:
                self.ship.rudder_angle -= np.sign(self.ship.rudder_angle) * return_speed
            else:
                self.ship.rudder_angle = 0

        # 舵角限制
        max_angle = self.ship.physics_config['max_rudder_angle']
        self.ship.rudder_angle = np.clip(self.ship.rudder_angle, -max_angle, max_angle)

    def _apply_brake(self, dt):
        """刹车系统"""
        if abs(self.ship.velocity) > 0.0:
            brake_force = self.ship.physics_config['brake_force']
            delta_v = (brake_force / self.ship.physics_config['mass']) * dt
            self.ship.velocity -= np.sign(self.ship.velocity) * delta_v

    def check_collision(self):
        """精确碰撞检测"""
        self.ship.is_colliding = False
        self.ship.collision_screen_points = []
        for point in self.ship.collision_global:
            raw_x, raw_y = int(point.x / TILE_SCALE), int(point.y / TILE_SCALE)  # 转换为原始地图坐标
            raw_x, raw_y = np.clip(raw_x, 0, ORIGINAL_SIZE - 1), np.clip(raw_y, 0, ORIGINAL_SIZE - 1)  # 边界保护
            if self.island_map.raw_map[raw_y, raw_x] == 1:
                self.ship.is_colliding = True
                screen_x, screen_y = int(point.x * self.ship.screen_scale), int(point.y * self.ship.screen_scale)
                self.ship.collision_screen_points.append((screen_x, screen_y))  # 记录屏幕坐标

    def run(self):
        """主循环"""
        while self.running:
            dt = self.clock.tick(60) / 1000.0  # 获取真实时间差

            if self.ship.rl_model is None:
                self.handle_events(dt)  # 处理输入
            else:
                self.ship.auto_control(self.island_map)  # 自动控制

            # 更新状态
            self.ship.update(dt)
            self.check_collision()

            # 雷达检测
            self.ship.radar_detect(self.island_map, 128)  # 雷达半径为128像素

            # 渲染画面
            self.screen.blit(self.island_map.display_texture, (0, 0))
            self.island_map.draw_current_explored(self.screen)  # 绘制当前帧探索到的区域

            self.ship.draw(self.screen)
            pygame.display.flip()

        pygame.quit()


if __name__ == "__main__":
    simulator = NavigationSimulator()
    # simulator.ship = Ship(model_path="res\\ship_dqn.pth")
    simulator.run()
