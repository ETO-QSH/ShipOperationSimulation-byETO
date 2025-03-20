import pygame
import numpy as np
from 地图生成 import load_island_map

# 配置参数
ORIGINAL_SIZE = 1024  # 原始地图尺寸
TILE_SCALE = 2  # 每个逻辑单元放大倍数
FULL_SIZE = ORIGINAL_SIZE * TILE_SCALE  # 2048×2048
SCREEN_SIZE = (768, 768)  # 显示窗口尺寸

COLORS = {
    'sea': (31, 159, 255),
    'land': (31, 159, 31),
    'ship': (255, 127, 191)
}


class IslandMap:
    def __init__(self):
        # 加载地图数据
        self.raw_map = load_island_map()
        self.texture = self._create_full_texture()
        self.display_texture = pygame.transform.smoothscale(self.texture, SCREEN_SIZE)

    def _create_full_texture(self):
        """创建2048×2048的完整纹理"""
        texture = pygame.Surface((FULL_SIZE, FULL_SIZE))

        transposed_map = self.raw_map.T  # 交换行列索引, 不然莫名其妙是反的浪费我两个小时（怒）

        # 使用numpy向量化操作加速纹理生成
        land_mask = np.repeat(np.repeat(transposed_map, TILE_SCALE, axis=0), TILE_SCALE, axis=1)

        # 生成颜色矩阵
        color_array = np.where(land_mask[:, :, None], np.array(COLORS['land'], dtype=np.uint8), np.array(COLORS['sea'], dtype=np.uint8))

        # 转换为Pygame Surface
        pygame.surfarray.blit_array(texture, color_array)
        return texture


class Ship:
    def __init__(self):
        """
        船舶模拟类，负责处理船舶物理特性、运动状态和碰撞检测
        """
        # 图形和碰撞配置
        self.config = {
            'start_pos': (FULL_SIZE//2, FULL_SIZE//2),  # 初始地图坐标 (像素)
            'ship_shape': [(0, -30), (-15, 30), (15, 30)],  # 船体三角形顶点 (局部坐标)
            'collision_points': [(0, -30), (-15, 30), (15, 30)],  # 碰撞检测点 (局部坐标)
            'collision_color': (255, 0, 255),  # 正常碰撞框颜色 (BGR)
            'collision_alert': (0, 255, 255),  # 碰撞警告颜色 (BGR)
            'collision_width': 1,  # 碰撞框线宽 (像素)
            'bow_vector_color': (255, 127, 0),  # 船头方向向量颜色 (RGB)
            'rudder_vector_color': (192, 192, 0),  # 舵向方向向量颜色 (RGB)
            'vector_width': 2  # 向量线线宽 (像素)
        }

        # 物理参数配置（所有单位基于像素和秒）
        self.physics_config = {
            'mass': 1500.0,              # 船舶质量 (kg)
            'max_rudder_angle': 30.0,    # 最大舵角 (度)
            'rudder_rate': 30.0,         # 舵角变化速率 (度/秒)
            'rudder_efficiency': 0.005,  # 转向效率系数 (1/像素)
            'rudder_return': 1.5,        # 自动回舵系数
            'propulsion_force': 9000.0,  # 最大推进力 (N)
            'water_resistance': 45.0,    # 线性水阻系数 (N·s/pixel)
            'hull_drag': 0.75,           # 二次水阻系数 (N·s²/pixel²)
            'min_steering_speed': 2.5,   # 最小有效转向速度 (pixel/s)
            'brake_force': 4500.0,       # 刹车力度 (N)
            'side_resistance': 3.2,      # 侧舷转向阻力系数 (N·s²/pixel²)
            'max_gear_forward': 4,       # 最大前进档位
            'max_gear_reverse': 2,       # 最大后退档位
            'neutral_drag': 0.9,         # 空档阻力系数 (N·s²/pixel²)
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

    @property
    def screen_scale(self):
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
            return ratio * self.physics_config['propulsion_force']
        else:
            ratio = abs(self.gear) / self.physics_config['max_gear_reverse']
            return -ratio * self.physics_config['brake_force']

    def _calculate_resistance(self):
        """计算综合阻力（含侧舷转向带来的速度损失）"""
        speed = abs(self.velocity)
        sign = np.sign(self.velocity)

        # 基础阻力 = 线性项 + 二次项
        base_res = self.physics_config['water_resistance'] * speed + self.physics_config['hull_drag'] * speed ** 2

        # 空档附加二次阻力（模拟流体分离效应）
        if self.gear == 0:
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
        rad = np.deg2rad(self.heading)
        dx = self.velocity * np.cos(rad) * dt
        dy = -self.velocity * np.sin(rad) * dt  # Pygame坐标系Y轴向下

        self.map_pos += pygame.Vector2(dx, dy)
        self.map_pos.x = max(0.0, min(self.map_pos.x, FULL_SIZE))
        self.map_pos.y = max(0.0, min(self.map_pos.y, FULL_SIZE))

    def _update_collision_points(self):
        """更新碰撞检测点全局坐标"""
        angle = np.deg2rad(-self.heading + 90)  # 转换为Pygame旋转方向
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        self.collision_global = []

        for x, y in self.config['collision_points']:
            # 应用旋转矩阵
            rot_x, rot_y = x * cos_a - y * sin_a, x * sin_a + y * cos_a
            # 转换为全局坐标
            global_pos = pygame.Vector2(self.map_pos.x + rot_x, self.map_pos.y + rot_y)
            self.collision_global.append(global_pos)

    def draw(self, screen):
        """绘制船舶和碰撞检测元素及向量"""
        center_screen = self.map_pos.x * self.screen_scale, self.map_pos.y * self.screen_scale

        # 船头方向向量
        head_angle = np.deg2rad(self.heading)
        head_end = (
            center_screen[0] + 90 * np.cos(head_angle) * self.screen_scale,
            center_screen[1] - 90 * np.sin(head_angle) * self.screen_scale  # Y轴取反
        )
        pygame.draw.line(screen, self.config['bow_vector_color'], center_screen, head_end, self.config['vector_width'])

        # 舵角方向向量
        rudder_angle = np.deg2rad(self.heading + self.rudder_angle)
        rudder_end = (
            center_screen[0] + 90 * np.cos(rudder_angle) * self.screen_scale,
            center_screen[1] - 90 * np.sin(rudder_angle) * self.screen_scale  # Y轴取反
        )
        pygame.draw.line(screen, self.config['rudder_vector_color'], center_screen, rudder_end, self.config['vector_width'])

        # 绘制船体
        self._draw_ship(screen)
        # 绘制碰撞框
        self._draw_collision(screen)

        # 绘制碰撞点标记
        if self.collision_screen_points:
            for p in self.collision_screen_points:
                pygame.draw.circle(screen, (0, 255, 0), p, 2)

        rad = np.deg2rad(self.heading + 180)  # 正后方方向
        text_offset = 10 * self.screen_scale  # 偏移量
        text_pos = (
            self.map_pos.x * self.screen_scale + np.cos(rad) * text_offset,
            self.map_pos.y * self.screen_scale - np.sin(rad) * text_offset  # 注意Y轴取反
        )

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
        for (x, y) in self.config['ship_shape']:
            # 坐标旋转
            rot_x, rot_y = self._rotate_point(x, y)
            # 坐标转换
            screen_x = (self.map_pos.x + rot_x) * self.screen_scale
            screen_y = (self.map_pos.y + rot_y) * self.screen_scale
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
            pygame.draw.circle(screen, (255, 255, 255), (int(p[0]), int(p[1])), 2)

    def _rotate_point(self, x, y):
        """坐标旋转辅助函数"""
        angle = np.deg2rad(-self.heading + 90)
        return x * np.cos(angle) - y * np.sin(angle), x * np.sin(angle) + y * np.cos(angle)


class NavigationSimulator:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption("高级船舶模拟器")
        self.clock = pygame.time.Clock()
        self.island_map = IslandMap()
        self.ship = Ship()
        self.running = True

    def handle_events(self, dt):
        keys = pygame.key.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self._handle_gear_change(event)

        # 持续按键检测（每帧执行）
        if keys[pygame.K_SPACE]:  # 按住空格持续刹车
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
            # 自动回舵（阻尼效应）
            return_speed = max_rate * self.ship.physics_config['rudder_return'] * dt
            current_angle = abs(self.ship.rudder_angle)
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
            # 转换为原始地图坐标
            raw_x = int(point.x / TILE_SCALE)
            raw_y = int(point.y / TILE_SCALE)
            # 边界保护
            raw_x = np.clip(raw_x, 0, ORIGINAL_SIZE - 1)
            raw_y = np.clip(raw_y, 0, ORIGINAL_SIZE - 1)

            if self.island_map.raw_map[raw_y, raw_x]:
                self.ship.is_colliding = True
                # 记录屏幕坐标
                screen_x = int(point.x * self.ship.screen_scale)
                screen_y = int(point.y * self.ship.screen_scale)
                self.ship.collision_screen_points.append((screen_x, screen_y))

    def run(self):
        """主循环"""
        while self.running:
            dt = self.clock.tick(60) / 1000.0  # 获取真实时间差

            # 处理输入
            self.handle_events(dt)

            # 更新状态
            self.ship.update(dt)
            self.check_collision()

            # 渲染画面
            self.screen.blit(self.island_map.display_texture, (0, 0))
            self.ship.draw(self.screen)
            pygame.display.flip()

        pygame.quit()


if __name__ == "__main__":
    simulator = NavigationSimulator()
    simulator.run()
