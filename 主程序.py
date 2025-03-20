import pygame
import numpy as np
from 地图生成 import load_island_map

# 配置参数
ORIGINAL_SIZE = 1024  # 原始地图尺寸
TILE_SCALE = 2  # 每个逻辑单元放大倍数
FULL_SIZE = ORIGINAL_SIZE * TILE_SCALE  # 2048×2048
SCREEN_SIZE = (768, 768)  # 显示窗口尺寸

COLORS = {
    'sea': (30, 144, 255),
    'land': (34, 139, 34),
    'ship': (255, 0, 0)
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

        transposed_map = self.raw_map.T  # 交换行列索引

        # 使用numpy向量化操作加速纹理生成
        land_mask = np.repeat(np.repeat(transposed_map, TILE_SCALE, axis=0), TILE_SCALE, axis=1)

        # 生成颜色矩阵
        color_array = np.where(land_mask[:, :, None], np.array(COLORS['land'], dtype=np.uint8), np.array(COLORS['sea'], dtype=np.uint8))

        # 转换为Pygame Surface
        pygame.surfarray.blit_array(texture, color_array)
        return texture


class Ship:
    def __init__(self):
        # 参数配置
        self.config = {
            # 初始位置和运动参数
            'start_pos': (FULL_SIZE // 2, FULL_SIZE // 2),
            'base_speed': 0.0,
            'heading': 0.0,  # 初始航向（度）
            'turn_rate': 1.0,  # 转向速率（度/帧）
            'acceleration': 0.02,  # 加速度
            'deceleration': 0.03,  # 减速度
            'max_forward': 1.0,  # 最大前进速度
            'max_reverse': -0.5,  # 最大倒车速度

            # 船舶图形参数
            'ship_shape': [(0, -30), (-15, 30), (15, 30)],  # 船体顶点（局部坐标）
            'collision_points': [(0, -30), (-15, 30), (15, 30)],  # 碰撞检测点（局部坐标）
            'collision_color': (255, 0, 255),  # 正常颜色
            'collision_alert': (0, 255, 255),  # 碰撞颜色
            'collision_width': 2  # 碰撞线宽
        }

        # 动态状态
        self.map_pos = pygame.Vector2(self.config['start_pos'])
        self.speed = self.config['base_speed']
        self.heading = self.config['heading']
        self.collision_screen_points = []
        self.collision_global = []
        self.is_colliding = False

    @property
    def screen_pos(self):
        """地图坐标到屏幕坐标的转换比例"""
        return SCREEN_SIZE[0] / FULL_SIZE

    def update(self):
        # 应用速度限制
        self.speed = np.clip(self.speed, self.config['max_reverse'], self.config['max_forward'])

        # 计算位移
        rad = np.deg2rad(self.heading)
        delta_x = self.speed * np.cos(rad)
        delta_y = -self.speed * np.sin(rad)  # Pygame坐标系Y轴向下
        self.map_pos += pygame.Vector2(delta_x, delta_y)

        # 地图边界约束
        self.map_pos.x = max(0.0, min(self.map_pos.x, FULL_SIZE))
        self.map_pos.y = max(0.0, min(self.map_pos.y, FULL_SIZE))

        # 更新碰撞检测点
        self._update_collision_points()

    def _update_collision_points(self):
        """更新碰撞检测点坐标"""
        angle = np.deg2rad(-self.heading + 90)  # 转换到Pygame旋转方向
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        self.collision_global = []
        for x, y in self.config['collision_points']:
            # 应用旋转矩阵
            rot_x = x * cos_a - y * sin_a
            rot_y = x * sin_a + y * cos_a
            # 转换到全局坐标
            self.collision_global.append(pygame.Vector2(self.map_pos.x + rot_x, self.map_pos.y + rot_y))

    def draw(self, screen):
        # 绘制船体
        self._draw_ship(screen)
        # 绘制碰撞箱
        self._draw_collision(screen)
        # 新增碰撞点绘制
        if self.collision_screen_points:
            for p in self.collision_screen_points:
                pygame.draw.circle(screen, (0, 255, 0), p, 3)

    def _draw_ship(self, screen):
        """绘制船体三角形"""
        screen_points = []
        for (x, y) in self.config['ship_shape']:
            # 计算旋转后的坐标
            rot_x, rot_y = self._rotate_point(x, y)
            # 转换到屏幕坐标
            screen_x = (self.map_pos.x + rot_x) * self.screen_pos
            screen_y = (self.map_pos.y + rot_y) * self.screen_pos
            screen_points.append((screen_x, screen_y))

        pygame.draw.polygon(screen, COLORS['ship'], screen_points)

    def _draw_collision(self, screen):
        """绘制碰撞检测框"""
        color = self.config['collision_alert'] if self.is_colliding else self.config['collision_color']

        screen_points = [(p.x * self.screen_pos, p.y * self.screen_pos) for p in self.collision_global]

        # 绘制多边形轮廓
        pygame.draw.polygon(screen, color, screen_points, self.config['collision_width'])

        # 绘制碰撞点标记
        for p in screen_points:
            pygame.draw.circle(screen, (255, 255, 255), (int(p[0]), int(p[1])), 1)

    def _rotate_point(self, x, y):
        """坐标旋转辅助函数"""
        angle = np.deg2rad(-self.heading + 90)
        return x * np.cos(angle) - y * np.sin(angle), x * np.sin(angle) + y * np.cos(angle)


class NavigationSimulator:
    def __init__(self):
        # 初始化Pygame
        pygame.init()
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption("船舶航行模拟器")

        # 初始化组件
        self.clock = pygame.time.Clock()
        self.island_map = IslandMap()
        self.ship = Ship()
        self.running = True

    def handle_events(self):
        keys = pygame.key.get_pressed()

        # 事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.ship.speed = 0  # 急停

        # 转向控制
        self.ship.heading += (keys[pygame.K_d] - keys[pygame.K_a]) * self.ship.config['turn_rate']
        self.ship.heading %= 360  # 标准化角度

        # 速度控制
        if keys[pygame.K_w]:
            self.ship.speed += self.ship.config['acceleration']
        if keys[pygame.K_s]:
            self.ship.speed -= self.ship.config['deceleration']

    def check_collision(self):
        """精确多点碰撞检测"""
        self.ship.is_colliding = False
        self.ship.collision_screen_points = []

        for point in self.ship.collision_global:
            # 使用地板除确保坐标不越界
            raw_x = int(point.x / TILE_SCALE + 0.5)
            raw_y = int(point.y / TILE_SCALE + 0.5)

            # 添加边界保护
            raw_x = np.clip(raw_x, 0, ORIGINAL_SIZE - 1)
            raw_y = np.clip(raw_y, 0, ORIGINAL_SIZE - 1)

            if self.island_map.raw_map[raw_y, raw_x]:
                self.ship.is_colliding = True
                screen_x = int(point.x * self.ship.screen_pos + 0.5)
                screen_y = int(point.y * self.ship.screen_pos + 0.5)
                self.ship.collision_screen_points.append((screen_x, screen_y))

    def run(self):
        """主循环"""
        while self.running:
            # 处理输入
            self.handle_events()

            # 更新状态
            self.ship.update()
            self.check_collision()

            # 渲染画面
            self.screen.blit(self.island_map.display_texture, (0, 0))
            self.ship.draw(self.screen)
            pygame.display.flip()

            # 维持帧率
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    simulator = NavigationSimulator()
    simulator.run()
