import numpy as np
import matplotlib.pyplot as plt

# 参数设置
L = 2.0  # 方形板边长
n = 2    # x方向模数（自由边界）
m = 1    # y方向模数

# 生成坐标网格
x = np.linspace(-L/2, L/2, 200)
y = np.linspace(-L/2, L/2, 200)
X, Y = np.meshgrid(x, y)

# 计算驻波模式（自由边界，使用余弦函数）
X_mode = np.cos(n * np.pi * X / L)
Y_mode = np.cos(m * np.pi * Y / L)
Z = X_mode * Y_mode

# 绘制二维等高线图（节点线为振幅零点）
plt.figure(figsize=(8, 6))
contour = plt.contour(X, Y, Z, levels=[0], colors='k')  # 节点线
plt.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
plt.colorbar(label='振幅')
plt.title(f'自由边界驻波模式 (n={n}, m={m})\n节点线为黑色轮廓')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.show()