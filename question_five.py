# 导入所需的库
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 为了在图表中正确显示中文，进行字体设置
# 如果您的系统中没有'SimHei'字体，可以换成'Microsoft YaHei'或其他支持中文的字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
except Exception as e:
    print(f"中文字体设置失败，将使用默认字体。错误: {e}")


# --- 1. 定义原始数据 ---
# 将题目中的x和y数据存入Numpy数组
x_data = np.array([1, 3, 5, 6, 8, 9])
y_data = np.array([3.9, 24.5, 59.1, 83.0, 142.8, 178.6])


# --- 2. 求解问题 (1) ---
print("===== 问题 (1): 拟合 y = ax^2 + b =====")

# 定义问题(1)的模型函数
# 这个函数告诉curve_fit我们的方程是什么样的
def model_func_1(x, a, b):
    return a * x**2 + b

# 使用 curve_fit 进行曲线拟合
# 它会返回最优的参数值(popt)和参数的协方差矩阵(pcov)
popt1, pcov1 = curve_fit(model_func_1, x_data, y_data)

# 提取并打印结果
a1, b1 = popt1
print(f"通过最小二乘法拟合得到的参数为:")
print(f"a = {a1:.4f}")
print(f"b = {b1:.4f}")
print(f"因此，拟合得到的方程为: y = {a1:.4f}x^2 + {b1:.4f}")


# --- 3. 求解问题 (2) ---
print("\n===== 问题 (2): 拟合 y = ax^2 + bx + c =====")

# 定义问题(2)的模型函数
def model_func_2(x, a, b, c):
    return a * x**2 + b * x + c

# 使用 curve_fit 进行曲线拟合
popt2, pcov2 = curve_fit(model_func_2, x_data, y_data)

# 提取并打印结果
a2, b2, c2 = popt2
print(f"通过最小二乘法拟合得到的参数为:")
print(f"a = {a2:.4f}")
print(f"b = {b2:.4f}")
print(f"c = {c2:.4f}")
print(f"因此，拟合得到的方程为: y = {a2:.4f}x^2 + {b2:.4f}x + {c2:.4f}")


# --- 4. 结果可视化 ---
print("\n正在生成拟合曲线图...")

# 创建一个平滑的x轴用于绘制拟合曲线
x_smooth = np.linspace(x_data.min(), x_data.max(), 200)

# 计算两个模型在平滑x轴上的y值
y_fit1 = model_func_1(x_smooth, a1, b1)
y_fit2 = model_func_2(x_smooth, a2, b2, c2)

# 开始绘图
plt.figure(figsize=(12, 7)) # 创建图窗

# 绘制原始数据散点图
plt.scatter(x_data, y_data, color='red', s=50, zorder=5, label='原始数据点')

# 绘制模型(1)的拟合曲线
plt.plot(x_smooth, y_fit1, 'g--', linewidth=2, label=f'模型(1) y = {a1:.2f}x² + {b1:.2f}')

# 绘制模型(2)的拟合曲线
plt.plot(x_smooth, y_fit2, 'b-', linewidth=2, label=f'模型(2) y = {a2:.2f}x² + {b2:.2f}x + {c2:.2f}')

# 添加图表元素，使其更清晰
plt.title('数据曲线拟合结果对比', fontsize=16)
plt.xlabel('x 值', fontsize=12)
plt.ylabel('y 值', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()