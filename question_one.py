import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# 目标函数系数（最小化 -R）
c = [-3, -5, -7, 0, 0, 0]  # 前3个是 x1,x2,x3，后3个是 y1,y2,y3

# 不等式约束 A @ x <= b
A = [
    [2, 3, 5, 0, 0, 0],           # 2x1 + 3x2 + 5x3 <= 600
    [120, 100, 180, 0, 0, 0],     # 120x1 + 100x2 + 180x3 <= 40000
    [1, 0, 0, -1000, 0, 0],       # x1 - 1000*y1 <= 0
    [0, 1, 0, 0, -1000, 0],       # x2 - 1000*y2 <= 0
    [0, 0, 1, 0, 0, -1000],       # x3 - 1000*y3 <= 0
    [-1, 0, 0, 80, 0, 0],         # -x1 + 80*y1 <= 0  (即 x1 >= 80*y1)
    [0, -1, 0, 0, 100, 0],        # -x2 + 100*y2 <= 0
    [0, 0, -1, 0, 0, 60],         # -x3 + 60*y3 <= 0
]
b = [600, 40000, 0, 0, 0, 0, 0, 0]

linear_constraint = LinearConstraint(A, -np.inf, b)

# 决策变量类型：前3个是连续变量，后3个是整数（0或1）
integrality = [0, 0, 0, 1, 1, 1]

# 变量上下界（默认 x ≥ 0）
bounds = Bounds([0]*6, [np.inf]*3 + [1]*3)

# 求解 MILP
res = milp(c=c, constraints=[linear_constraint], integrality=integrality, bounds=bounds)

# 输出结果
if res.success:
    x1, x2, x3, y1, y2, y3 = res.x
    print("最大值 R =", -res.fun)  # 注意要变号还原
    print(f"x1 = {x1:.2f}, x2 = {x2:.2f}, x3 = {x3:.2f}")
    print(f"y1 = {round(y1)}, y2 = {round(y2)}, y3 = {round(y3)}")
else:
    print("求解失败:", res.message)
