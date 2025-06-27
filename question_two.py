import numpy as np
from scipy.optimize import linprog

def solve_transportation(cost_matrix, supply, demand, title=""):
    c = cost_matrix.flatten()

    # 供给约束：每行 <= 对应产地供应量
    A_supply = np.zeros((7, 63))
    for i in range(7):
        A_supply[i, i * 9:(i + 1) * 9] = 1

    # 需求约束：每列 == 对应销售地需求量
    A_demand = np.zeros((9, 63))
    for j in range(9):
        A_demand[j, j::9] = 1

    A_ub = A_supply
    b_ub = supply
    A_eq = A_demand
    b_eq = demand

    bounds = [(0, None)] * 63

    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')

    print(f"\n===== {title} =====")
    if res.success:
        print(f"最小运输成本为: {res.fun:.2f}")
        x_opt = res.x.reshape((7, 9))
        for i in range(7):
            for j in range(9):
                if x_opt[i, j] > 1e-5:  # 忽略太小的数值
                    print(f"从A{i+1}到B{j+1} 运量: {x_opt[i,j]:.2f}")
    else:
        print("求解失败：", res.message)

# --------------------------
# 输入部分
# --------------------------

# 成本矩阵（原始）
base_cost = np.array([
    [6, 2, 3, 7, 4, 2, 5, 10, 2],
    [4, 2, 8, 5, 4, 3, 9, 4, 3],
    [5, 4, 5, 3, 4, 3, 6, 5, 4],
    [4, 3, 6, 2, 3, 4, 4, 3, 3],
    [8, 6, 4, 6, 7, 3, 6, 5, 4],
    [4, 4, 4, 5, 4, 5, 6, 3, 4],
    [5, 4, 5, 3, 4, 5, 5, 6, 3]
])

# 参数 m = 5（3班），n = 2（学号后两位和）
supply = np.array([60+5, 56+2, 51, 44, 41, 52, 51])
demand = np.array([36, 41, 37, 22, 32, 41, 43, 32, 38])

# 问题一：无额外约束
solve_transportation(base_cost.copy(), supply, demand, title="问题一：正常运输问题")

# 问题二：加特殊限制
cost2 = base_cost.copy()

# B1 只能来自 A1, A2, A3
for i in range(3, 7):
    cost2[i][0] = 1e6

# B3 只能来自 A4, A5, A6
cost2[0][2] = cost2[1][2] = cost2[6][2] = 1e6

solve_transportation(cost2, supply, demand, title="问题二：带运输限制的运输问题")
