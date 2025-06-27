import numpy as np
from scipy.optimize import linprog

# --- 1. 定义数据和参数 ---

# 假设 m 和 n 的值 (请根据您的实际情况修改)
m = 3
n = 7 # 假设学号后两位是25, 2+5=7

# 成本矩阵 C (7个产地, 9个销地)
cost = np.array([
    [6, 2, 6, 7, 4, 2, 5, 10, 7],  # A1
    [4, 8, 5, 4, 8, 5, 7, 3, 3],  # A2
    [5, 4, 2, 9, 7, 4, 3, 3, 6],  # A3
    [8, 6, 7, 3, 9, 3, 9, 2, 3],  # A4
    [4, 4, 8, 5, 7, 3, 6, 5, 5],  # A5
    [5, 5, 2, 2, 9, 2, 4, 2, 3],  # A6
    [5, 4, 7, 5, 6, 5, 7, 5, 4]   # A7
])

# 供应量/产量 S
supply = np.array([
    60 + m,
    56 - n,
    51,
    44,
    41,
    52,
    51
])

# 需求量/销量 D
demand = np.array([
    36, 37, 22, 32, 41, 32, 43, 38, 36
])

# 获取维度信息
num_sources, num_dests = cost.shape
num_vars = num_sources * num_dests

# --- 2. 构建线性规划模型 ---

# 目标函数：最小化总成本。linprog需要一个一维数组
c = cost.flatten()

# 等式约束 (需求约束): 每列的和必须等于需求量
# A_eq 矩阵的形状是 (销地数量, 总变量数)
A_eq = np.zeros((num_dests, num_vars))
for j in range(num_dests):
    for i in range(num_sources):
        A_eq[j, i * num_dests + j] = 1
b_eq = demand

# 不等式约束 (供应约束): 每行的和必须小于等于供应量
# A_ub 矩阵的形状是 (产地数量, 总变量数)
A_ub = np.zeros((num_sources, num_vars))
for i in range(num_sources):
    for j in range(num_dests):
        A_ub[i, i * num_dests + j] = 1
b_ub = supply

# 变量边界约束
# 一般来说，运量 x_ij >= 0，所以边界是 (0, None)
bounds = [(0, None) for _ in range(num_vars)]

# 应用特殊约束：将特定路径的运量边界设为(0, 0)，即强制为0
# B1 (j=0) 只能从 A1,A2,A3 (i=0,1,2) 进货
for i in [3, 4, 5, 6]: # A4, A5, A6, A7
    bounds[i * num_dests + 0] = (0, 0) # 对应 x_i1

# B2 (j=1) 只能从 A4,A5,A6 (i=3,4,5) 进货
for i in [0, 1, 2, 6]: # A1, A2, A3, A7
    bounds[i * num_dests + 1] = (0, 0) # 对应 x_i2


# --- 3. 调用求解器并输出结果 ---
print("正在求解线性规划问题...")
# 使用 'highs' 方法，它是 scipy 中较新且非常高效的求解器
result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

# 检查求解是否成功
if result.success:
    print("\n求解成功！")
    print(f"最小运输费用为: {result.fun:.2f}")
    
    # 将一维的结果变量重新整理为 7x9 的矩阵以便查看
    shipping_plan = result.x.reshape((num_sources, num_dests))
    
    print("\n最优运输方案 (从产地 A_i 到销地 B_j 的运量):")
    # 打印表头
    print("      ", end="")
    for j in range(num_dests):
        print(f"   B{j+1}  ", end="")
    print("\n" + "-" * (8 * (num_dests + 1)))

    # 打印每一行的数据
    for i in range(num_sources):
        print(f"A{i+1} |", end="")
        for j in range(num_dests):
            # 对运量取整并格式化输出，便于阅读
            print(f" {shipping_plan[i, j]:>6.1f} ", end="")
        print()

else:
    print("\n求解失败。")
    print(f"失败信息: {result.message}")