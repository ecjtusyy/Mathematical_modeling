import numpy as np
import matplotlib.pyplot as plt

def solve_sir_model(h, title=""):
    """
    使用欧拉法求解SIR模型，并输出结果和绘图。

    参数:
    h (float): 计算时使用的步长 (time step)。
    title (str): 结果和图表的标题。
    """
    # 1) 定义模型参数
    n = 2  # 班级参数
    m = 20 # 学号参数
    
    lambda_val = 1 - 0.1 * n
    mu_val = 0.1 + 0.005 * m

    # 初始条件
    i0 = 0.01
    s0 = 0.98
    
    # 时间设置
    t_start = 0
    t_end = 50
    
    # 2) 初始化数据存储
    # 计算总步数
    num_steps = int((t_end - t_start) / h)
    
    # 创建时间点数组和结果数组
    t_points = np.linspace(t_start, t_end, num_steps + 1)
    i_results = np.zeros(num_steps + 1)
    s_results = np.zeros(num_steps + 1)
    
    # 设置初始值
    i_results[0] = i0
    s_results[0] = s0

    # 3) 欧拉法数值求解
    # 迭代计算每一步的 i 和 s
    for k in range(num_steps):
        s_current = s_results[k]
        i_current = i_results[k]
        
        # SIR 微分方程
        ds_dt = -lambda_val * s_current * i_current
        di_dt = lambda_val * s_current * i_current - mu_val * i_current
        
        # 欧拉法更新下一步的值
        s_next = s_current + h * ds_dt
        i_next = i_current + h * di_dt
        
        s_results[k+1] = s_next
        i_results[k+1] = i_next

    # 4) 输出指定时间点的结果
    print(f"\n===== {title} (步长 h = {h}) =====")
    print(f"参数: λ = {lambda_val:.2f}, μ = {mu_val:.2f}")
    print("-" * 35)
    print(" t       i(t)         s(t)")
    print("-" * 35)
    
    for t_target in [10, 20, 30, 40, 50]:
        # 找到最接近目标时间的索引
        idx = int(t_target / h)
        print(f"{t_target:<5}    {i_results[idx]:<10.6f}   {s_results[idx]:<10.6f}")
    print("-" * 35)

    # 5) 绘制图像
    # 创建一个包含3个子图的图窗
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 15))
    fig.suptitle(f'SIR Model Simulation: {title} (h={h})', fontsize=16)

    # 子图1: i(t) vs t
    ax1.plot(t_points, i_results, 'r-', label='i(t) - Infected')
    ax1.set_title('Infected Population over Time')
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Proportion of Population')
    ax1.grid(True)
    ax1.legend()
    
    # 子图2: s(t) vs t
    ax2.plot(t_points, s_results, 'b-', label='s(t) - Susceptible')
    ax2.set_title('Susceptible Population over Time')
    ax2.set_xlabel('Time (t)')
    ax2.set_ylabel('Proportion of Population')
    ax2.grid(True)
    ax2.legend()

    # 子图3: i vs s 相轨图
    ax3.plot(s_results, i_results, 'g-')
    ax3.set_title('Phase Plot: i vs s')
    ax3.set_xlabel('Susceptible (s)')
    ax3.set_ylabel('Infected (i)')
    ax3.grid(True)

    # 调整子图间距并显示
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# --------------------------
# 主程序
# --------------------------
# 执行步长为 1 的计算
solve_sir_model(h=1.0, title="步长为1的计算结果")

# 执行步长为 0.1 的计算
solve_sir_model(h=0.1, title="步长为0.1的计算结果")