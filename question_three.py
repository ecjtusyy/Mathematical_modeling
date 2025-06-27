import numpy as np

# 1) 构造矩阵 A（m=n=0）
A = np.zeros((6,6))
# 上三角手动填
A[0] = [1, 2, 4, 5, 2, 5]
A[1,1:] = [1, 2, 3, 4, 5]
A[2,2:] = [1, 2, 3, 2]
A[3,3:] = [1, 0.5, 1]
A[4,4:] = [1, 2]
A[5,5]  = 1
# 下三角由互反性 a[j,i]=1/a[i,j]
for i in range(6):
    for j in range(i):
        A[i,j] = 1/A[j,i]

# 2) 求特征值/向量
vals, vecs = np.linalg.eig(A)

# 3) 找到最大特征值及其对应的向量
idx = np.argmax(vals.real)
λ_max = vals.real[idx]
v = np.abs(vecs[:,idx].real)      # 取正
r = v / v.sum()                   # 归一化和为1

print("λ_max =", λ_max)
print("归一化特征向量 r =", r)
