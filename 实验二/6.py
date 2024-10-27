import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 数据集 URL
data_url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'

# 加载数据
raw_df = pd.read_csv(data_url)

# 选择特征 'RM' 和目标 'MEDV'
data = raw_df['rm'].values  # 房间数量
target = raw_df['medv'].values  # 房价中位数

# 标准化特征
data_mean = np.mean(data)
data_std = np.std(data)
data_normalized = (data - data_mean) / data_std

# 动态学习率设置
initial_lr = 0.0001  # 初始学习率
b = -200.0  # 偏差
w = 0.0  # 权重
iterations = 10000  # 迭代次数

# 存储损失值
losses = []

# 动态学习率的梯度下降算法
for i in range(iterations):
    # 计算预测值
    predictions = w * data_normalized + b
    # 计算损失（均方误差）
    loss = np.mean((predictions - target) ** 2)
    losses.append(loss)

    # 计算梯度
    gradient_w = (2 / len(data)) * np.dot(data_normalized, (predictions - target))
    gradient_b = (2 / len(data)) * np.sum(predictions - target)

    # 更新权重和偏差
    w -= initial_lr * gradient_w
    b -= initial_lr * gradient_b

    # 每1000次迭代减少学习率
    if i % 1000 == 0:
        initial_lr *= 0.95  # 每1000次迭代学习率减小5%

# 绘制损失变化图
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.figure(figsize=(10, 6))
plt.plot(losses, color='blue')
plt.title('动态学习率的损失函数变化图')
plt.xlabel('迭代次数')
plt.ylabel('均方误差')
plt.grid()

# 在右上角固定位置显示最终的权重和偏差
plt.text(0.95, 0.95, f'最终权重: {w:.2f}\n最终偏差: {b:.2f}', 
         bbox=dict(facecolor='white', alpha=0.5), ha='right', va='top', 
         transform=plt.gca().transAxes)

plt.show()

# 输出最终的权重和偏差
print(f'最终权重: {w}, 最终偏差: {b}')
