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

# 定义均方误差损失函数
def compute_loss(w, b, x, y):
    predictions = w * x + b
    return np.mean((predictions - y) ** 2)

# 网格搜索范围
w_values = np.linspace(0, 20, 100)
b_values = np.linspace(-50, 50, 100)

# 存储损失值
losses = np.zeros((len(w_values), len(b_values)))

# 计算损失
for i, w in enumerate(w_values):
    for j, b in enumerate(b_values):
        losses[i, j] = compute_loss(w, b, data, target)

# 绘制热力图
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.figure(figsize=(10, 8))
plt.imshow(losses, extent=(b_values.min(), b_values.max(), w_values.min(), w_values.max()),
           aspect='auto', origin='lower', cmap='hot')
plt.colorbar(label='Loss')
plt.xlabel('Bias (b)')
plt.ylabel('Weight (w)')
plt.title('Loss Distribution Heatmap')
plt.scatter([-47], [11], color='red', label='最优解 (y = 11x - 47)')
plt.legend()
plt.show()
# 预测函数
def best_prediction_function(x):
    return 11 * x - 47

# 生成预测值
predicted = best_prediction_function(data)

# 绘制预测曲线
plt.figure(figsize=(10, 6))
plt.scatter(data, target, color='blue', label='实际房价', alpha=0.5)
plt.plot(data, predicted, color='red', label='最佳预测函数', linewidth=2)

# 显示函数公式在图上
plt.text(1, 50, 'y = 11x - 47', fontsize=14, color='red', bbox=dict(facecolor='white', alpha=0.5))

plt.xlabel('房间数量')
plt.ylabel('房价中位数（千美元）')
plt.title('最佳房价预测函数')
plt.legend()
plt.grid()
plt.xlim(left=0)  # 确保 x 轴从 0 开始
plt.ylim(bottom=0)  # 确保 y 轴从 0 开始
plt.show()
