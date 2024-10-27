import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 数据集 URL
data_url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'

# 加载数据
raw_df = pd.read_csv(data_url)

# 打印数据的列名
print(raw_df.columns)

# 选择特征和目标（这里以房间数量RM作为特征）
data = raw_df['rm'].values
target = raw_df['medv'].values

# 定义线性模型
def linear_model_1(x):
    return 0.5 * x + 20

def linear_model_2(x):
    return 0.8 * x + 15

def linear_model_3(x):
    return 1.0 * x + 10

# 创建散点图
plt.scatter(data, target, color='blue', label='实际房价')

# 生成预测值
x_range = np.linspace(data.min(), data.max(), 100)
plt.plot(x_range, linear_model_1(x_range), color='red', label='模型1: y=0.5x+20')
plt.plot(x_range, linear_model_2(x_range), color='green', label='模型2: y=0.8x+15')
plt.plot(x_range, linear_model_3(x_range), color='orange', label='模型3: y=1.0x+10')

# 图形设置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.xlabel('房间数量 (RM)', fontsize=14)
plt.ylabel('房价中位数 (MEDV)', fontsize=14)
plt.title('房价预测模型对比', fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()

# 显示图形
plt.show()
