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

# 反比例回归模型的函数
def inverse_model(x, a, b):
    """
    反比例模型
    :param x: 输入特征（房间数量）
    :param a: 模型参数
    :param b: 模型参数
    :return: 预测值
    """
    return a / x + b

# 参数选择
a = -50  # 根据数据调整
b = 40   # 根据数据调整

# 生成预测值
predicted = inverse_model(data, a, b)

# 绘制散点图和预测曲线
plt.scatter(data, target, color='blue', label='实际房价', alpha=0.5)
plt.plot(data, predicted, color='red', label='反比例预测模型', linewidth=2)

# 显示函数公式在左上角

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.scatter(data, target, color='blue', label='实际房价', alpha=0.5)
plt.plot(data, predicted, color='red', label='反比例预测模型', linewidth=2)
plt.xlabel('房间数量')
plt.ylabel('房价中位数（千美元）')
plt.title('反比例回归模型预测房价')
plt.legend()
plt.grid()
plt.show()
