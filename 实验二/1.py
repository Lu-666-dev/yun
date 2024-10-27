import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 数据集 URL
data_url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'

# 加载数据
raw_df = pd.read_csv(data_url)

# 打印原始数据的形状
print(f'Raw data shape: {raw_df.shape}')

# 选择特征和目标
data = raw_df.iloc[:, :-1].values  # 选择前 13 列作为特征
target = raw_df.iloc[:, -1].values  # 选择最后一列作为目标

# 打印选择后的数据和目标的形状
print(f'Selected data shape: {data.shape}')
print(f'Target shape: {target.shape}')

# 更新特征名称
fts_names = [
    '犯罪率（%）',
    '大住宅用地占比（%）',
    '非零售商业用地占比（%）',
    '景观房',
    '氮氧化物浓度（ppm）',
    '平均房间数',
    '老旧房屋占比（%）',
    '离就业中心的加权距离',
    '辐射路可达性指标',
    '每万元房产税',
    '学生-教师比',
    '低层次人口占比（%）',
    '附加变量'  # 新增的特征名称
]


# 绘图设置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 支持负号


num_fts = data.shape[1] # Get the number of features
num_col = 6 # Number of columns in the figure
num_row = int(np.ceil(num_fts / num_col)) # Number of rows in the figure

label_size = 18 # Label size
ticklabel_size = 14 # Tick label size

_, axes = plt.subplots(num_row, num_col, figsize=(18, 3*num_row)) # Create a figure

for i in range(num_fts): # Loop through all features
    row = int(i / num_col) # Get the row index
    col = i % num_col # Get the column index

    ax = axes[row, col]
    ax.scatter(data[:, i], target) # Plot scatter fig of i-th feature and target
    ax.tick_params(axis='both', which='major', labelsize=ticklabel_size) # Set tick label size
    ax.set_xlabel(fts_names[i], fontsize=label_size) # Label the x-axis
    ax.set_ylabel('房价中位数（千美元）', fontsize=label_size) # Label the y-axis

plt.tight_layout() # Adjust the layout of the figure
plt.show() # Display the figure
