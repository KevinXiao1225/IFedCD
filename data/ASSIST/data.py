import pandas as pd


# 读取ASSIST2009数据集
df = pd.read_csv('ASSIST2009.csv')

# 提取user id和problem id作为uid和iid
dataset = df[['user id', 'problem id']].copy()

# 重命名列
dataset.columns = ['uid', 'iid']

# 添加index列，从1开始
dataset.insert(0, 'index', range(1, len(dataset) + 1))

# 保存为新的CSV文件，包含列名
dataset.to_csv('dataset.csv', index=False)

