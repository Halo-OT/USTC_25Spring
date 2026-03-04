import pandas as pd
df = pd.read_csv('data.csv', index_col=0)
print(df.head(10))
print(f"数据集有{df.shape[0]}行和{df.shape[1]}列") # 索引不再当作列数， 表头不计入行数

#输出各列缺失值的个数， 

#删除数据集的最后一列。
df = df.iloc[:, :-1] # 删除最后一列
# 基于更新后的数据， 展示哪一列的缺失值最多， 哪些列没有缺失值。
df.isnull().sum() # 统计每一列缺失值的个数
df.isnull().sum().idxmax() # 找到缺失值最多的列
