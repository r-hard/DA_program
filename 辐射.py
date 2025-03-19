import pandas as pd

data = pd.read_excel('J00371-历史综合数据.xlsx')

# 查看相关列的数据的相关系数
correlation = data[['实际功率', 'NWP-晴空总辐射', 'NWP-法相直辐射']].corr()
print("相关系数：",correlation)

# 可视化分析
import seaborn as sns
import matplotlib.pyplot as plt

# 解决字体显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 散点图
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(data=data, x='NWP-晴空总辐射', y='实际功率')
plt.title('实际功率 vs NWP-晴空总辐射')

plt.subplot(1, 2, 2)
sns.scatterplot(data=data, x='NWP-法相直辐射', y='实际功率')
plt.title('实际功率 vs NWP-法相直辐射')

plt.tight_layout()
plt.show()