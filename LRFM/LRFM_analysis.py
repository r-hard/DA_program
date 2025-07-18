# 导入相关数据库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import warnings
import io
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 检查文件是否存在
file_path = os.path.join('LRFM', 'data.csv')  # 使用os.path.join来避免转义字符问题
if not os.path.exists(file_path):
    print(f"错误: 找不到文件 '{file_path}'")
    print("请确保data.csv文件位于LRFM目录中")
    exit(1)

# 读取所需数据
print("正在加载数据...")
try:
    with open(file_path, 'rb') as f:
        content = f.read()
    # 尝试直接解析日期列
    df = pd.read_csv(
        io.StringIO(content.decode('utf-8', errors='replace')),
        parse_dates=['InvoiceDate']
    )
except Exception as e:
    print(f"读取文件时出错: {str(e)}")
    try:
        # 备选方案：不解析日期，后续手动转换
        df = pd.read_csv(io.StringIO(content.decode('utf-8', errors='replace')))
    except Exception as e:
        print(f"备选方案也失败: {str(e)}")
        exit(1)

print("数据加载完成！")
print("\n原始数据前5行：")
print(df.head())

# 数据预处理
print("\n开始数据预处理...")
# 查看数据信息
print("\n数据基本信息：")
print(df.info())

# 删除缺失值
df_clean = df.dropna(subset=['CustomerID'])
print(f"\n删除缺失CustomerID后的数据量: {df_clean.shape[0]}")

# 移除取消的订单（Quantity为负的记录）
df_clean = df_clean[df_clean['Quantity'] > 0]
print(f"移除取消订单后的数据量: {df_clean.shape[0]}")

# 检查异常值
print("\n价格范围:", df_clean['UnitPrice'].min(), "到", df_clean['UnitPrice'].max())
print("数量范围:", df_clean['Quantity'].min(), "到", df_clean['Quantity'].max())

# 将InvoiceDate转换为datetime类型
print("转换日期格式...")
try:
    # 尝试标准转换
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    print("日期转换成功！")
except Exception as e:
    print(f"标准日期转换失败: {str(e)}")
    # 尝试使用多种常见格式
    formats = ['%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M', '%m/%d/%Y %H:%M', '%Y/%m/%d %H:%M']
    for fmt in formats:
        try:
            df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'], format=fmt)
            print(f"使用格式 {fmt} 转换成功")
            break
        except:
            continue
    else:
        print("所有日期格式转换都失败了")
        exit(1)

# 打印转换后的类型和样例，确认转换成功
print("转换后InvoiceDate的数据类型:", df_clean['InvoiceDate'].dtype)
print("InvoiceDate样例:\n", df_clean['InvoiceDate'].head())

# 计算LRFM指标
print("\n计算LRFM指标...")

# 找到数据集中的最近日期
max_date = df_clean['InvoiceDate'].max() + timedelta(days=1)
print(f"数据集中最近日期: {max_date.date()}")

# 将CustomerID转换为字符串类型
df_clean['CustomerID'] = df_clean['CustomerID'].astype(str)

# 计算L (Loyalty) - 客户首次购买距今的天数
df_clean['FirstPurchaseDate'] = df_clean.groupby('CustomerID')['InvoiceDate'].transform('min')
df_clean['Loyalty'] = (max_date - df_clean['FirstPurchaseDate']).dt.days

# 计算R (Recency) - 最近一次购买距今的天数
df_clean['LastPurchaseDate'] = df_clean.groupby('CustomerID')['InvoiceDate'].transform('max')
df_clean['Recency'] = (max_date - df_clean['LastPurchaseDate']).dt.days

# 计算F (Frequency) - 购买频率
df_clean['Frequency'] = df_clean.groupby('CustomerID')['InvoiceNo'].transform('nunique')

# 计算M (Monetary) - 消费金额
df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']
df_clean['Monetary'] = df_clean.groupby('CustomerID')['TotalAmount'].transform('sum')

# 创建LRFM汇总表
lrfm_df = df_clean[['CustomerID', 'Loyalty', 'Recency', 'Frequency', 'Monetary']].drop_duplicates()
print("\nLRFM指标计算完成！LRFM汇总表前5行：")
print(lrfm_df.head())

# LRFM分数计算
print("\n计算LRFM分数...")
# 将LRFM指标分为5个等级
lrfm_df['L_Score'] = pd.qcut(lrfm_df['Loyalty'], q=5, labels=[5, 4, 3, 2, 1])
lrfm_df['R_Score'] = pd.qcut(lrfm_df['Recency'], q=5, labels=[1, 2, 3, 4, 5])
lrfm_df['F_Score'] = pd.qcut(lrfm_df['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
lrfm_df['M_Score'] = pd.qcut(lrfm_df['Monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])

# 计算总分
lrfm_df['LRFM_Score'] = lrfm_df['L_Score'].astype(str) + lrfm_df['R_Score'].astype(str) + \
                        lrfm_df['F_Score'].astype(str) + lrfm_df['M_Score'].astype(str)

# 客户分类
def customer_segment(row):
    # 将L, R, F, M分数转换为整数
    l = int(row['L_Score'])
    r = int(row['R_Score'])
    f = int(row['F_Score'])
    m = int(row['M_Score'])
    
    # 高价值客户
    if r >= 4 and f >= 4 and m >= 4:
        return '高价值客户'
    # 忠诚客户
    elif l >= 4 and r >= 3 and f >= 3:
        return '忠诚客户'
    # 潜力客户
    elif r >= 3 and (f >= 3 or m >= 3):
        return '潜力客户'
    # 新客户
    elif l <= 2 and r >= 4:
        return '新客户'
    # 流失风险客户
    elif l >= 4 and r <= 2:
        return '流失风险客户'
    # 流失客户
    elif r <= 2:
        return '已流失客户'
    # 其他
    else:
        return '一般客户'

lrfm_df['客户类型'] = lrfm_df.apply(customer_segment, axis=1)
print("\n客户分类完成！")

# 可视化客户分布
print("\n生成客户分布可视化...")
plt.figure(figsize=(12, 6))
segment_counts = lrfm_df['客户类型'].value_counts()
sns.barplot(x=segment_counts.index, y=segment_counts.values)
plt.title('客户类型分布')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('客户类型分布.png')

# 计算每个客户类型的平均LRFM指标
segment_avg = lrfm_df.groupby('客户类型')[['Loyalty', 'Recency', 'Frequency', 'Monetary']].mean()
print("\n各客户类型的平均LRFM指标：")
print(segment_avg)

# 增加输出聚类结果统计
print("\n客户类型统计:")
type_stats = lrfm_df['客户类型'].value_counts()
for customer_type, count in type_stats.items():
    percentage = (count / len(lrfm_df)) * 100
    print(f"{customer_type}: {count}人 ({percentage:.2f}%)")

# 增加更详细的客户类型特征描述
print("\n客户类型特征描述:")
for customer_type in lrfm_df['客户类型'].unique():
    subset = lrfm_df[lrfm_df['客户类型'] == customer_type]
    print(f"\n{customer_type} (共{len(subset)}人):")
    print(f"  平均忠诚度: {subset['Loyalty'].mean():.2f}天")
    print(f"  平均近期购买: {subset['Recency'].mean():.2f}天前")
    print(f"  平均购买频率: {subset['Frequency'].mean():.2f}次")
    print(f"  平均消费金额: {subset['Monetary'].mean():.2f}")

# 保存结果
print("\n保存分析结果...")
lrfm_df.to_excel('LRFM分析结果.xlsx', index=False)
segment_avg.to_excel('客户类型分析.xlsx')

print("\nLRFM分析完成！结果已保存为'LRFM分析结果.xlsx'和'客户类型分析.xlsx'")
print("客户类型分布图已保存为'客户类型分布.png'")
