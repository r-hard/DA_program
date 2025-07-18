'''
@author: ranjinsheng
@date: 2025-07-17
@goal:对网页进行A/B测试，通过A/B实验帮助公司分析和决定是否应该使用新页面，保留旧页面，或者延长旧页面测试时间

@字段含义：
user_id:用户ID
timestamp:用户访问时间
group:用户分组，control(对应旧页面)或treatment(对应新页面)
landing_page:用户访问的页面，old_page或new_page
converted:用户是否转换，1表示转换，0表示未转换

A/B测试分析思路：
1. 数据探索和清洗：检查数据质量，处理不匹配和重复数据
2. 描述性统计：计算各组转化率，观察初步差异
3. 假设检验：使用统计方法验证新页面是否显著优于旧页面
4. 回归分析：控制其他变量影响，进一步验证结果
5. 交互效应分析：探索不同国家用户对新页面的反应差异
'''

# ==================== 1. 导入必要的库 ====================
import pandas as pd      # 数据处理和分析
import numpy as np       # 数值计算
import matplotlib.pyplot as plt  # 数据可视化
import seaborn as sns    # 统计图表
import random           # 随机数生成
import statsmodels.api as sm  # 统计建模
import warnings
warnings.filterwarnings('ignore')

# 正常显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 2. 数据导入和基本探索 ====================
# 读取A/B测试数据
df = pd.read_csv('D:\\Cursor_program\\数分项目\\A-B\\ab_data.csv')

# 查看数据基本信息：数据类型、内存使用、非空值数量
print(df.info())

# 查看前5行数据，了解数据结构
print(df.head())

# 查看数据集的行列数
print(df.shape)

# ==================== 3. 数据质量检查 ====================
# 统计不同用户ID的数量（检查是否有重复用户）
print("用户数量:",df['user_id'].nunique())

# 计算总体转化率（converted列的平均值，因为是0/1数据）
print("转化用户占比:",df['converted'].mean())   # mean()方法可用于二值数据（1或者0）列的计算

# ==================== 4. 数据一致性检查 ====================
# 检查分组与页面的匹配度：treatment组应该对应new_page，control组应该对应old_page
# 使用!=操作符检查不匹配的情况，然后求和统计不匹配次数
no_sum = ((df['landing_page'] =='new_page') != (df['group'] == 'treatment')).sum()
print("treatment与new_page的不匹配次数:",no_sum)

# 检查是否存在缺失值
print("缺失值数量：\n",df.isnull().sum())

print("--------------对于treatment与new_page不匹配的行，我们不能确定是否接受了新页面还是旧页面，所以进行测试2------------------------")

# ==================== 5. 数据清洗：创建一致性数据集 ====================
# 创建一个符合逻辑的数据集：treatment组对应new_page，control组对应old_page
# 使用布尔索引筛选出匹配的行
df2 = df[(df['group'] == 'treatment') == (df['landing_page'] == 'new_page')].copy()

# 验证清洗后的数据集是否还存在不匹配的情况
df2_check = df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]
print("df2中是否存在不匹配的值:",df2_check)

# 统计清洗后数据集中的不同用户数量
print("不同用户数量：",df2.user_id.nunique())

# ==================== 6. 处理重复用户 ====================
# 查找重复的user_id（keep=False表示所有重复项都标记为True）
df2_repeat_user = df2[df2['user_id'].duplicated(keep = False)].user_id
print("重复的user_id:\n",df2_repeat_user)

# 查看重复用户的详细信息
df2_repeat_user_information = df2[df2['user_id'].duplicated(keep = False)]
print("重复的user_id信息：\n",df2_repeat_user_information)

# 删除重复的用户记录（这里选择删除索引为2893的行）
df2.drop(index = 2893, inplace = True)

print("-------------------------------------使用优化过后的df2，做接下的测试-------------------------------------------------------")

# ==================== 7. 描述性统计分析 ====================
# 计算总体转化率（不区分页面类型）
q0 = df2.converted.mean()
print("用户成功转化的概率是：",q0)

# 计算control组（旧页面）的转化率
q1 = df2.query("group == 'control'").converted.mean()
print("control组的转化率是：",q1)

# 计算treatment组（新页面）的转化率
q2 = df2.query("group == 'treatment'").converted.mean()
print("treatment组的转化率是：",q2)

# 计算两组转化率的差值（旧页面 - 新页面）
diff = q1 - q2
print("旧页面与新页面之间的差值是：",diff)

# 计算新用户收到新页面的概率（实验设计的均衡性检查）
q3_new_page = (df2['landing_page'] == 'new_page').mean()
print("一个新用户收到新页面的概率是：",q3_new_page)

print('---------------总结：不能证明旧页面和新页面能够提高转化率，因为两者之间的差值仅仅只有0.16%---------------------------------------')

'''
                         A/B测试假设检验设计
假设：
1、零假设(H0):假设旧页面（control）效果更加或者不差于新页面 (p_old >= p_new)
2、备择假设(H1):新页面(treatment)效果更好 (p_new > p_old)
3、一类错误：旧页面本来就很好，但我们却错误的认为新页面更好（一类错误概率不超过5%）
4、决策规则：只有当测试数据在统计上显著(p<0.05)时，我们才拒绝H0,接受H1,即新页面更好；否者，我们保留H0，旧页面更佳
'''

print("---------------------------------开始进行A/B测试---------------------------------")

# ==================== 8. 假设检验：模拟零假设分布 ====================
# 在零假设中，假设新旧页面的转化率相等，都等于总体转化率q0
p_new = q0  # 零假设下新页面的转化率
p_old = q0  # 零假设下旧页面的转化率

# 计算新页面和旧页面的样本量
N_new = (df2['landing_page'] == 'new_page').sum()
N_old = (df2['landing_page'] == 'old_page').sum()

# 使用二项分布模拟零假设下新页面的转化结果
# np.random.choice从[0,1]中随机选择N_new个值，概率分别为(1-p_new, p_new)
random.seed(42)  # 设置随机种子保证结果可重现
new_page_converted = np.random.choice([0,1],N_new,p=(1-p_new,p_new))

# 使用二项分布模拟零假设下旧页面的转化结果
random.seed(42)
old_page_converted = np.random.choice([0,1],N_old,p=(1-p_old,p_old))

# 计算模拟数据下两组转化率的差值
p = new_page_converted.mean() - old_page_converted.mean()
print("p_new和p_old的差值是：",p)

# ==================== 9. 计算p值：蒙特卡洛模拟 ====================
# 通过1000次模拟来估计在零假设下观测到当前差值或更极端差值的概率
p_diffs = []  # 存储每次模拟的转化率差值
for i in range(1000):
    # 模拟新页面转化率
    new_con_mean = np.random.choice([0,1],N_new,p=(1-p_new,p_new)).mean()
    # 模拟旧页面转化率
    old_con_mean = np.random.choice([0,1],N_old,p=(1-p_old,p_old)).mean()
    # 计算差值（新页面 - 旧页面）
    no_con_mean = new_con_mean - old_con_mean
    p_diffs.append(no_con_mean)

# ==================== 10. 可视化零假设分布 ====================
# 绘制1000次模拟差值的直方图
plt.hist(p_diffs)
# 在零值处添加红色虚线，表示零假设的期望值
plt.axvline(x=0, color='red', linestyle='--')
plt.title('零假设下转化率差值分布')
plt.xlabel('新页面转化率 - 旧页面转化率')
plt.ylabel('频次')
plt.show()

# ==================== 11. 计算p值 ====================
# 统计模拟结果中大于实际观测值的比例，这就是单侧检验的p值
p_count = (p_diffs > (q2 - q1)).mean()
print("p_diffs中大于实际观测值（q2 - q1）的次数是：",p_count)

# ==================== 12. 使用正态分布近似进行假设检验 ====================
# 计算各组的转化用户数量
convert_old = df2.query('group == "control"').converted.sum()
convert_new = df2.query('group == "treatment"').converted.sum()
# 获取各组的样本量
n_old = (df2['landing_page'] == "old_page").sum()
n_new = (df2['landing_page'] == "new_page").sum()

# 使用比例z检验计算z统计量和p值
# alternative='smaller'表示单侧检验：H1: p_new > p_old
z_score,p_value = sm.stats.proportions_ztest([convert_old,convert_new],[n_old,n_new],alternative = 'smaller')
print("z-score的值是：",z_score)
print("p-value的值是：",p_value)

print('---------------------------------------进行回归分析，A / B测试中获得的结果也可以通过执行回归来获取---------------------------------------------------------------')

# ==================== 13. 逻辑回归分析 ====================
# 创建截距项（逻辑回归需要常数项）
df2['intercept'] = 1
# 创建处理组的虚拟变量：treatment=1, control=0
df2['ab_page'] = df2['group'].map({'treatment':1,'control':0})

# 构建逻辑回归模型：converted ~ intercept + ab_page
log_mod = sm.Logit(df2['converted'],df2[['intercept','ab_page']])
# 拟合模型
result = log_mod.fit()
# 输出模型结果摘要
print("模型摘要：\n",result.summary())

# ==================== 14. 加入国家变量的分析 ====================
# 读取包含国家信息的数据
df3 = pd.read_csv('D:\\Cursor_program\\数分项目\\A-B\\countries.csv')
# 将国家数据与主数据集合并
df4 = df2.merge(df3,on = "user_id")
# 检查合并后是否有缺失值
print("缺失值总数：", df4.isnull().sum().sum())

# ==================== 15. 创建国家虚拟变量 ====================
# 为国家变量创建虚拟变量（one-hot编码）
country_dummies = pd.get_dummies(df4['country'], prefix='', prefix_sep='')
# 将虚拟变量添加到数据框中
df4 = pd.concat([df4, country_dummies], axis=1)

# 确保虚拟变量是整数类型（0或1）
for col in ['CA', 'UK', 'US']:
    if col in df4.columns:
        df4[col] = df4[col].astype(int)

# ==================== 16. 包含国家效应的逻辑回归 ====================
# 构建包含国家虚拟变量的逻辑回归模型
# 注意：US作为参考组被省略（避免多重共线性）
log_mod2 = sm.Logit(df4['converted'], df4[['intercept','ab_page','CA','UK']])
result2 = log_mod2.fit()
print("模型摘要：",result2.summary())

# ==================== 17. 交互效应分析 ====================
# 创建页面类型的虚拟变量：new_page=1, old_page=0
df4['new_page'] = df4['landing_page'].map({'new_page':1,'old_page':0})

# 创建交互项：新页面与国家的交互效应
df4['new_CA'] = df4['new_page'] * df4['CA']  # 新页面×加拿大
df4['new_UK'] = df4['new_page'] * df4['UK']  # 新页面×英国

# 重新创建截距项
df4['intercept'] = 1

# ==================== 18. 包含交互效应的逻辑回归 ====================
# 构建包含交互项的逻辑回归模型
# 模型包括：主效应（ab_page, CA, UK）和交互效应（new_CA, new_UK）
log_mod3 = sm.Logit(df4['converted'],df4[['intercept','ab_page','new_CA','CA','new_UK','UK']])
result3 = log_mod3.fit()
print("模型摘要：",result3.summary())

'''
==================== 分析结论 ====================
总结：根据以上概率，假设检验以及逻辑回归的结果显示，
我们不能得出新页面的转化率高于旧页面，而根据逻辑回归中添加的新变量跟交叉项的结果显示，
我们能确定各变量对转化率没有明显影响，因此需要继续延长实验时间，并且从其他变量上再次考虑影响因素。

分析思路总结：
1. 数据质量检查：确保数据的完整性和一致性
2. 描述性分析：初步了解两组的差异
3. 假设检验：使用统计方法验证差异的显著性
4. 回归分析：控制混杂变量，更精确地评估处理效应
5. 交互效应：探索不同子群体对处理的反应差异
6. 结论和建议：基于统计evidence做出业务决策
'''






