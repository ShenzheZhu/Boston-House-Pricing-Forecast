import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams['font.sans-serif'] = ['SimHei']

# 数据导入

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B',
                 'LSTAT']
boston_df = pd.DataFrame(data, columns=feature_names)
boston_df["PRICE"] = target

# 检查是否存在空值
boston_df.isnull().sum()
# 查看数据大小
boston_df.shape
# 打印前五行数据
boston_df.head()
# 打印均值，最大值，最小值等信息
boston_df.describe()
# 提出异常样本有主意提升训练质量
boston_df = boston_df.loc[boston_df['PRICE'] < 50]

# correlation分析
# 分析各个特征与PRICE的相关性，并用可视化工具展示
"""
plt.figure(facecolor='grey')
corr = boston_df.corr()
corr = corr['PRICE']
corr[abs(corr) > 0.5].sort_values().plot.bar()
plt.show()
"""

# 分析前三相关的特征的散点图

"""
# LSTAT
plt.figure(facecolor='grey')
plt.scatter(boston_df['LSTAT'], boston_df['PRICE'], s=30, edgecolors='white')
plt.title('LSTAT')
plt.show()

# PTRATIO
plt.figure(facecolor='grey')
plt.scatter(boston_df['PTRATIO'], boston_df['PRICE'], s=30, edgecolors='white')
plt.title('PTRATIO')
plt.show()

# RM
plt.figure(facecolor='grey')
plt.scatter(boston_df['RM'], boston_df['PRICE'], s=30, edgecolors='white')
plt.title('RM')
plt.show()

# INDUS
plt.figure(facecolor='grey')
plt.scatter(boston_df['INDUS'], boston_df['PRICE'], s=30, edgecolors='white')
plt.title('INDUS')
plt.show()
"""

# 通过分析柱状图与散点图，我们发现LSTAT, PT RATIO, RM和PRICE的相关系数绝对值最高

# 数据筛选
# 在数据集的特征中删除掉除了上面三个之外的所有特征
boston_df = boston_df[['LSTAT', 'RM', 'PTRATIO', 'PRICE']]
y = np.array(boston_df['PRICE'])
boston_df = boston_df.drop(['PRICE'], axis=1)
x = np.array(boston_df)

# 划分训练集与测试集
# x_train, y_train 表示训练集中的特征值与目标值
# x_test, y_train 表示测试集中的特征值与目标值
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# 数据归一化
# 创建min_max标准化器实例
min_max = preprocessing.MinMaxScaler()
# 数据归一
x_train = min_max.fit_transform(x_train)
y_train = min_max.fit_transform(y_train.reshape(-1, 1))
x_test = min_max.fit_transform(x_test)
y_test = min_max.fit_transform(y_test.reshape(-1, 1))

# 模型训练

# 采用线性回归模型进行训练与预测
lr = LinearRegression()
# 拟合
lr.fit(x_train, y_train)
# 得出预测值
y_test_pre = lr.predict(x_test)
y_train_pre = lr.predict(x_train)

# 基于sklearn中的预测性能得分功能进行模型分析
# score越高则代表预测性能越好
score = lr.score(x_test, y_test)
MSE_test = mean_squared_error(y_test, y_test_pre)
MSE_train = mean_squared_error(y_train, y_train_pre)
coefficient = lr.coef_
intercept = lr.intercept_

print(f"Correlation Coefficient: w = %s, b = %s\n"
      f"Score:{score}\nTest Error:{MSE_test}\nTrain Error:{MSE_train}" % (coefficient, intercept))


# 绘制真实值与预测值的图像
plt.plot(y_test, label='real')
plt.plot(y_test_pre, label='predicted')
plt.title('Predicted and Real Value ' + r'$R^2=%.4f$' % (r2_score(y_train_pre, y_train)))
plt.legend()
plt.show()


