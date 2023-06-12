import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import datasets

plt.rcParams['font.sans-serif'] = ['SimHei']


class LinearRegressionTraining:
    def data_import(self):
        data_url = "http://lib.stat.cmu.edu/datasets/boston"
        raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        target = raw_df.values[1::2, 2]
        feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B',
                         'LSTAT']

        # print(target)

        boston_df = pd.DataFrame(data, columns=feature_names)
        boston_df["PRICE"] = target

        # 检查是否存在空值
        # boston_df.isnull().sum()
        # 查看数据大小
        # print(boston_df.shape)
        # 打印前五行数据
        # boston_df.head()
        # 打印均值，最大值，最小值等信息
        print(boston_df.describe())



if __name__ == "__main__":
    test = LinearRegressionTraining()
    test.data_import()
