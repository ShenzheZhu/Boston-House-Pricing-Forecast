import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei']

# Data Import
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B',
                 'LSTAT']
boston_df = pd.DataFrame(data, columns=feature_names)
boston_df["PRICE"] = target

# Check if there is any Null value
boston_df.isnull().sum()
# Check the size of data
boston_df.shape
# Print out the first five rows of data
boston_df.head()
# Print mean, maximum, minimum and other information
boston_df.describe()
# Presenting anomalous samples helps to improve the training quality
# boston_df = boston_df.loc[boston_df['PRICE'] < 50]

# correlation analysis
# Analyze the correlation of each feature with PRICE and present it with visualization
"""
plt.figure(facecolor='grey')
corr = boston_df.corr()
corr = corr['PRICE']
corr[abs(corr) > 0.5].sort_values().plot.bar()
plt.show()
"""

# Analysis of scatter plots of the first three features associated with PRICE

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

boston_df.corr()['PRICE'].abs().sort_values(ascending=False).head(4)

# By analyzing the histogram and scatter plot, we found that LSTAT, PT RATIO, RM and PRICE have the highest absolute
# values of correlation coefficients
# Conclusion: The higher the average number of rooms per dwelling, the higher the
# average housing price; the greater the proportion of low-status population in the area, the lower the average
# housing price。

# Data Filtering
# Remove all but the above three features from the dataset's features
# boston_df = boston_df[['LSTAT', 'RM', 'PTRATIO', 'PRICE']]
y = np.array(boston_df['PRICE'])
boston_df = boston_df.drop(['PRICE'], axis=1)
X = np.array(boston_df)

# Dividing the training set and test set
# x_train, y_train denote the feature values and target values in the training set
# x_test, y_train denotes the feature values and target values in the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=888)

"""
# Data normalization
# Create min_max normalizer instance
min_max = preprocessing.MinMaxScaler()
# Normalize the data
X_train = min_max.fit_transform(X_train)
y_train = min_max.fit_transform(y_train.reshape(-1, 1))
X_test = min_max.fit_transform(X_test)
y_test = min_max.fit_transform(y_test.reshape(-1, 1))
"""

# Model training

# Training and prediction using Linear Regression model
lr = LinearRegression()
# Regression
lr.fit(X_train, y_train)
# Get the predicted value
y_test_pre_linear = lr.predict(X_test)
y_train_pre_linear = lr.predict(X_train)
# Model analysis based on the prediction performance score function in sklearn
# Higher score means better prediction performance
score = lr.score(X_test, y_test)
MSE_test = mean_squared_error(y_test, y_test_pre_linear)
MSE_train = mean_squared_error(y_train, y_train_pre_linear)
coefficient = lr.coef_
intercept = lr.intercept_
r2 = r2_score(y_train_pre_linear, y_train)
print("LINEAR REGRESSION")
print(f"Correlation Coefficient: w = %s, b = %s\nR2: {r2}\n"
      f"Score:{score}\nTest Error:{MSE_test}\nTrain Error:{MSE_train}\n" % (coefficient, intercept))


"""Conclusion: the following horizontal optimization scheme is not ideal, and when trying to de-regularize the linear 
regression model, the size of the data set is too small, which instead makes the performance of the model much worse 
It is recommended to optimize vertically, i.e., to develop more learning models
"""

# Training and predicting using Ridge Regression model
ridge = Ridge()
ridge.fit(X_train, y_train)
y_test_pre_ridge = ridge.predict(X_test)
y_train_pre_ridge = ridge.predict(X_train)

score = ridge.score(X_test, y_test)
MSE_test = mean_squared_error(y_test, y_test_pre_ridge)
MSE_train = mean_squared_error(y_train, y_train_pre_ridge)
coefficient = ridge.coef_
intercept = ridge.intercept_

print("RIDGE REGRESSION")
print(f"Correlation Coefficient: w = %s, b = %s\n"
      f"Score:{score}\nTest Error:{MSE_test}\nTrain Error:{MSE_train}\n" % (coefficient, intercept))

# Training and predicting using Lasso Regression model
la = Lasso()
la.fit(X_train, y_train)
y_test_pre_la = la.predict(X_test)
y_train_pre_la = la.predict(X_train)

score = la.score(X_test, y_test)
MSE_test = mean_squared_error(y_test, y_test_pre_la)
MSE_train = mean_squared_error(y_train, y_train_pre_la)
coefficient = la.coef_
intercept = la.intercept_
print("LASSO REGRESSION")
print(f"Correlation Coefficient: w = %s, b = %s\n"
      f"Score:{score}\nTest Error:{MSE_test}\nTrain Error:{MSE_train}\n" % (coefficient, intercept))

# Training and predicting using Elastic Net model
en = ElasticNet()
en.fit(X_train, y_train)
y_test_pre_en = en.predict(X_test)
y_train_pre_en = en.predict(X_train)

score = en.score(X_test, y_test)
MSE_test = mean_squared_error(y_test, y_test_pre_en)
MSE_train = mean_squared_error(y_train, y_train_pre_en)
coefficient = en.coef_
intercept = en.intercept_
print("ElasticNet REGRESSION")
print(f"Correlation Coefficient: w = %s, b = %s\n"
      f"Score:{score}\nTest Error:{MSE_test}\nTrain Error:{MSE_train}\n" % (coefficient, intercept))


# Training and predicting using GradientBoosting model
gbr = ensemble.GradientBoostingRegressor()
gbr.fit(X_train, y_train)
y_test_pre_gbr = gbr.predict(X_test)
y_train_pre_gbr = gbr.predict(X_train)


score = gbr.score(X_test, y_test)
MSE_test = mean_squared_error(y_test, y_test_pre_gbr)
MSE_train = mean_squared_error(y_train, y_train_pre_gbr)
r2 = r2_score(y_train_pre_gbr, y_train)
print("GradientBoostingRegressor")
print(f"R2: {r2}\nScore:{score}\nTest Error:{MSE_test}\nTrain Error:{MSE_train}\n")


# Plot the image of the true and predicted values
plt.plot(y_test, label='real')
plt.plot(y_test_pre_gbr, label='Gradient Boosting Regressor')
plt.plot(y_test_pre_linear, label='Linear Regression')
plt.plot(y_test_pre_ridge, label='Ridge Regression')
plt.plot(y_test_pre_la, label='Lasso Regression')
plt.plot(y_test_pre_en, label='ElasticNet Regression')

plt.title('Predicted Value and Real Value')
plt.legend()
plt.show()
