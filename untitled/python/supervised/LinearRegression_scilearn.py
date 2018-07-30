#Regression outputs a response that is ordered and continuous.
#Diabetes Dataset
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import numpy as np

diabetes = load_diabetes()
diabetes_X = diabetes.data[:, None, 2]
LinReg = LinearRegression()
#train and test for given dataset
X_trainset, X_testset, y_trainset, y_testset = train_test_split(diabetes_X, diabetes.target, test_size=0.3, random_state=7)
LinReg.fit(X_trainset, y_trainset)
plt.scatter(X_testset, y_testset, color='black')
plt.plot(X_testset, LinReg.predict(X_testset), color='blue', linewidth=3)
plt.show()
print("MAE - Mean Avg Error")
print(np.mean(abs(LinReg.predict(X_testset) - y_testset)))
print("MSE - Mean Square Error")
print(np.mean((LinReg.predict(X_testset) - y_testset) ** 2) )
print("RMSE - Root Mean Square Error")
print(np.mean((LinReg.predict(X_testset) - y_testset) ** 2) ** (0.5) )