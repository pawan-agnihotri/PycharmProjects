#type of Support Vector Machine
from sklearn.svm import SVR
import numpy as np
np.random.seed(5)
X = np.sort(10 * np.random.rand(30, 1), axis=0)
print (X)
#we will take the sin of each using the sin function of np (sinusoidal function),
# and then use the ravel function to format it correctly. This will be stored as y.
y = np.sin(X).ravel()
print (y)

#we will be looking at 3 different kernels that SVR uses: rbf, linear, and sigmoid.
svr_rbf = SVR(kernel='rbf', C=1e3)
svr_linear = SVR(kernel='linear', C=1e3)
svr_sigmoid = SVR(kernel='sigmoid', C=1e3)

svr_rbf.fit(X,y)
svr_linear.fit(X,y)
svr_sigmoid.fit(X,y)

y_pred_rbf = svr_rbf.predict(X)
y_pred_linear = svr_linear.predict(X)
y_pred_sigmoid = svr_sigmoid.predict(X)
print (y_pred_rbf)
print (y_pred_linear)
print (y_pred_sigmoid)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition

plt.scatter(X, y, c='k', label='data')
plt.plot(X, y_pred_rbf, c='g', label='RBF model')
plt.plot(X, y_pred_linear, c='r', label='Linear model')
#plt.plot(X, y_pred_sigmoid, c='b', label='Sigmoid model')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
plt.show()