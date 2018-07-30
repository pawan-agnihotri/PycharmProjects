#a little information about the dataset. We are using a dataset called skulls.csv,
## which contains the measurements made on Egyptian skulls from five epochs
#epoch - The epoch the skull as assigned to, a factor with levels
# c4000BC c3300BC, c1850BC, c200BC, and cAD150,
# where the years are only given approximately.
# mb - Maximal Breadth of the skull.
# bh - Basiregmatic Heights of the skull.
# bl - Basilveolar Length of the skull.
# nh - Nasal Heights of the skull.

import numpy as np
import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
#%matplotlib inline
from sklearn import metrics
import matplotlib.pyplot as plt

my_data = pandas.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/skulls.csv", delimiter=",")
print (my_data[0:5])
#Using my_data as the skulls.csv data read by pandas, declare the following variables:
#X as the Feature Matrix (data of my_data)
#y as the response vector (target)
#targetNames as the response vector names (target names)
#featureNames as the feature matrix column names
featureNames = list(my_data.columns.values)[2:6]
# Remove the column containing the target name since it doesn't contain numeric values.
# axis=1 means we are removing columns instead of rows.
X = my_data.drop(my_data.columns[[0,1]], axis=1).values
print (X[0:5])
targetNames = my_data["epoch"].unique().tolist()
print (targetNames)
y = my_data["epoch"]
print(y[0:5])
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
print (X_trainset.shape)
print (y_trainset.shape)
print (X_testset.shape)
print (y_testset.shape)
skullsTree = DecisionTreeClassifier(criterion="entropy")
skullsTree.fit(X_trainset,y_trainset)
predTree = skullsTree.predict(X_testset)
print (predTree [0:5])
print (y_testset [0:5])

print("DecisionTrees's Accuracy: ")
print (metrics.accuracy_score(y_testset, predTree))

# You can uncomment and install pydotplus if you have not installed before.
#!pip install pydotplus
#1 . Download and install graphviz-2.38.msi https://graphviz.gitlab.io/_pages/Download/Download_windows.html
# 2 . Set the path variable
# Control Panel > System and Security > System > Advanced System Settings > Environment Variables > Path > Edit add 'C:\Program Files (x86)\Graphviz2.38\bin'
dot_data = StringIO()
filename = "skulltree.png"
out=tree.export_graphviz(skullsTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')