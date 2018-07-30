import numpy as np
import pandas as pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics

my_data = pandas.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/skulls.csv", delimiter=",")
# Remove the column containing the target name since it doesn't contain numeric values.
# Also remove the column that contains the row number
# axis=1 means we are removing columns instead of rows.
# Function takes in a pandas array and column numbers and returns a numpy array without
# the stated columns
def removeColumns(pandasArray, *column):
    return pandasArray.drop(pandasArray.columns[[column]], axis=1).values

def target(numpyArray, targetColumnIndex):
    target_dict = dict()
    target = list()
    count = -1
    for i in range(len(my_data.values)):
        if my_data.values[i][targetColumnIndex] not in target_dict:
            count += 1
            target_dict[my_data.values[i][targetColumnIndex]] = count
        target.append(target_dict[my_data.values[i][targetColumnIndex]])
    return np.asarray(target)

X = removeColumns(my_data, 0, 1)
y = target(my_data, 1)
#split the data into training and test
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=7)
#check the shape of dataset
print (X_trainset.shape)
print (y_trainset.shape)
print (X_testset.shape)
print (y_testset.shape)

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X_trainset,y_trainset)
neigh23 = KNeighborsClassifier(n_neighbors=23)
neigh23.fit(X_trainset,y_trainset)
neigh90 = KNeighborsClassifier(n_neighbors=90)
neigh90.fit(X_trainset,y_trainset)

pred = neigh.predict(X_testset)
pred23 = neigh23.predict(X_testset)
pred90 = neigh90.predict(X_testset)

print("Neigh's 1 Accuracy: ")
print (metrics.accuracy_score(y_testset, pred))
print("Neigh's 23 Accuracy: ")
print(metrics.accuracy_score(y_testset, pred23))
print("Neigh's 90 Accuracy: ")
print(metrics.accuracy_score(y_testset, pred90))