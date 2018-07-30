import numpy as np
import pandas as pandas
from sklearn.neighbors import KNeighborsClassifier

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
print (X[0])
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X,y)
neigh7 = KNeighborsClassifier(n_neighbors=7)
neigh7.fit(X,y)
print (X[30])
#reshape for single feature
print (X[30].reshape(1,-1))
#reshape for single sample
print (X[30].reshape(-1,1))
print (y[30])

print("Neigh's Prediction: ")
print (neigh.predict(X[30].reshape(1,-1)))
print("Neigh7's Prediction: ")
print(neigh7.predict(X[30].reshape(1,-1)))
print('Actual:')
print (y[30])