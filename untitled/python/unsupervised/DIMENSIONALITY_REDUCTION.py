#We will first be looking at Feature Selection with VarianceThreshold.
# VarianceThreshold is a useful tool to removing features with a threshold variance.
# It is a simple and basic Feature Selection.
from sklearn.feature_selection import VarianceThreshold
# nstantiate VarianceThreshold as a variable
sel = VarianceThreshold()
#VarianceThreshold removes all zero-variance features by default.
# These features are any constant value features.

dataset = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
print (dataset)
#VarianceThreshold removes all zero-variance features by default.
# These features are any constant value features
sel.fit_transform(dataset)
#We can change the threshold by adding threshold='threshold value'
# inside the brackets during the instantiation of VarianceThreshold.
# Where 'threshold value' is equal to p(1-p) Where 'p' is your threshold % in decimal format.
# 60% threshold
sel60 = VarianceThreshold(threshold=(0.6 * (1 - 0.6)))
#We will need to import SelectKBest from sklearn.feature_selection,
# chi2 from sklearn.feature_selection, numpy as np, and pandas.
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
import pandas
my_data = pandas.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/skulls.csv", delimiter=",")

# Remove the column containing the target name since it doesn't contain numeric values.
# Also remove the column that contains the row number
# axis=1 means we are removing columns instead of rows.
# Function takes in a pandas array and column numbers and returns a numpy array without
# the stated columns
def removeColumns(pandasArray, *column):
    return pandasArray.drop(pandasArray.columns[[column]], axis=1).values
X = removeColumns(my_data, 0, 1)

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
y = target(my_data, 1)
print (X.shape)
#How Univariance works is that it selects features based off of univariance statistical tests.
# chi2 is used as a univariance scoring function which returns p values.
# We specified k=3 for the 3 best features to be chosen.
X_new = SelectKBest(chi2, k=3).fit_transform(X, y)
#we will use the fit_transform function with parameters
# X, y of SelectKBest with parameters chi2, k=3. This will be stored as X_new.
print (X_new.shape)