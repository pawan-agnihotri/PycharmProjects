from sklearn.ensemble import RandomForestClassifier
import pandas
from sklearn.cross_validation import train_test_split
from sklearn import metrics

my_data = pandas.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/skulls.csv", delimiter=",")
print (my_data[0:5])
X = my_data.drop(my_data.columns[[0,1]], axis=1).values
print (X[0:5])
targetNames = my_data["epoch"].unique().tolist()
print (targetNames)
y = my_data["epoch"]
print(y[0:5])
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

skullsForest=RandomForestClassifier(n_estimators=10, criterion="entropy")
skullsForest.fit(X_trainset, y_trainset)

predForest = skullsForest.predict(X_testset)
print (predForest)
print (y_testset)

print("RandomForests's Accuracy: ")
print(metrics.accuracy_score(y_testset, predForest))

print(skullsForest.estimators_)