
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score 

#This is file for Knn distance function from here we can import 3 distance function to our main program file

"""fit_transform() is used on the training data so that we can scale the training data and also learn the scaling parameters of that data.
 Here, the model built by us will learn the mean and variance of the features of the training set. These learned parameters are then used to 
 scale our test data."""

dataset = pd.read_csv('data/parkinsons.csv')
X = dataset.iloc[:, 1:-1]   # input feactures 
y = dataset.iloc[:, -1]     # output

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

def euclidean_dist(X_train, X_test, y_train, y_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    result = []
    for i in range(4, 11):
        classifier = KNeighborsClassifier(n_neighbors = i,metric= 'euclidean' ,p=2)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        result.append(accuracy_score(y_test, y_pred))
    return max(result), result.index(max(result)) + 4

def mahalanobis_dist(X_train, X_test, y_train, y_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    result = []
    for i in range(4, 11):
        classifier = KNeighborsClassifier(metric='mahalanobis', n_neighbors = i, metric_params={'VI': np.cov(X)})
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        result.append(accuracy_score(y_test, y_pred))
    return max(result), result.index(max(result)) + 4

def correlation_dist(X_train, X_test, y_train, y_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    result = []
    for i in range(4, 11):
        classifier = KNeighborsClassifier(n_neighbors = i,metric= 'correlation')
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        result.append(accuracy_score(y_test, y_pred))
    return max(result), result.index(max(result)) + 4



print(euclidean_dist(X_train, X_test, y_train, y_test))
print(mahalanobis_dist(X_train, X_test, y_train, y_test))
print(correlation_dist(X_train, X_test, y_train, y_test))