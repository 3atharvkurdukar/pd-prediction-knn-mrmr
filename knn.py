# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# from fknn import FuzzyKNN
import sklearn.feature_selection as fs

# %%
dataset = pd.read_csv('data/parkinsons.csv')
X = dataset.iloc[:, 1:-1]   # input feactures
y = dataset.iloc[:, -1]     # output

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)


# %%
def euclidean_dist(X_train, X_test, y_train, y_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    result = []
    for i in range(4, 11):
        classifier = KNeighborsClassifier(
            n_neighbors=i, metric='euclidean', p=2)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        result.append(accuracy_score(y_test, y_pred))
    return max(result), result.index(max(result)) + 4


# %%
def mahalanobis_dist(X_train, X_test, y_train, y_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    result = []
    for i in range(4, 11):
        classifier = KNeighborsClassifier(
            metric='mahalanobis', n_neighbors=i, metric_params={'VI': np.cov(X)})
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        result.append(accuracy_score(y_test, y_pred))
    return max(result), result.index(max(result)) + 4


# %%
def correlation_dist(X_train, X_test, y_train, y_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    result = []
    for i in range(4, 11):
        classifier = KNeighborsClassifier(n_neighbors=i, metric='correlation')
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        result.append(accuracy_score(y_test, y_pred))
    return max(result), result.index(max(result)) + 4


# %%
def fuzzy(X_train, X_test, y_train, y_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    result = []
    for i in range(5, 12, 2):
        classifier = FuzzyKNN(k=i)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        result.append(accuracy_score(y_test, y_pred))
    return max(result), result.index(max(result)) + 4


# %%
def weighted_knn(X_train, X_test, y_train, y_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    result = []
    for i in range(4, 11):
        mutual_info = fs.mutual_info_classif(
            X_train, y_train, discrete_features='auto', n_neighbors=i, copy=True, random_state=None)
        X_train_new = X_train * mutual_info
        result.append(euclidean_dist(X_train_new, X_test, y_train, y_test))
    return max(result), result.index(max(result)) + 4


# %%
print(euclidean_dist(X_train, X_test, y_train, y_test))
# %%
print(mahalanobis_dist(X_train, X_test, y_train, y_test))
# %%
print(correlation_dist(X_train, X_test, y_train, y_test))
# %%
# print(fuzzy(X_train, X_test, y_train, y_test))
print(weighted_knn(X_train, X_test, y_train, y_test))
