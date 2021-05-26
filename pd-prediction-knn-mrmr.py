# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import mifs
# %%
dataset = pd.read_csv('data/parkinsons.csv')
X = dataset.iloc[:, 1:-1]   # input feactures
y = dataset.iloc[:, -1]     # output
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)
# %%
for i in range(1, 22):
    feat_selector = mifs.MutualInformationFeatureSelector('MRMR', k=i)
    feat_selector.fit(X_train, y_train)
    # call transform() on X to filter it down to selected features
    X_filtered = feat_selector.transform(X_train.values)
    # Create list of features
    feature_name = X_train.columns[feat_selector.ranking_]
    print(feature_name)
