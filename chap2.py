# Try to build a classifier for the MNIST dataset that achieves over 97% accuracy on the test set.

################################################    1    ################################################

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier

mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist.data, mnist.target
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# KNN_classifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
base_accuracy = knn_clf.score(X_test, y_test)

from sklearn.model_selection import GridSearchCV

# To speed up the search, let's train only on the first 10,000 images
grid_cv_params = [{'weights': ['uniform', 'distance'], 'n_neighbors': [3,4,5,6]}]

knn_clf = KNeighborsClassifier()
grid_srch = GridSearchCV(knn_clf, grid_cv_params, cv = 5)
grid_srch.fit(X_train[:10_000], y_train[:10_000])

grid_srch.best_params_

grid_srch.best_score_

# Retrain data with the best model (hyperparameter tunned)
grid_srch.best_estimator_.fit(X_train, y_train)
tuned_accuracy = grid_srch.score(X_test, y_test)
print(tuned_accuracy) # Reached over 97%

################################################    2    ################################################
