# Try to build a classifier for the MNIST dataset that achieves over 97% accuracy on the test set.

################################################    1    ################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from scipy.ndimage import shift
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

########################### DATA LOAD ###########################

# 1. Load MNIST Dataset Using PyTorch
transform = transforms.Compose([
    transforms.ToTensor(),               # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,)) # Normalize pixel values to range [-1, 1]
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Convert the dataset to numpy arrays for use with sklearn
X_train = train_dataset.data.numpy().reshape(-1, 28 * 28)  # Flatten images
y_train = train_dataset.targets.numpy()

X_test = test_dataset.data.numpy().reshape(-1, 28 * 28)
y_test = test_dataset.targets.numpy()

# Normalize pixel values to [0, 1] for compatibility with sklearn
X_train = X_train / 255.0
X_test = X_test / 255.0

#################################################################

# mnist = fetch_openml('mnist_784', as_frame=False)
# X, y = mnist.data, mnist.target
# X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# KNN_classifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
base_accuracy = knn_clf.score(X_test, y_test)

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
# print(tuned_accuracy) # Reached over 97%
# Base KNN Accuracy on Test Set: 0.9714

################################################    2    ################################################

# Apply translations (shifts) to images for data augmentation or transformations.
def shift_image(img, dx, dy):
    img = img.reshape((28, 28)) # img is reshaped from a 1D array of length 784 into a 2D array of shape (28, 28).
    shifted_img = shift(img, [dy, dx], cval=0, mode="constant") # filled with black (pixel value 0).
    return shifted_img.reshape([-1]) # The shifted image, which is still a 2D array of shape (28, 28), is flattened back into a 1D array of length 784

img = X_train[1000]
shifted_image_down = shift_image(img, 0, 5)
shifted_image_up = shift_image(img, 0, -5)
shifted_image_left = shift_image(img, -5, 0)
shifted_image_right = shift_image(img, 5, 0)

plt.figure(figsize=(12, 3))
plt.subplot(131)
plt.title("Original", fontsize=15)
plt.imshow(img.reshape(28, 28), interpolation="nearest", cmap="Greys")

plt.subplot(132)
plt.title("Shifted up", fontsize=14)
plt.imshow(shifted_image_up.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.subplot(133)

plt.title("Shifted right", fontsize=14)
plt.imshow(shifted_image_right.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.show()

X_augmented_train = [image for image in X_train]
y_augmented_train = [image for image in y_train]

for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    for image, label in zip(X_train, y_train):
        X_augmented_train.append(shift_image(image, dx, dy))
        y_augmented_train.append(label)

X_augmented_train = np.array(X_augmented_train)
y_augmented_train = np.array(y_augmented_train)

# Shuffles the augmented dataset to ensure that the training process is unbiased by the order of the data.
random_inx = np.random.permutation(len(X_augmented_train))
X_augmented_train = X_augmented_train[random_inx]
y_augmented_train = y_augmented_train[random_inx]

knn_clf = KNeighborsClassifier(**grid_srch.best_params_)
knn_clf.fit(X_augmented_train, y_augmented_train)

y_prediction = knn_clf.predict(X_test)
score = accuracy_score(y_test, y_prediction)
# print("Accuracy score : ", score)
# Augmented KNN Accuracy on Test Set: 0.9754

################################################    3    ################################################

# The goal is to predict whether or not a passenger survived based on attributes such as their age, sex, passenger class, where they embarked and so on.

try:
  from sklearn.impute import SimpleImputer
except ImportError:
  from sklearn.preprocessing import Imputer as SimpleImputer

train_data = pd.read_csv("./titanic/train.csv")
test_data = pd.read_csv("./titanic/test.csv")


# Survived: that's the target, 0 means the passenger did not survive, while 1 means he/she survived.
# Pclass: passenger class.
# Name, Sex, Age: self-explanatory
# SibSp: how many siblings & spouses of the passenger aboard the Titanic.
# Parch: how many children & parents of the passenger aboard the Titanic.
# Ticket: ticket id
# Fare: price paid (in pounds)
# Cabin: passenger's cabin number
# Embarked: where the passenger embarked the Titanic

train_data.head()

train_data.info()

train_data.describe()

train_data["Survived"].value_counts()

train_data["Pclass"].value_counts()

train_data["Sex"].value_counts()

train_data["Embarked"].value_counts()

class DataFramSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
      return self
    def transform(self, X):
      return X[self.attribute_names]

# build the pipeline for the numerical attributes:
num_pipeline = Pipeline([
    ("select_numeric", DataFramSelector(["Age", "SibSp", "Parch", "Fare"])),
    ("imputer", SimpleImputer(strategy="median")),
])

num_pipeline.fit_transform(train_data)

class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X], index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)

try:
  from sklearn.preprocessing import OrdinalEncoder
  from sklearn.preprocessing import OneHotEncoder
except ImportError:
  from future_encoders import OrdinalEncoder

# build the pipeline for the categorical attributes:
cat_pipeline = Pipeline([
    ("select_cat", DataFramSelector(["Pclass", "Sex", "Embarked"])),
    ("imputer", MostFrequentImputer()),
    ("cat_encoder", OneHotEncoder(sparse_output=False))
])

cat_pipeline.fit_transform(train_data)

# join the numerical and categorical pipelines
from sklearn.pipeline import FeatureUnion
preprocess_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

X_train = preprocess_pipeline.fit_transform(train_data)
X_train

y_train = train_data["Survived"]

from sklearn.svm import SVC

svm_clf = SVC(gamma="auto")
svm_clf.fit(X_train, y_train)

X_test = preprocess_pipeline.transform(test_data)
y_pred = svm_clf.predict(X_test)

from sklearn.model_selection import cross_val_score

svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
svm_scores.mean()

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
forest_scores.mean()

plt.figure(figsize=(8, 4))
plt.plot([1]*10, svm_scores, ".")
plt.plot([2]*10, forest_scores, ".")
plt.boxplot([svm_scores, forest_scores], tick_labels=("SVM","Random Forest"))
plt.ylabel("Accuracy", fontsize=14)
plt.show()