# Try to build a classifier for the MNIST dataset that achieves over 97% accuracy on the test set.

################################################    1    ################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from scipy.ndimage import shift
from sklearn.metrics import accuracy_score

########################### DATA LOAD ###########################
# Load the dataset
train_data = pd.read_csv("./mnist_train.csv")
test_data = pd.read_csv("./mnist_test.csv")

# Split features and labels
X_train = train_data.iloc[:, 1:].values  # Features
y_train = train_data.iloc[:, 0].values  # Labels

X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

# Normalize pixel values to [0, 1]
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

################################################    2    ################################################

# Apply translations (shifts) to images for data augmentation or transformations.
def shift_image(img, dx, dy):
    img = img.reshape((28, 28)) # img is reshaped from a 1D array of length 784 into a 2D array of shape (28, 28).
    shifted_img = shift(img, [dy, dx], cval=0, mode="constant") # filled with black (pixel value 0).
    return shifted_img.reshape([-1]) # The shifted image, which is still a 2D array of shape (28, 28), is flattened back into a 1D array of length 784

img = X_train[1000]
# shifted_image_down = shift_image(img, 0, 5)
shifted_image_up = shift_image(img, 0, -5)
# shifted_image_left = shift_image(img, -5, 0)
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

# X_augmented_train = [image for image in X_train[:10000]]
# y_augmented_train = [label for label in y_train[:10000]]

# for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
#     for image, label in zip(X_train[:10000], y_train[:10000]):
#         X_augmented_train.append(shift_image(image, dx, dy))
#         y_augmented_train.append(label)

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
print("Accuracy score : ", score)