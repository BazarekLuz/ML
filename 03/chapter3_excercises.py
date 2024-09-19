import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import shift
from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

mnist = fetch_openml('mnist_784', as_frame=False)

X, y = mnist.data, mnist.target
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# EX 1
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
baseline_accuracy = knn_classifier.score(X_test, y_test)
print(baseline_accuracy)
# 0.9688 < 0.97

param_grid = [{'weights': ["uniform", "distance"], "n_neighbors": [3, 4, 5, 6]}]

knn_classifier = KNeighborsClassifier()
grid_search = GridSearchCV(knn_classifier, param_grid, cv=5)
grid_search.fit(X_train[:10_000], y_train[:10_000])
print(grid_search.best_estimator_)

grid_search.best_estimator_.fit(X_train, y_train)
tuned_accuracy = grid_search.score(X_test, y_test)
print(tuned_accuracy)


# EX 2
def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode='constant')
    return shifted_image.reshape([-1])


img = X_train[1010]
shifted_image_down = shift_image(img, 0, 5)
shifted_image_left = shift_image(img, -5, 0)

plt.figure(figsize=(28, 28))
plt.subplot(131)
plt.title('Original')
plt.imshow(img.reshape(28, 28), interpolation='nearest', cmap="Greys")

plt.subplot(132)
plt.title('Shifted down')
plt.imshow(shifted_image_down.reshape(28, 28), interpolation='nearest', cmap="Greys")

plt.subplot(133)
plt.title('Shifted left')
plt.imshow(shifted_image_left.reshape(28, 28), interpolation='nearest', cmap="Greys")

plt.show()

X_train_augmented = [image for image in X_train]
y_train_augmented = [label for label in y_train]

for dx, dy in ((-1, 0), (1, 0), (0, 1), (0, -1)):
    for image, label in zip(X_train, y_train):
        X_train_augmented.append(shift_image(image, dx, dy))
        y_train_augmented.append(label)

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

shuffle_idx = np.random.permutation(len(X_train_augmented))
X_train_augmented = X_train_augmented[shuffle_idx]
y_train_augmented = y_train_augmented[shuffle_idx]


knn_classifier = KNeighborsClassifier(n_neighbors=4, weights='distance')
knn_classifier.fit(X_train_augmented, y_train_augmented)

augmented_accuracy = knn_classifier.score(X_test, y_test)
print(augmented_accuracy)
