import keras.src.datasets.mnist
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras.backend import get_value, set_value
import tensorflow as tf

(X_train_full, y_train_full), (X_test, y_test) = keras.src.datasets.mnist.load_data()

X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.

n_rows = 4
n_cols = 10
plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
        plt.axis('off')
        plt.title(y_train[index])
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()

class ExponentialLearningRate(keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []
    def on_batch_end(self, batch, logs=None):
        self.rates.append(get_value(self.model.optimizer.learning_rate))
        self.losses.append(logs['loss'])
        set_value(self.model.opimizer.learning_rate, self.model.optimizer.learning_rate * self.factor)

np.random.seed(42)
tf.random.set_seed(42)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

optimizer = keras.optimizers.SGD(learning_rate=3e-1)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
expon_lr = ExponentialLearningRate(factor=1.005)

