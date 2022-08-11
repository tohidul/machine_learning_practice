import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashionMNIST = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashionMNIST.load_data()

np.set_printoptions(linewidth=320)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss')<0.4):
            print("\nLoss is lower than 0.4 so cancelling the training!")
            self.model.stop_training = True

callbacks = myCallback()

print(f'LABEL: {train_labels[0]}')
print(f'IMAGE: \n{train_images[0]}')

# plt.imshow(train_images[0])
# plt.show()

# train_images = train_images/255.0
# test_images = test_images/255.0

# model = tf.keras.models.Sequential([keras.layers.Dense(28, input_shape = (28,), activation = tf.nn.relu),
#                                     keras.layers.Dense(10, activation = tf.nn.softmax)])
model = tf.keras.models.Sequential([keras.layers.Flatten(),
                                    keras.layers.Dense(128, activation = tf.nn.relu),
                                    keras.layers.Dense(256, activation = tf.nn.relu),
                                    keras.layers.Dense(10, activation = tf.nn.softmax)])


model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=30, callbacks = [callbacks])
model.evaluate(test_images, test_labels)

