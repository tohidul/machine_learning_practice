import os
import tensorflow as tf
from tensorflow import keras

mnistDataset = tf.keras.datasets.mnist

(x_train, y_train), _ = mnistDataset.load_data()

x_train = x_train/255.0

data_shape = x_train.shape

print(f"There are {data_shape[0]} examples with shape ({data_shape[1]}, {data_shape[2]})")


class myCallback(tf.keras.callbacks.Callback):
    # Define the correct function signature for on_epoch_end
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.99:
            print("\nReached 99% accuracy so cancelling training!")

            # Stop training once the above condition is met
            self.model.stop_training = True


# GRADED FUNCTION: train_mnist
def train_mnist(x_train, y_train):
    ### START CODE HERE

    # Instantiate the callback class
    callbacks = myCallback()

    # Define the model
    model = tf.keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # Compile the model
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Fit the model for 10 epochs adding the callbacks
    # and save the training history
    history = model.fit(x_train, y_train, epochs=8, callbacks=[callbacks])

    ### END CODE HERE

    return history

hist = train_mnist(x_train, y_train)