import tensorflow as tf
from tensorflow import keras

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

fashionMNIST = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashionMNIST.load_data()
train_images = train_images/255.0
test_images = test_images/255.0

model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=30)
model.evaluate(test_images, test_labels)


model.summary()