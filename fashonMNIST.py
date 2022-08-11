import tensorflow as tf
from tensorflow import keras
import numpy as np

fashionMNIST = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashionMNIST.load_data()