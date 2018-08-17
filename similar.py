"""similar.py
Calculate visual similarity of two photos.
"""

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import numpy as np


model = tf.keras.applications.ResNet50(include_top=False, pooling='avg')


def load_image(img_path='elephant.jpg'):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return preprocess_batch(x)


def preprocess_batch(x):  # (None, 224, 224, 3) 'channels_last'
    if len(np.shape(x)) == 3:
        x = np.expand_dims(x, axis=0)
    return preprocess_input(x)


def calc(x):
    return model.predict(x)


if __name__ == "__main__":
    print calc(load_image())
