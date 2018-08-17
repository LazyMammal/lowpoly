"""similar.py
Calculate visual similarity of two photos.
"""

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import numpy as np
from scipy import spatial


model = tf.keras.applications.ResNet50(include_top=False, pooling='avg')


def load_image(img_path='elephant.jpg', target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    return preprocess_batch(x)


def preprocess_batch(x):  # 'channels_last'  (None, height, width, 3)
    if len(np.shape(x)) == 3:
        x = np.expand_dims(x, axis=0)
    return preprocess_input(x)


def calc(x):
    return model.predict(x)


def distance(a, b):
    return spatial.distance.cosine(a, b)


if __name__ == "__main__":
    embed = calc(load_image())
    print np.shape(embed)
    print embed
    print '-------'

    x1 = load_image('elephant.jpg', target_size=(800, 600))
    x2 = load_image('elephant.jpg', target_size=(400, 300))
    result = [calc(x1), calc(x2)]
    print result
    print distance(result[0], result[1])
    print '-------'

    x1 = load_image('elephant.jpg', target_size=(224, 224))
    x2 = load_image('elephant2.jpg', target_size=(224, 224))
    batch = np.append(x1, x2, axis=0)
    result = calc(batch)
    print result
    print distance(result[0], result[1])
    print '-------'
