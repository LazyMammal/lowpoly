"""similar.py
Calculate visual similarity of two photos.
"""

import math
import tensorflow as tf
import keras  # NOT from tensorflow (throws error "AttributeError: 'Tensor' object has no attribute '_keras_shape'")
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras import backend as K
import numpy as np
from scipy import spatial


model = keras.applications.ResNet50(include_top=False)
outputs = [layer.output for layer in model.layers][1:]      # all layer outputs (except input)
# functor = K.function([model.input, K.learning_phase()], outputs)    # evaluation function
layer_model = Model(inputs=[model.input], outputs=outputs)  # alternate form (either use this or K.function, not both)


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


def activations(x):
    # return functor([x, 0.])  # test mode == 0
    return layer_model.predict(x)


def distance(a, b):
    return spatial.distance.cosine(np.ravel(a), np.ravel(b))


def similarity(a, b):
    return 1.0 - 2.0 * math.acos(1.0 - np.clip(distance(a, b), -1, 1)) / math.pi


if __name__ == "__main__":
    embed = calc(load_image())
    print np.shape(embed)
    print '-------'

    act = activations(load_image())
    for c, layer in enumerate(act):
        print "layer ", c, np.shape(layer)
    print '-------'

    x1 = load_image('elephant.jpg', target_size=(224, 224))
    x2 = load_image('elephant2.jpg', target_size=(224, 224))
    batch = np.append(x1, x2, axis=0)
    result = calc(batch)
    print np.shape(result)
    print distance(result[0], result[1]), similarity(result[0], result[1])
    print '-------'

    x1 = load_image('elephant.jpg', target_size=(800, 500))
    x2 = load_image('elephant2.jpg', target_size=(800, 500))
    batch = np.append(x1, x2, axis=0)
    result = calc(batch)
    print np.shape(result)
    print distance(result[0], result[1]), similarity(result[0], result[1])
    print '-------'
