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


def cosine_distance(a, b):
    return spatial.distance.cosine(np.ravel(a), np.ravel(b))


def similarity(dist):
    return 1.0 - 2.0 * math.acos(1.0 - np.clip(dist, 0, 1)) / math.pi


def layer_distance(u, v):
    u /= np.linalg.norm(u, axis=3, keepdims=True) + keras.backend.epsilon()  # normalize along channel dimension
    v /= np.linalg.norm(v, axis=3, keepdims=True) + keras.backend.epsilon()
    diff = (u - v)**2       # squared difference
    mean = np.mean(np.mean(diff, axis=2), axis=1)  # average spatially
    return np.sum(mean)     # sum channel-wise


def net_distance(n, m, dist_func=layer_distance):
    dist = [dist_func(u, v) for u, v in zip(n, m)]
    return np.mean(dist)


def debug_chart(act1, act2):
    print "                cosine,    L2"
    print "embed layer {:10.4f} {:10.4f}".format(cosine_distance(act1[-1], act2[-1]), layer_distance(act1[-1], act2[-1]))
    print "all layers  {:10.4f} {:10.4f}".format(net_distance(act1, act2, dist_func=cosine_distance), net_distance(act1, act2, dist_func=layer_distance))
    print "layers[:5]  {:10.4f} {:10.4f}".format(net_distance(act1[:5], act2[:5], dist_func=cosine_distance), net_distance(act1[:5], act2[:5], dist_func=layer_distance))
    print '-------'


if __name__ == "__main__":
    print '-------'

    print "ex_ref vs ex_p0 (224, 224)"
    act_exref_224 = activations(load_image('ex_ref.png', target_size=(224, 224)))
    act_exp0_224 = activations(load_image('ex_p0.png', target_size=(224, 224)))
    debug_chart(act_exref_224, act_exp0_224)

    print "ex_ref vs ex_p1 (224, 224)"
    act_exp1_224 = activations(load_image('ex_p1.png', target_size=(224, 224)))
    debug_chart(act_exref_224, act_exp1_224)

    print "elephant vs self (224, 224)"
    act_elephant_224 = activations(load_image('elephant.jpg', target_size=(224, 224)))
    debug_chart(act_elephant_224, act_elephant_224)

    print "elephant vs elephant2 (224, 224)"
    act_elephant2_224 = activations(load_image('elephant2.jpg', target_size=(224, 224)))
    debug_chart(act_elephant_224, act_elephant2_224)

    print "elephant vs elephant2 (800, 500)"
    act_elephant_800 = activations(load_image('elephant.jpg', target_size=(800, 500)))
    act_elephant2_800 = activations(load_image('elephant2.jpg', target_size=(800, 500)))
    debug_chart(act_elephant_800, act_elephant2_800)

    print "elephant vs firetruck (800, 500)"
    act_firetruck_800 = activations(load_image('firetruck.jpg', target_size=(800, 500)))
    debug_chart(act_elephant_800, act_firetruck_800)
