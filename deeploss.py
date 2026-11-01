"""Frozen VGG19 perceptual features used by the PIS training objective.

``weights`` defaults to ImageNet. A local Keras weight path is also accepted;
``None`` is intended for offline synthetic graph tests, not paper experiments.
"""

import numpy as np
import tensorflow as tf
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras.applications.vgg19 import VGG19
from tensorflow.compat.v1.keras import backend as K


def preprocess_input(x):
    """Preprocesses a tensor encoding a batch of images.
        x: input tensor, 4D in [-1,1]
        Preprocessed tensor.
    """
    x = (x + 1.0) / 2.0 * 255.0
    x = x[:, :, :, ::-1]
    x = x - np.array([103.939, 116.779, 123.68]).reshape((1, 1, 1, 3))
    return x


class VGG19Features(object):
    def __init__(self, session,
            feature_layers=None, feature_weights=None, gram_weights=None, weights="imagenet"):
        K.set_session(session)
        self.base_model = VGG19(
                include_top=False,
                weights=weights)
        self.base_model.trainable = False
        if feature_layers is None:
            feature_layers = [
                    "input_1",
                    "block1_conv2", "block2_conv2",
                    "block3_conv2", "block4_conv2",
                    "block5_conv2"]
        self.layer_names = [layer.name for layer in self.base_model.layers]
        features = []
        for name in feature_layers:
            if name == "input_1":
                features.append(self.base_model.input)
            elif name in self.layer_names:
                features.append(self.base_model.get_layer(name).output)
            else:
                raise KeyError("Invalid layer {}. Available layers: {}".format(name, self.layer_names))
        self.model = Model(inputs=self.base_model.input, outputs=features)
        self.model.trainable = False
        if feature_weights is None:
            feature_weights = len(feature_layers) * [1.0]
        if gram_weights is None:
            gram_weights = len(feature_layers) * [0.1]
        self.feature_weights = feature_weights
        self.gram_weights = gram_weights
        assert len(self.feature_weights) == len(features)
        self.use_gram = np.max(self.gram_weights) > 0.0

        self.variables = self.base_model.weights

    def extract_features(self, x):
        """x should be rgb in [-1,1]."""
        x = preprocess_input(x)
        features = self.model.predict(x)
        return features

    def make_feature_ops(self, x):
        """x should be rgb tensor in [-1,1]."""
        x = preprocess_input(x)
        features = self.model(x)
        return features

    def grams(self, fs):
        gs = list()
        for f in fs:
            shape = tf.shape(f)
            denominator = tf.cast(4 * shape[1] * shape[2], f.dtype)
            f = tf.reshape(f, [shape[0], shape[1] * shape[2], shape[3]])
            ft = tf.transpose(f, [0,2,1])
            g = tf.matmul(ft, f)
            g = g / denominator
            gs.append(g)
        return gs

    def make_loss_op(self, x, y):
        """x, y should be rgb tensors in [-1,1]."""
        x = preprocess_input(x)
        x_features = self.model(x)

        y = preprocess_input(y)
        y_features = self.model(y)

        x_grams = self.grams(x_features)
        y_grams = self.grams(y_features)

        losses = [
                tf.reduce_mean(tf.abs(xf - yf)) for xf, yf in zip(
                    x_features, y_features)]
        gram_losses = [
                tf.reduce_mean(tf.abs(xg - yg)) for xg, yg in zip(
                    x_grams, y_grams)]

        for i in range(len(losses)):
            losses[i] = self.feature_weights[i] * losses[i]
            gram_losses[i] = self.gram_weights[i] * gram_losses[i]
        loss = tf.add_n(losses)
        if self.use_gram:
            loss = loss + tf.add_n(gram_losses)

        self.losses = losses
        self.gram_losses = gram_losses

        return loss
