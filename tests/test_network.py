"""Small CPU checks for the released graph-mode network contracts."""
import numpy as np
import pytest
import tensorflow as tf

import deeploss
import models
import nn


tf.compat.v1.disable_v2_behavior()


@pytest.fixture(autouse=True)
def clean_graph():
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    yield
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()


def test_nin_accepts_multiple_batch_sizes_without_rebuilding():
    x = tf.compat.v1.placeholder(tf.float32, [None, 2, 3, 4])
    with models.model_arg_scope():
        output = nn.nin(x, 5)
    assert output.shape.as_list() == [None, 2, 3, 5]
    sample = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
    with tf.compat.v1.Session() as session:
        session.run(tf.compat.v1.global_variables_initializer())
        one = session.run(output, {x: sample})
        three = session.run(output, {x: np.repeat(sample, 3, axis=0)})
    assert one.shape == (1, 2, 3, 5)
    assert three.shape == (3, 2, 3, 5)
    np.testing.assert_allclose(three, np.repeat(one, 3, axis=0), rtol=1e-5, atol=1e-6)


def test_encoder_decoder_support_dynamic_batch_and_configured_identities():
    x = tf.compat.v1.placeholder(tf.float32, [None, 8, 8, 24])
    pose = tf.compat.v1.placeholder(tf.float32, [None, 16, 16, 3])
    encode_up = models.make_model("enc_up", models.enc_up, n_scales=2, n_filters=8)
    encode_down = models.make_model("enc_down", models.enc_down, n_scales=2)
    decode_up = models.make_model("dec_up", models.dec_up, n_scales=3, n_filters=4)
    decode_down = models.make_model("dec_down", models.dec_down, n_scales=3)
    decode_parameters = models.make_model("dec_params", models.dec_parameters)
    classifier = models.make_model("classifer", models.classifer, num_identities=3)
    hidden = encode_up(x, None, dropout_p=0.0)
    _, posterior, _ = encode_down(hidden, dropout_p=0.0)
    condition = decode_up(pose, dropout_p=0.0)
    decoded, _, _ = decode_down(condition, posterior, training=True, dropout_p=0.0)
    image = decode_parameters(decoded[-1])
    logits = classifier(posterior, dropout_p=0.0)
    with tf.compat.v1.Session() as session:
        session.run(tf.compat.v1.global_variables_initializer())
        for size in (1, 3):
            generated, first, second = session.run(
                [image, *logits],
                {x: np.zeros((size, 8, 8, 24), np.float32),
                 pose: np.zeros((size, 16, 16, 3), np.float32)},
            )
            assert generated.shape == (size, 16, 16, 3)
            assert first.shape == second.shape == (size, 3)
            assert np.isfinite(generated).all()


def test_transposed_convolution_keeps_compatible_dynamic_batch_interface():
    x = tf.compat.v1.placeholder(tf.float32, [None, 4, 4, 3])
    with models.model_arg_scope():
        output = nn.upsample(x, 2, method="conv_transposed")
    with tf.compat.v1.Session() as session:
        session.run(tf.compat.v1.global_variables_initializer())
        actual = session.run(output, {x: np.ones((2, 4, 4, 3), np.float32)})
    assert actual.shape == (2, 8, 8, 2)
    assert np.isfinite(actual).all()


def test_frozen_vgg_preserves_image_gradients_after_input_name_changes():
    tf.keras.layers.Input(shape=(32, 32, 3))
    with tf.compat.v1.Session() as session:
        features = deeploss.VGG19Features(
            session, feature_layers=["input_1", "block1_conv2"],
            feature_weights=[1.0, 1.0], gram_weights=[0.0, 0.0], weights=None,
        )
        assert not features.model.trainable_weights
        image = tf.compat.v1.placeholder(tf.float32, [None, 32, 32, 3])
        reference = tf.zeros_like(image)
        loss = features.make_loss_op(reference, image)
        gradient = tf.gradients(loss, image)[0]
        assert gradient is not None
        session.run(tf.compat.v1.global_variables_initializer())
        value, grad = session.run([loss, gradient], {image: np.ones((1, 32, 32, 3), np.float32)})
    assert np.isfinite(value) and value > 0
    assert np.isfinite(grad).all()
    assert np.any(grad != 0)
