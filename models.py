"""PIS variational U-Net; historical variable scopes are kept for checkpoints."""

import tensorflow as tf
from tf_slim import arg_scope

import nn


def model_arg_scope(**kwargs):
    """Create new counter and apply arg scope to all arg scoped nn
    operations."""
    counters = {}
    return arg_scope(
        [nn.conv2d, nn.deconv2d, nn.residual_block, nn.dense, nn.activate],
        counters=counters, **kwargs)


def make_model(name, template, **kwargs):
    """Create model with fixed kwargs."""
    run = lambda *args, **kw: template(*args, **dict((k, v) for kws in (kw, kwargs) for k, v in kws.items()))
    return tf.compat.v1.make_template(name, run, unique_name_=name)


def dec_up(
        c, init=False, dropout_p=0.5,
        n_scales=1, n_residual_blocks=2, activation="elu", n_filters=64, max_filters=128):
    with model_arg_scope(
            init=init, dropout_p=dropout_p, activation=activation):
        hs = []
        h = nn.nin(c, n_filters)
        for l in range(n_scales):
            for i in range(n_residual_blocks):
                h = nn.residual_block(h)
                hs.append(h)
            if l + 1 < n_scales:
                n_filters = min(2 * n_filters, max_filters)
                h = nn.downsample(h, n_filters)
        return hs  # 128, 32  64,64  32, 128  16,128  8, 4, 2  length:14


def dec_down(
        gs, zs_posterior, training, init=False, dropout_p=0.5,
        n_scales=1, n_residual_blocks=2, activation="elu",
        n_latent_scales=2):
    assert n_residual_blocks % 2 == 0
    gs = list(gs)
    zs_posterior = list(zs_posterior)
    with model_arg_scope(
            init=init, dropout_p=dropout_p, activation=activation):
        hs = []  # hidden units
        ps = []  # priors
        zs = []  # prior samples
        n_filters = gs[-1].shape.as_list()[-1]
        h = nn.nin(gs[-1], n_filters)
        for l in range(n_scales):
            for i in range(n_residual_blocks // 2):
                h = nn.residual_block(h, gs.pop())
                hs.append(h)
            if l < n_latent_scales:
                spatial_shape = h.shape.as_list()[1]
                n_h_channels = h.shape.as_list()[-1]
                if spatial_shape == 1:
                    p = latent_parameters(h)
                    ps.append(p)
                    z_prior = latent_sample(p)
                    zs.append(z_prior)
                else:
                    if training:
                        z_posterior_groups = nn.split_groups(zs_posterior[0])
                    p_groups = []
                    z_groups = []
                    p_features = tf.compat.v1.space_to_depth(nn.residual_block(h), 2)

                    for i in range(4):
                        p_group = latent_parameters(p_features, num_filters=n_h_channels)
                        p_groups.append(p_group)
                        z_group = latent_sample(p_group)
                        z_groups.append(z_group)
                        if training:
                            feedback = z_posterior_groups.pop(0)
                        else:
                            feedback = z_group
                        if i + 1 < 4:
                            p_features = nn.residual_block(p_features, feedback)
                    if training:
                        assert not z_posterior_groups
                    p = nn.merge_groups(p_groups)
                    ps.append(p)
                    z_prior = nn.merge_groups(z_groups)
                    zs.append(z_prior)
                if training:
                    z = zs_posterior.pop(0)
                else:
                    z = z_prior
                for i in range(n_residual_blocks // 2):
                    n_h_channels = h.shape.as_list()[-1]
                    h = tf.concat([h, z], axis=-1)
                    h = nn.nin(h, n_h_channels)
                    h = nn.residual_block(h, gs.pop())
                    hs.append(h)
            else:
                for i in range(n_residual_blocks // 2):
                    h = nn.residual_block(h, gs.pop())
                    hs.append(h)
            if l + 1 < n_scales:
                n_filters = gs[-1].shape.as_list()[-1]
                h = nn.upsample(h, n_filters)
        assert not gs
        if training:
            assert not zs_posterior

        return hs, ps, zs


def enc_up(
        x, c, init=False, dropout_p=0.5,
        n_scales=1, n_residual_blocks=2, activation="elu", n_filters=64, max_filters=128):
    with model_arg_scope(
            init=init, dropout_p=dropout_p, activation=activation):
        hs = []
        xc = x
        h = nn.nin(xc, n_filters)
        for l in range(n_scales):
            for i in range(n_residual_blocks):
                h = nn.residual_block(h)
                hs.append(h)
            if l + 1 < n_scales:
                n_filters = min(2 * n_filters, max_filters)
                h = nn.downsample(h, n_filters)
        return hs  # 64,64  32, 128  16,128  8, 4, 2


def enc_down(
        gs, init=False, dropout_p=0.5,
        n_scales=1, n_residual_blocks=2, activation="elu",
        n_latent_scales=2):
    assert n_residual_blocks % 2 == 0
    gs = list(gs)
    with model_arg_scope(
            init=init, dropout_p=dropout_p, activation=activation):
        hs = []  # hidden units
        qs = []  # posteriors
        zs = []  # samples from posterior
        n_filters = gs[-1].shape.as_list()[-1]
        h = nn.nin(gs[-1], n_filters)
        for l in range(n_scales):
            for i in range(n_residual_blocks // 2):
                h = nn.residual_block(h, gs.pop())
                hs.append(h)
            if l < n_latent_scales:
                q = latent_parameters(h)  # 4, 2, 4, 2, 4 128
                qs.append(q)
                z = latent_sample(q)
                zs.append(z)
                for i in range(n_residual_blocks // 2):
                    gz = tf.concat([gs.pop(), z], axis=-1)
                    h = nn.residual_block(h, gz)
                    hs.append(h)
            else:
                break
            if l + 1 < n_scales:
                n_filters = gs[-1].shape.as_list()[-1]
                h = nn.upsample(h, n_filters)

        return hs, qs, zs  # length: 5,2,2


def classifer(qs, init=False, dropout_p=0.5, activation="elu", num_identities=729):
    with model_arg_scope(
            init=init, dropout_p=dropout_p, activation=activation):
        flatten = tf.compat.v1.layers.flatten(qs[0])
        wfc1 = tf.compat.v1.get_variable(name='FC_W1', shape=[flatten.get_shape()[1], num_identities],
                                         initializer=tf.compat.v1.initializers.glorot_normal())
        bfc1 = tf.compat.v1.get_variable(name='FC_b1', initializer=tf.constant(0.0, shape=[1]))
        logits1 = tf.matmul(flatten, wfc1) + bfc1

        flatten2 = tf.compat.v1.layers.flatten(qs[1])
        wfc2 = tf.compat.v1.get_variable(name='FC_W2', shape=[flatten2.get_shape()[1], num_identities],
                                         initializer=tf.compat.v1.initializers.glorot_normal())
        bfc2 = tf.compat.v1.get_variable(name='FC_b2', initializer=tf.constant(0.0, shape=[1]))
        logits2 = tf.matmul(flatten2, wfc2) + bfc2
    return logits1, logits2


def dec_parameters(
        h, init=False, **kwargs):
    with model_arg_scope(init=init):
        num_filters = 3
        return nn.conv2d(h, num_filters)


def latent_parameters(
        h, init=False, **kwargs):
    num_filters = kwargs.get("num_filters", h.shape.as_list()[-1])
    return nn.conv2d(h, num_filters)


def latent_sample(p):
    mean = p
    stddev = 1.0
    eps = tf.random.normal(tf.shape(mean), mean=0.0, stddev=1.0)
    return mean + stddev * eps


def latent_kl(q, p):
    mean1 = q
    mean2 = p

    kl = 0.5 * tf.square(mean2 - mean1)
    kl = tf.reduce_sum(kl, axis=[1, 2, 3])
    kl = tf.reduce_mean(kl)
    return kl
