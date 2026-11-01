"""PIS: conditional VAE training and identity-shift interpolation.

Adapted from the supplied PIS/VUNet implementation. Network scopes are kept
compatible; machine-specific paths and unused experimental modes are removed.
"""
import argparse
import datetime
import json
import logging
from pathlib import Path

import numpy as np
import yaml


class Model:
    def __init__(self, config, session):
        import tensorflow as tf
        import models
        self.tf, self.models = tf, models
        self.config, self.session = config, session
        self.size = int(config['spatial_size'])
        self.box_factor = int(config['box_factor'])
        self.img_shape = [self.size, self.size, 3]
        self.norm_shape = [self.size // 2 ** self.box_factor] * 2 + [24]
        n_scales = 1 + int(np.round(np.log2(self.size))) - int(config['bottleneck_factor'])
        self.enc_up_pass = models.make_model('enc_up', models.enc_up,
            n_scales=n_scales-self.box_factor, n_filters=32 * 2 ** self.box_factor)
        self.enc_down_pass = models.make_model('enc_down', models.enc_down,
            n_scales=n_scales-self.box_factor, n_latent_scales=2)
        self.dec_up_pass = models.make_model('dec_up', models.dec_up,
            n_scales=n_scales, n_filters=32)
        self.dec_down_pass = models.make_model('dec_down', models.dec_down,
            n_scales=n_scales, n_latent_scales=2)
        self.dec_params = models.make_model('dec_params', models.dec_parameters)
        self.classifier = models.make_model('classifer', models.classifer,
            num_identities=int(config.get('num_identities') or 729))

    def encode(self, normalized, condition, **kwargs):
        hidden = self.enc_up_pass(normalized, condition, **kwargs)
        return self.enc_down_pass(hidden, **kwargs)

    def decode(self, condition, latent, **kwargs):
        features = self.dec_up_pass(condition, **kwargs)
        hidden, priors, _ = self.dec_down_pass(features, latent, training=True, **kwargs)
        return self.dec_params(hidden[-1], **kwargs), priors, features

    def build_training(self):
        import nn
        import deeploss
        tf, cfg = self.tf, self.config
        self.vgg = deeploss.VGG19Features(self.session,
            feature_layers=cfg['feature_layers'], feature_weights=cfg['feature_weights'],
            gram_weights=cfg['gram_weights'], weights=cfg.get('vgg_weights', 'imagenet'))
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.lr = nn.make_linear_var(self.global_step, cfg['lr_decay_begin'],
            cfg['lr_decay_end'], cfg['lr'], 0.0, 0.0, cfg['lr'])
        kl_weight = nn.make_linear_var(self.global_step, cfg['lr_decay_end']//2,
            3*cfg['lr_decay_end']//4, 1e-6, 1.0, 1e-6, 1.0)
        batch_size = int(cfg['batch_size'])
        init_size = int(cfg['init_batches']) * batch_size
        self.xn_init = tf.compat.v1.placeholder(tf.float32, [init_size] + self.norm_shape)
        self.cn_init = tf.compat.v1.placeholder(tf.float32, [init_size] + self.norm_shape)
        self.c_init = tf.compat.v1.placeholder(tf.float32, [init_size] + self.img_shape)
        init_kwargs = dict(init=True, dropout_p=cfg['drop_prob'])
        _, _, samples = self.encode(self.xn_init, self.cn_init, **init_kwargs)
        self.data_init = self.decode(self.c_init, samples, **init_kwargs)[0]
        self.x = tf.compat.v1.placeholder(tf.float32, [batch_size] + self.img_shape)
        self.c = tf.compat.v1.placeholder(tf.float32, [batch_size] + self.img_shape)
        self.xn = tf.compat.v1.placeholder(tf.float32, [batch_size] + self.norm_shape)
        self.cn = tf.compat.v1.placeholder(tf.float32, [batch_size] + self.norm_shape)
        self.pid = tf.compat.v1.placeholder(tf.int64, [batch_size])
        kwargs = dict(init=False, dropout_p=cfg['drop_prob'])
        _, posterior_means, posterior_samples = self.encode(self.xn, self.cn, **kwargs)
        self.output, reconstruction_priors, features = self.decode(self.c, posterior_samples, **kwargs)
        reconstruction_loss = 5.0 * self.vgg.make_loss_op(self.x, self.output)
        kl_loss = tf.add_n([self.models.latent_kl(q, p)
                          for q, p in zip(posterior_means, reconstruction_priors)])
        labels = tf.one_hot(self.pid, cfg['num_identities'])
        logits = self.classifier(posterior_means)
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        identity_loss = tf.add_n([cce(labels, score) for score in logits])
        total = reconstruction_loss + kl_weight * kl_loss + cfg.get('identity_weight', 1.0) * identity_loss
        self.losses = {'loss': total, 'reconstruction': reconstruction_loss,
                      'kl': kl_weight * kl_loss, 'identity': identity_loss}
        templates = [self.enc_up_pass, self.enc_down_pass, self.dec_up_pass,
                     self.dec_down_pass, self.dec_params, self.classifier]
        train_variables = [v for template in templates for v in template.trainable_variables]
        optimizer = tf.compat.v1.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9)
        update = optimizer.minimize(total, var_list=train_variables)
        with tf.control_dependencies([update]):
            self.train_op = tf.compat.v1.assign_add(self.global_step, 1)
        self.cycle_enabled = bool(cfg.get('cycle_training', False))
        if self.cycle_enabled:
            alpha = float(cfg.get('cycle_alpha', 0.5))
            self.cycle_targets = [(1-alpha)*q + alpha*q[::-1] for q in posterior_means]
            # Keep these priors separate: CVAE KL above uses reconstruction_priors.
            mixed_hidden, mixed_priors, _ = self.dec_down_pass(
                features, self.cycle_targets, training=True, **kwargs)
            self.cycle_images = self.dec_params(mixed_hidden[-1], **kwargs)
            self.cycle_normalized = tf.compat.v1.placeholder(tf.float32,
                [batch_size] + self.norm_shape, name='cycle_normalized')
            self.cycle_target_inputs = [tf.compat.v1.placeholder(tf.float32, q.shape,
                name='cycle_target_{}'.format(i)) for i, q in enumerate(self.cycle_targets)]
            _, cycle_means, _ = self.encode(self.cycle_normalized, self.cn, **kwargs)
            # Supplied Cycle snapshot uses mean-squared error between unit-variance means.
            self.cycle_loss = tf.add_n([tf.reduce_mean(tf.square(pred-target))
                for pred, target in zip(cycle_means, self.cycle_target_inputs)])
            cycle_variables = list(self.enc_up_pass.trainable_variables) + list(self.enc_down_pass.trainable_variables)
            cycle_optimizer = tf.compat.v1.train.AdamOptimizer(
                cfg.get('cycle_lr_scale', 0.1)*self.lr, beta1=0.5, beta2=0.9)
            self.cycle_op = cycle_optimizer.minimize(
                cfg.get('cycle_weight', 1.0)*self.cycle_loss, var_list=cycle_variables)
        vgg_names = {v.name for v in self.vgg.variables}
        self.variables = [v for v in tf.compat.v1.global_variables() if v.name not in vgg_names]
        self.saver = tf.compat.v1.train.Saver(self.variables, max_to_keep=5)

    def initialize(self, batch):
        tf = self.tf
        self.session.run(tf.compat.v1.variables_initializer(self.variables))
        if self.config.get('vgg_weights', 'imagenet') is None:
            self.session.run(tf.compat.v1.variables_initializer(self.vgg.variables))
        self.session.run(self.data_init, {self.xn_init: batch[2], self.cn_init: batch[3], self.c_init: batch[1]})

    def train_step(self, batch):
        feed = {self.x: batch[0], self.c: batch[1], self.xn: batch[2],
                self.cn: batch[3], self.pid: batch[4]}
        fetches = {'train': self.train_op, 'losses': self.losses, 'output': self.output}
        if self.cycle_enabled:
            if batch[6] is None:
                raise ValueError('Cycle training requires the source image joint coordinates.')
            fetches.update(cycle_images=self.cycle_images, cycle_targets=self.cycle_targets)
        result = self.session.run(fetches, feed)
        losses = result['losses']
        if self.cycle_enabled:
            from batches import normalize, DEFAULT_JOINT_ORDER
            normalized, _ = normalize(result['cycle_images'], batch[6], batch[1],
                self.config.get('joint_order', DEFAULT_JOINT_ORDER), self.box_factor)
            cycle_feed = {self.cycle_normalized: normalized, self.cn: batch[3]}
            cycle_feed.update(zip(self.cycle_target_inputs, result['cycle_targets']))
            _, losses['cycle'] = self.session.run([self.cycle_op, self.cycle_loss], cycle_feed)
        return losses, result['output']

    def build_generation(self):
        tf = self.tf
        self.c = tf.compat.v1.placeholder(tf.float32, [None] + self.img_shape, name='pose')
        self.xn = tf.compat.v1.placeholder(tf.float32, [None] + self.norm_shape, name='appearance')
        self.source_indices = tf.compat.v1.placeholder(tf.int32, [None], name='source_indices')
        self.partner_indices = tf.compat.v1.placeholder(tf.int32, [None], name='partner_indices')
        self.alpha = tf.compat.v1.placeholder(tf.float32, [], name='alpha')
        kwargs = dict(init=False, dropout_p=0.0)
        _, means, _ = self.encode(self.xn, [], **kwargs)
        mixed = [(1-self.alpha)*tf.gather(q, self.source_indices)
                 + self.alpha*tf.gather(q, self.partner_indices) for q in means]
        self.output, _, _ = self.decode(tf.gather(self.c, self.source_indices), mixed, **kwargs)
        self.variables = self.tf.compat.v1.global_variables()
        self.saver = tf.compat.v1.train.Saver(self.variables)

    def generate(self, poses, normalized, sources, partners, alpha):
        if not 0 <= alpha <= 1:
            raise ValueError('alpha must lie in [0, 1].')
        return self.session.run(self.output, {self.c: poses, self.xn: normalized,
            self.source_indices: sources, self.partner_indices: partners, self.alpha: alpha})

    def restore(self, checkpoint, weights_only=False):
        tf = self.tf
        if not Path(str(checkpoint) + '.index').is_file():
            raise FileNotFoundError('Pass a TensorFlow checkpoint prefix with an existing .index file: ' + str(checkpoint))
        if hasattr(self, 'vgg') and self.config.get('vgg_weights', 'imagenet') is None:
            self.session.run(tf.compat.v1.variables_initializer(self.vgg.variables))
        if not weights_only:
            self.saver.restore(self.session, str(checkpoint))
            return
        # New stage: load learned model weights; initialize fresh optimizers/global step.
        self.session.run(tf.compat.v1.variables_initializer(self.variables))
        available = dict(tf.train.list_variables(str(checkpoint)))
        templates = [self.enc_up_pass, self.enc_down_pass, self.dec_up_pass, self.dec_down_pass, self.dec_params]
        required = [v for template in templates for v in template.trainable_variables]
        for v in required:
            if available.get(v.op.name) != v.shape.as_list():
                raise ValueError('Checkpoint does not match PIS network variable: ' + v.op.name)
        compatible = required + [v for v in self.classifier.trainable_variables
                                if available.get(v.op.name) == v.shape.as_list()]
        tf.compat.v1.train.Saver(compatible).restore(self.session, str(checkpoint))

    def save(self, prefix):
        Path(prefix).parent.mkdir(parents=True, exist_ok=True)
        return self.saver.save(self.session, prefix, global_step=self.global_step)


def parse_args():
    parser = argparse.ArgumentParser(description='PIS VAE training and identity-shift generation')
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['train', 'mix', 'recon'], default='train')
    parser.add_argument('--data-index', help='Override pose/image index path')
    parser.add_argument('--match-file', help='JSON source identity -> partner identity mapping')
    parser.add_argument('--checkpoint', help='TensorFlow prefix, without .index/.data suffix')
    parser.add_argument('--log-dir', '--log_dir', dest='log_dir', default='logs')
    parser.add_argument('--retrain', action='store_true', help='Start a new stage from model weights')
    parser.add_argument('--alpha', type=float, help='Interpolation weight for the partner identity')
    parser.add_argument('--split', choices=['train', 'test'], default='train')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def run(args):
    import tensorflow as tf
    from batches import IndexFlow, save_dataset, plot_batch
    tf.compat.v1.disable_v2_behavior()
    with open(args.config, encoding='utf-8') as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError('Configuration must be a YAML mapping.')
    if args.data_index:
        cfg['data_index'] = args.data_index
    if args.match_file:
        cfg['match_path'] = args.match_file
    if args.alpha is not None:
        cfg['alpha'] = args.alpha
    checkpoint = args.checkpoint or cfg.get('checkpoint')
    if (args.mode != 'train' or args.retrain) and not checkpoint:
        raise ValueError('This mode requires --checkpoint with trained PIS weights.')
    if args.retrain and args.mode != 'train':
        raise ValueError('--retrain is only valid in train mode.')
    alpha = float(cfg.get('alpha', 0.5))
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must lie in [0, 1].')
    size, batch_size = int(cfg['spatial_size']), int(cfg['batch_size'])
    if size < 32 or size & (size-1) or cfg['bottleneck_factor'] != 1 or cfg['box_factor'] != 1:
        raise ValueError('Use power-of-two spatial_size >=32 and box_factor=bottleneck_factor=1.')
    np.random.seed(args.seed)
    tf.compat.v1.set_random_seed(args.seed)
    shape = [batch_size, size, size, 3]
    flow = IndexFlow(shape, cfg['data_index'], train=(args.split == 'train'),
        box_factor=cfg['box_factor'], sample='match' if args.mode == 'mix' else 'random',
        shuffle=(args.mode == 'train'), fill_batches=(args.mode == 'train'),
        match_path=cfg.get('match_path'), batch_id_size=cfg.get('identities_per_batch', 27))
    if args.mode == 'train':
        if args.split != 'train':
            raise ValueError('Training must use --split train.')
        expected = cfg.get('num_identities')
        if expected is not None and int(expected) != len(flow.pids):
            raise ValueError('num_identities differs from the valid training identities.')
        cfg['num_identities'] = len(flow.pids)
        cfg['joint_order'] = flow.jo
    out_dir = Path(args.log_dir) / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S-%f')
    out_dir.mkdir(parents=True)
    (out_dir/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(out_dir/'log.txt'), logging.StreamHandler()])
    logging.info('Output directory: %s', out_dir)
    session_config = tf.compat.v1.ConfigProto()
    session_config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=session_config) as session:
        model = Model(cfg, session)
        if args.mode == 'train':
            model.build_training()
            if checkpoint:
                model.restore(checkpoint, weights_only=args.retrain)
            else:
                init_flow = IndexFlow([batch_size*cfg['init_batches'], size, size, 3], cfg['data_index'],
                    train=True, box_factor=cfg['box_factor'])
                model.initialize(next(init_flow))
            (out_dir/'identity_labels.json').write_text(json.dumps(flow.pid2label, indent=2), encoding='utf-8')
            start = int(session.run(model.global_step))
            for _ in range(start, int(cfg['lr_decay_end'])):
                losses, output = model.train_step(next(flow))
                step = int(session.run(model.global_step))
                if not all(np.isfinite(value) for value in losses.values()):
                    raise FloatingPointError('Non-finite training loss: ' + str(losses))
                if step % cfg['log_freq'] == 0 or step == start+1:
                    logging.info('step=%s losses=%s', step, losses)
                    plot_batch(output, str(out_dir/'reconstruction-{}.png'.format(step)))
                if step % cfg['ckpt_freq'] == 0:
                    model.save(str(out_dir/'checkpoints'/'model.ckpt'))
            model.save(str(out_dir/'checkpoints'/'model.ckpt'))
        else:
            model.build_generation()
            model.restore(checkpoint)
            steps = flow.n_groups if args.mode == 'mix' else int(np.ceil(flow.n/batch_size))
            count = 0
            for _ in range(steps):
                batch = next(flow)
                if args.mode == 'mix':
                    poses, normalized, names, counts = batch[1], batch[2], batch[4], batch[5]
                    n_donors = len(counts)
                    sources = np.arange(n_donors, len(names), dtype=np.int32)
                    partners = np.repeat(np.arange(n_donors, dtype=np.int32), counts)
                    names = names[n_donors:]
                else:
                    poses, normalized, names = batch[1], batch[2], batch[5]
                    sources = partners = np.arange(len(names), dtype=np.int32)
                output = model.generate(poses, normalized, sources, partners, alpha if args.mode == 'mix' else 0.0)
                save_dataset(output, out_dir/'images', names,
                    identity_offset=cfg.get('identity_offset', 1500) if args.mode == 'mix' else 0)
                count += len(names)
            logging.info('Saved %s images to %s', count, out_dir/'images')


if __name__ == '__main__':
    run(parse_args())