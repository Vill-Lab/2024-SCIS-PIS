"""Synthetic train/checkpoint/generation checks; no research data or downloads."""
import importlib.util
import logging
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tensorflow as tf

tf.compat.v1.disable_v2_behavior()
from main import Model
from batches import DEFAULT_JOINT_ORDER


class PipelineTest(unittest.TestCase):
    def test_train_save_restore_and_variable_length_generation(self):
        config = dict(spatial_size=32, box_factor=1, bottleneck_factor=1,
                      batch_size=2, init_batches=1, num_identities=2,
                      drop_prob=0.0, lr=1e-4, lr_decay_begin=1,
                      lr_decay_end=10, vgg_weights=None,
                      feature_layers=['input_1', 'block1_conv2'],
                      feature_weights=[1.0, 1.0], gram_weights=[0.0, 0.0],
                      identity_weight=0.0, cycle_training=False)
        rng = np.random.RandomState(2)
        batch = [rng.uniform(-1, 1, (2, 32, 32, 3)).astype('float32'),
                 rng.uniform(-1, 1, (2, 32, 32, 3)).astype('float32'),
                 rng.uniform(-1, 1, (2, 16, 16, 24)).astype('float32'),
                 rng.uniform(-1, 1, (2, 16, 16, 24)).astype('float32'),
                 np.array([0, 1]), ['train/0001_c1.jpg', 'train/0002_c1.jpg'],
                 np.tile(np.array([[.7,.9],[.7,.7],[.65,.5],[.35,.5],[.3,.7],[.3,.9],
                                   [.9,.5],[.8,.35],[.65,.2],[.35,.2],[.2,.35],[.1,.5],
                                   [.5,.18],[.5,.1],[.45,.08],[.55,.08],[.4,.1],[.6,.1]],
                                  dtype='float32')[None]*32, (2,1,1))]
        with tempfile.TemporaryDirectory() as tmp:
            with tf.Graph().as_default(), tf.compat.v1.Session() as session:
                tf.compat.v1.set_random_seed(7)
                model = Model(config, session)
                model.build_training()
                model.initialize(batch)
                losses, output = model.train_step(batch)
                self.assertTrue(all(np.isfinite(v) for v in losses.values()))
                self.assertEqual(output.shape, (2, 32, 32, 3))
                self.assertEqual(session.run(model.global_step), 1)
                tracked = model.enc_up_pass.variables[0]
                tracked_name = tracked.name
                expected = session.run(tracked)
                checkpoint = model.save(str(Path(tmp) / 'model.ckpt'))
            stage_two = dict(config, identity_weight=1.0, cycle_training=True,
                             cycle_weight=1.0, cycle_alpha=0.5, cycle_lr_scale=0.1,
                             joint_order=DEFAULT_JOINT_ORDER)
            with tf.Graph().as_default(), tf.compat.v1.Session() as session:
                model = Model(stage_two, session)
                model.build_training()
                model.restore(checkpoint, weights_only=True)
                self.assertEqual(session.run(model.global_step), 0)
                before_cycle = session.run(model.enc_up_pass.trainable_variables[0])
                losses, _ = model.train_step(batch)
                self.assertIn('cycle', losses)
                self.assertTrue(all(np.isfinite(v) for v in losses.values()))
                self.assertGreater(losses['cycle'], 0)
                after_cycle = session.run(model.enc_up_pass.trainable_variables[0])
                self.assertTrue(np.any(before_cycle != after_cycle))
                expected = session.run(next(v for v in tf.compat.v1.global_variables() if v.name == tracked_name))
                checkpoint = model.save(str(Path(tmp) / 'pis.ckpt'))
            with tf.Graph().as_default(), tf.compat.v1.Session() as session:
                model = Model(dict(config, num_identities=None), session)
                model.build_generation()
                self.assertFalse(any('block1_conv' in v.name for v in tf.compat.v1.global_variables()))
                model.restore(checkpoint)
                restored = next(v for v in tf.compat.v1.global_variables() if v.name == tracked_name)
                np.testing.assert_array_equal(session.run(restored), expected)
                result = model.generate(batch[1], batch[2], [0, 1], [1, 0], 0.5)
                self.assertEqual(result.shape, (2, 32, 32, 3))
                self.assertTrue(np.isfinite(result).all())
                smaller = model.generate(batch[1][:1], batch[2][:1], [0], [0], 0.0)
                self.assertEqual(smaller.shape, (1, 32, 32, 3))
                self.assertTrue(np.isfinite(smaller).all())


if __name__ == '__main__':
    unittest.main()