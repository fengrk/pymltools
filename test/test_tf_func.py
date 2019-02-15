# -*- coding:utf-8 -*-
from __future__ import absolute_import

import logging
import math
import unittest

import numpy as np
import os
import shutil
import tensorflow as tf
import tensorflow.contrib.slim as slim

try:
    from pymltools.tf_utils import init_logger, AbstractEstimator, DatasetUtils, tf_model_fn, OptimizerType
except ImportError:
    from pymltools.pymltools.tf_utils import init_logger, AbstractEstimator, DatasetUtils, tf_model_fn, OptimizerType

init_logger(None)


class TestTFFunc(unittest.TestCase):

    def testNorm(self):
        """
            Test When axis=1
            axis = 0: means normalize each dim by info from this batch
            axis = 1: means normalize x each dim only by x info
        Returns:

        """

        def _get_norm_by_np(arr: np.ndarray) -> np.ndarray:
            _norm = np.linalg.norm(arr, ord=2, axis=test_axis, keepdims=True)
            _std_arr = arr / _norm

            _hand_arr = arr.copy()
            for i in range(arr.shape[0]):
                _norm_float = math.sqrt(sum([j * j for j in arr[i]]))
                _hand_arr[i] = arr[i] / _norm_float
                self.assertTrue(isinstance(_norm_float, float))

            self.assertTrue((abs(_std_arr - _hand_arr) < 1e-5).all())

            return _std_arr

        test_shape = (2, 8)
        self.assertTrue(len(test_shape) == 2)
        test_axis = 1
        test_array_list = [np.random.random(test_shape).astype(np.float32) + 10 - i for i in range(10)]

        embedding = tf.placeholder(dtype=tf.float32, shape=test_shape, name='embedding')
        logging.info("shape of tensor is {}".format(embedding.shape))

        norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), axis=test_axis, keepdims=True))
        normalized_embedding_1 = embedding / norm
        normalized_embedding_2 = tf.nn.l2_normalize(embedding, axis=test_axis)

        with tf.Session() as sess:
            done = False
            for test_arr in test_array_list:
                arr1, arr2 = sess.run([normalized_embedding_1, normalized_embedding_2], feed_dict={embedding: test_arr})
                if not done:
                    logging.info("arr is {}, normalized arr is {}".format(test_arr, arr2))
                    done = True

                arr_std = _get_norm_by_np(test_arr)
                self.assertEqual(arr1.shape, arr2.shape)
                self.assertEqual(arr_std.shape, arr2.shape)
                self.assertTrue((abs(arr1 - arr2) < 1e-5).all())
                self.assertTrue((abs(arr_std - arr2) < 1e-5).all())

    def testMean(self):
        def tf_process(arr_list: list, arr_shape) -> list:
            with tf.Graph().as_default() as graph:
                embedding = tf.placeholder(dtype=tf.float32, shape=arr_shape, name='embedding')
                mean_embedding = tf.reduce_mean(embedding, keepdims=True)
                array_list = []
                with tf.Session(graph=graph) as sess:
                    for arr in arr_list:
                        array_list.append(sess.run(mean_embedding, feed_dict={embedding: arr}))

                return array_list

        test_shape = (384, 384, 1)
        test_array_list = [np.random.random(test_shape).astype(np.float32) + 10 - i for i in range(10)]

        np_mean_list = [np.mean(img, keepdims=True) for img in test_array_list]
        tf_array_list = tf_process(test_array_list, test_shape)

        for index in range(len(test_array_list)):
            self.assertTrue((abs(np_mean_list[index] - tf_array_list[index]) < 1e-3).all())

    def testStd(self):
        def tf_process(arr_list: list, arr_shape) -> list:
            with tf.Graph().as_default() as graph:
                embedding = tf.placeholder(dtype=tf.float32, shape=arr_shape, name='embedding')
                m = tf.reduce_mean(embedding, axis=None, keepdims=True)
                devs_squared = tf.square(embedding - m)
                std_embedding = tf.sqrt(tf.reduce_mean(devs_squared, axis=None, keepdims=True))
                array_list = []
                with tf.Session(graph=graph) as sess:
                    for arr in arr_list:
                        array_list.append(sess.run(std_embedding, feed_dict={embedding: arr}))

                return array_list

        test_shape = (384, 384, 1)
        test_array_list = [np.random.random(test_shape).astype(np.float32) + 10 - i for i in range(10)]

        np_std_list = [np.std(img, keepdims=True) for img in test_array_list]
        tf_std_list = tf_process(test_array_list, test_shape)

        for index in range(len(test_array_list)):
            self.assertTrue((abs(np_std_list[index] - tf_std_list[index]) < 1e-3).all())


class TestTFGrad(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testSimpleGradient(self):
        """
            test tf.gradients
        Returns:

        """
        # simple gradient
        a = tf.constant(0.)
        b = 2 * a
        g = tf.gradients(a + b, [a, b], stop_gradients=[a, b])
        with tf.Session() as sess:
            g_a, g_b = sess.run(g)
            self.assertEqual(g_a, 1.0)
            self.assertEqual(g_b, 1.0)

        # reuse gradient
        h = tf.gradients(a * (a * (a + b) + b) + b, [a, b], stop_gradients=[a, b])
        with tf.Session() as sess:
            h_a, h_b = sess.run(h)
            self.assertEqual(h_a, 0.0)
            self.assertEqual(h_b, 1.0)

    # todo not finish
    def testSimpleReuse(self):
        """
            Warnings: no work as expected!
        Returns:

        """

        class BPDebugEstimator(AbstractEstimator):

            def __init__(self, train_ckpt_dir):
                super(BPDebugEstimator, self).__init__(model_name="BPDebug",
                                                       train_ckpt_dir=train_ckpt_dir,
                                                       pretrained_ckpt_file=None)
                self._key_x = "x"
                self._key_y = "y"
                self.optimizer_type = OptimizerType.adam

            def get_dataset_func(self, split_name, num_epochs=1, shuffle=True, batch_size=64, num_parallel_calls=2,
                                 prefetch_size=2, shuffle_size=4, input_list=None):
                _shuffle = False
                _num_epochs = 1

                if input_list:
                    train_data, train_labels = input_list
                elif split_name == DatasetUtils.SPLIT_TRAIN:
                    train_data, train_labels = self.get_dataset(is_training=True)
                    _shuffle, _num_epochs = True, num_epochs
                else:
                    train_data, train_labels = self.get_dataset(is_training=False)

                return tf.estimator.inputs.numpy_input_fn(
                    x={self._key_x: train_data, self._key_y: train_labels},
                    y=train_labels,
                    batch_size=batch_size,
                    num_epochs=_num_epochs,
                    shuffle=_shuffle,
                    queue_capacity=min(batch_size * 4, 2048 * 16),
                    num_threads=1
                )

            def model_fun(self, ):
                """ 返回func """

                def get_model_net(scope_name, features, params, is_training: bool, labels=None):
                    def fc_func(x, scope, reuse):
                        with tf.variable_scope(scope, 'x', [x], reuse=reuse) as sc:
                            end_points_collection = sc.original_name_scope + '_end_points'
                            with slim.arg_scope([slim.fully_connected], outputs_collections=[end_points_collection]):
                                net = slim.fully_connected(x, num_outputs=1, scope="fc")
                                end_points = slim.utils.convert_collection_to_dict(end_points_collection)

                                return net, end_points

                    inputs = features[self._key_x]
                    y_1, endpoints = fc_func(inputs, scope_name, False)
                    y_2, _ = fc_func(y_1, scope_name, True)
                    y_3, _ = fc_func(y_2, scope_name, True)

                    return y_3, endpoints

                def conv_model(features, labels, mode, params):
                    logits, end_points = get_model_net(self.model_name, features=features, params=None,
                                                       is_training=bool(mode == tf.estimator.ModeKeys.TRAIN))

                    # Compute predictions.
                    if mode == tf.estimator.ModeKeys.PREDICT:
                        predictions = {
                            'logits': logits,
                            "true": features[self._key_y],
                        }
                        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

                    # Compute loss.
                    loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)

                    # Create training op.
                    if mode == tf.estimator.ModeKeys.TRAIN:
                        # 全网络 优化器
                        learning_rate = self.create_learning_rate_op(self.learning_rate)
                        tf.summary.scalar('loss', loss)
                        # fc 优化器
                        fc_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                        fc_train_op = fc_optimizer.minimize(loss,
                                                            global_step=tf.train.get_global_step(),
                                                            var_list=None)

                        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=fc_train_op, scaffold=None)

                    # Compute evaluation metrics.
                    eval_metric_ops = {
                        'accuracy': tf.metrics.mean_squared_error(labels=labels, predictions=logits)
                    }
                    return tf.estimator.EstimatorSpec(
                        mode, loss=loss, eval_metric_ops=eval_metric_ops)

                return conv_model

            def get_dataset(self, is_training):
                def f(x):
                    return x * 0.8 + 0.4

                x_data = np.linspace(-1, 1, 32)[:, np.newaxis]
                y_data = f(f(f(x_data)))
                if is_training:
                    y_data += np.random.normal(loc=0, scale=0.05, size=x_data.shape)

                return x_data, y_data

        train_dir = "./test1/"
        estimator = BPDebugEstimator(train_dir)
        for i in range(10):
            estimator.train(batch_size=8, num_epochs=1)

        estimator.evaluate(batch_size=1, num_epochs=1)

        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
