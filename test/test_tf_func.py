# -*- coding:utf-8 -*-
from __future__ import absolute_import

import logging
import math
import unittest

import numpy as np
import tensorflow as tf

try:
    from pymltools.tf_utils import init_logger
except ImportError:
    from pymltools.pymltools.tf_utils import init_logger

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
