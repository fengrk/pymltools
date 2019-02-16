# -*- coding:utf-8 -*-
from __future__ import absolute_import

import logging
import unittest

import numpy as np
import os
import tensorflow as tf
from PIL import Image

try:
    from pymltools.tf_utils import siamese_net_preprocess_image_np, whale_siamese_preprocess_tf, init_logger
except ImportError:
    from pymltools.pymltools.tf_utils import siamese_net_preprocess_image_np, whale_siamese_preprocess_tf, init_logger

from pymltools.test.utils import list_jpg_file

init_logger(None)


class TestSiameseNetProcess(unittest.TestCase):
    def setUp(self):
        self.jpg_list = []
        raw_list = list_jpg_file()

        for jpg in raw_list:
            img = Image.open(jpg)
            img = img.resize((384, 385))
            img.save(jpg + ".x.jpg")
            self.jpg_list.append(jpg + ".x.jpg")

    def tearDown(self):
        raw_list = list_jpg_file()
        for jpg in self.jpg_list:
            if os.path.exists(jpg) and jpg not in raw_list:
                os.remove(jpg)

    def testPreprocess(self):
        """
            Todo not Pass, 有微小差距
        Returns:

        """

        def tf_process(jpg_file_list: list) -> list:
            with tf.Graph().as_default() as graph:
                file_input = tf.placeholder(tf.string, ())
                image = tf.image.decode_jpeg(tf.read_file(file_input), channels=1)
                image = whale_siamese_preprocess_tf(image, None, None)

                array_list = []
                with tf.Session(graph=graph) as sess:
                    for jpg_file in jpg_file_list:
                        array_list.append(sess.run(image, feed_dict={file_input: jpg_file}))

                return array_list

        image_file_list = list(self.jpg_list)

        tf_array_list = tf_process(image_file_list)
        keras_array_list = [siamese_net_preprocess_image_np(f, size=None) for f in image_file_list]
        self.assertEqual(tf_array_list[0].shape, keras_array_list[0].shape)

        logging.info("sum of tf array is {}".format([np.sum(arr, ) for arr in tf_array_list]))
        logging.info("sum of keras array is {}".format([np.sum(arr, ) for arr in keras_array_list]))

        logging.info("mean of tf array is {}".format([np.mean(arr) for arr in tf_array_list]))
        logging.info("mean of keras array is {}".format([np.mean(arr) for arr in keras_array_list]))

        logging.info("tf array is {}".format([arr[:5, 0, 0] for arr in tf_array_list]))
        logging.info("keras array is {}".format([arr[:5, 0, 0] for arr in keras_array_list]))

        for index in range(len(image_file_list)):
            self.assertTrue((abs(keras_array_list[index] - tf_array_list[index]) < 1e-3).all())
