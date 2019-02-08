# -*- coding:utf-8 -*-
from __future__ import absolute_import

import unittest

import os
import random
import tensorflow as tf

from pyxtools import list_files

try:
    from pymltools.tf_utils import SmartImageIterator, ImageIteratorTestDemo
except ImportError:
    from pymltools.pymltools.tf_utils import SmartImageIterator, ImageIteratorTestDemo


class TestSmartImageIterator(unittest.TestCase):
    def setUp(self):
        self.all_image_list = [
            img_file for img_file in list_files(os.path.dirname(__file__)) if img_file.endswith(".jpg")]

    def tearDown(self):
        pass

    def test_iter_once(self):
        smart_iter = SmartImageIterator(tf_decoder_func=ImageIteratorTestDemo.tf_decode_jpg_with_size)
        self._iter(smart_iter, self.all_image_list)

    def _iter(self, smart_iter: SmartImageIterator, image_list: list):
        print("parsing image: {}".format(image_list))
        smart_iter.set_file_list(jpg_file_list=image_list)
        image_iterator = smart_iter.get_tf_image_iterator()
        image_array_list, image_file_list = ImageIteratorTestDemo.run_image_iterator(image_iterator=image_iterator)
        print("parsed image: {}".format(image_file_list))

        self.assertEqual(len(image_list), len(image_file_list))
        self.assertEqual(len(image_list), len(image_array_list))
        self.assertEqual(";;".join(image_list), ";;".join(image_file_list))

    def test_iter_multi_times(self):
        all_image_list = list(self.all_image_list) * 3
        smart_iter = SmartImageIterator(tf_decoder_func=ImageIteratorTestDemo.tf_decode_jpg_with_size)
        for i in range(len(all_image_list)):
            count = i + 1
            print("count is {}".format(count))
            new_file_list = random.choices(all_image_list, k=count)
            self.assertTrue(len(new_file_list) == count)
            self._iter(smart_iter, new_file_list)


class TestDataset(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testIter(self):
        def tf_decode_jpg(x, ):
            return x

        image_list = [i for i in range(100)]
        dataset = tf.data.Dataset.from_tensor_slices((image_list,))
        dataset = dataset.map(tf_decode_jpg, num_parallel_calls=2)
        dataset = dataset.batch(8)

        image_iterator = dataset.make_initializable_iterator()

        images = image_iterator.get_next()

        file_list = []
        with tf.Session() as sess:
            sess.run(image_iterator.initializer)
            while True:
                try:
                    _image_list = sess.run([images])
                    file_list.extend([int(_image) for _image in _image_list[0]])
                except tf.errors.OutOfRangeError:
                    break

        print(image_list)
        print(file_list)
        self.assertEqual(";;".join([str(x) for x in file_list]), ";;".join(str(x) for x in image_list))
