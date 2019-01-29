# -*- coding:utf-8 -*-
from __future__ import absolute_import

import tensorflow as tf

from pyxtools import byte_to_string, gen_tf_bounding_boxes


def tf_image_crop(image, offset_height, offset_width, target_height, target_width):
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    h_height = h - offset_height
    w_width = w - offset_width

    image = tf.image.crop_to_bounding_box(
        image, offset_height, offset_width, tf.minimum(target_height, h_height), tf.minimum(target_width, w_width))
    return image


def tf_decode_with_crop(file_name, offset_height, offset_width, target_height, target_width):
    image_str = tf.read_file(file_name)
    image = tf.image.decode_jpeg(image_str, channels=3)
    image = tf_image_crop(image, offset_height, offset_width, target_height, target_width)
    return image, file_name


def tf_bounding_box_to_bbox(image, offset_height, offset_width, target_height, target_width):
    # todo not test
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    h_height = h - offset_height
    w_width = w - offset_width

    bbox = tf.convert_to_tensor([
        offset_height / h,
        offset_width / w,
        (tf.minimum(target_height, h_height) + offset_height) / h,
        (tf.minimum(target_width, w_width) + offset_width) / w,
    ], dtype=tf.float32)
    return tf.reshape(bbox, shape=(1, 1, 4))


def parse_bounding_boxes_list(bounding_boxes_list):
    offset_height_list = []
    offset_width_list = []
    target_height_list = []
    target_width_list = []
    for bounding_boxes in bounding_boxes_list:
        offset_height, offset_width, target_height, target_width = gen_tf_bounding_boxes(bounding_boxes)
        offset_height_list.append(offset_height)
        offset_width_list.append(offset_width)
        target_height_list.append(target_height)
        target_width_list.append(target_width)

    return offset_height_list, offset_width_list, target_height_list, target_width_list


def tf_decode_jpg(file_name, ):
    image_str = tf.read_file(file_name)
    image = tf.image.decode_jpeg(image_str, channels=3)
    return image, file_name


def get_tf_image_iterator_from_file_name(image_list: list):
    """ just demo """
    dataset = tf.data.Dataset.from_tensor_slices((image_list,))
    dataset = dataset.map(tf_decode_jpg, num_parallel_calls=2)
    dataset = dataset.batch(8)

    return dataset.make_initializable_iterator()


class SmartImageIterator(object):
    def __init__(self, tf_decoder_func=tf_decode_jpg):
        self._file_list = []
        self._iter = None
        self._tf_decoder_func = tf_decoder_func
        self.batch_size = 8

    def get_tf_image_iterator(self):
        def image_generator():
            for file_name in self._file_list:
                yield (file_name,)

        image_generator.out_type = (tf.string,)
        image_generator.out_shape = (tf.TensorShape([]),)

        if self._iter is None:
            dataset = tf.data.Dataset.from_generator(
                image_generator,
                output_types=image_generator.out_type,
                output_shapes=image_generator.out_shape, )

            dataset = dataset.map(self._tf_decoder_func, num_parallel_calls=2)
            dataset = dataset.batch(self.batch_size)
            self._iter = dataset.make_initializable_iterator()

        return self._iter

    def set_file_list(self, jpg_file_list: list):
        self._file_list.clear()
        self._file_list.extend(jpg_file_list)

    def get_file_list(self) -> list:
        return self._file_list


class ImageIteratorTestDemo(object):
    @classmethod
    def run_image_iterator(cls, image_iterator) -> (list, list):
        images, image_file_names = image_iterator.get_next()

        image_list = []
        file_list = []
        with tf.Session() as sess:
            sess.run(image_iterator.initializer)
            while True:
                try:
                    _image_list, _file_list = sess.run([images, image_file_names])
                    image_list.extend([_image for _image in _image_list])
                    file_list.extend([byte_to_string(file_name) for file_name in _file_list])
                except tf.errors.OutOfRangeError:
                    break

        return image_list, file_list

    @classmethod
    def tf_decode_jpg_with_size(cls, file_name, ):
        image_str = tf.read_file(file_name)
        image = tf.image.decode_jpeg(image_str, channels=3)
        image = tf.image.resize_images(image, size=(224, 224))
        return image, file_name

    @classmethod
    def tf_decode_jpg_with_big_size(cls, file_name, ):
        image_str = tf.read_file(file_name)
        image = tf.image.decode_jpeg(image_str, channels=3)
        image = tf.image.resize_images(image, size=(1000, 1000))
        return image, file_name


__all__ = ("tf_decode_with_crop", "tf_decode_jpg", "get_tf_image_iterator_from_file_name",
           "SmartImageIterator", "ImageIteratorTestDemo", "parse_bounding_boxes_list",
           "tf_bounding_box_to_bbox", "tf_image_crop")
