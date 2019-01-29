# -*- coding:utf-8 -*-
from __future__ import absolute_import

import tensorflow as tf


def int64_feature(values):
    """Returns a TF-Feature of int64s.

    Args:
      values: A scalar or list of values.

    Returns:
      A TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    """Returns a TF-Feature of bytes.

    Args:
      values: A string.

    Returns:
      A TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


class TfRecordUtils(object):
    @classmethod
    def decode(cls, serialized_example, is_training, process_image_fn, image_height, image_width):
        """Parses a single tf.Example into image and label tensors."""
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
                'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
                'image/class/label': tf.FixedLenFeature([], tf.int64),
            })

        img = features['image/encoded']
        # 这里需要对图片进行解码
        img = tf.image.decode_jpeg(img, channels=3)  # 这里，也可以解码为 1 通道
        processed_image = process_image_fn(img, image_height, image_width, is_training=is_training)
        processed_images = tf.expand_dims(processed_image, 0)  # (299,299,3) -> (1, 299,299,3)
        return {"images": processed_images, "filenames": features["image/filename"]}, features['image/class/label']

    @classmethod
    def decode_v2(cls, serialized_example, process_image_fn):
        """Parses a single tf.Example into image and label tensors."""
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
                'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
                'image/class/label': tf.FixedLenFeature([], tf.int64),
            })

        if process_image_fn:
            # decode_jpg/png, resize, crop
            img = process_image_fn(features['image/encoded'])
        else:
            img = features['image/encoded']
        processed_images = tf.expand_dims(img, 0)
        return {"images": processed_images, "filenames": features["image/filename"]}, features['image/class/label']

    @classmethod
    def encode(cls, file_name, class_id):
        """
            编码器
        :param file_name: str, image file name
        :param class_id: int, class id
        :return: example
        """
        image_data = tf.gfile.FastGFile(file_name, 'rb').read()

        example = tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': bytes_feature(image_data),
            'image/class/label': int64_feature(class_id),
            'image/filename': bytes_feature(file_name.encode("utf-8")),
        }))
        return example


__all__ = ("TfRecordUtils",)
