# -*- coding:utf-8 -*-
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops.image_ops_impl import ResizeMethod

# vgg setting
_VGG_R_MEAN = 123.68
_VGG_G_MEAN = 116.78
_VGG_B_MEAN = 103.94

_VGG_RESIZE_SIDE_MIN = 256
_VGG_RESIZE_SIDE_MAX = 512


def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].

    Args:
      x: input Tensor.
      func: Python function to apply.
      num_cases: Python int32, number of cases to sample sel from.

    Returns:
      The result of func(x, sel), where func receives the value of the
      selector as a python integer, but sel is sampled dynamically.
    """
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
        func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
        for case in range(num_cases)])[0]


def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
    """Generates cropped_image using a one of the bboxes randomly distorted.

    See `tf.image.sample_distorted_bounding_box` for more documentation.

    Args:
      image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged
        as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
        image.
      min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding box
        supplied.
      aspect_ratio_range: An optional list of `floats`. The cropped area of the
        image must have an aspect ratio = width / height within this range.
      area_range: An optional list of `floats`. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
      max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.
      scope: Optional scope for name_scope.
    Returns:
      A tuple, a 3-D Tensor cropped_image and the distorted bbox
    """
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].

        # A large fraction of image datasets contain a human-annotated bounding
        # box delineating the region of the image containing the object of interest.
        # We choose to create a new bounding box for the object which is a randomly
        # distorted version of the human-annotated bounding box that obeys an
        # allowed range of aspect ratios, sizes and overlap with the human-annotated
        # bounding box. If no box is supplied, then we assume the bounding box is
        # the entire image.
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        return cropped_image, distort_bbox


def _whale_gray_preprocess_for_train(image, height, width, bbox,
                                     fast_mode=True,
                                     scope=None,
                                     add_image_summaries=True, mean_tf_func=None):
    """Distort one image for training a network.

    Distorting images provides a useful technique for augmenting the data
    set during training in order to make the network invariant to aspects
    of the image that do not effect the label.

    Additionally it would create image_summaries to display the different
    transformations applied to the image.

    Args:
      mean_tf_func (func):
      image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
        [0, 1], otherwise it would converted to tf.float32 assuming that the range
        is [0, MAX], where MAX is largest positive representable number for
        int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
      height: integer
      width: integer
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged
        as [ymin, xmin, ymax, xmax].
      fast_mode: Optional boolean, if True avoids slower transformations (i.e.
        bi-cubic resizing, random_hue or random_contrast).
      scope: Optional scope for name_scope.
      add_image_summaries: Enable image summaries.
    Returns:
      3-D float Tensor of distorted image used for training with range [-1, 1].
    """
    with tf.name_scope(scope, 'distort_image', [image, height, width, bbox]):
        if bbox is None:
            bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                               dtype=tf.float32,
                               shape=[1, 1, 4])
        if image.shape[-1] == 3:
            image = tf.image.rgb_to_grayscale(image)
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].
        image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), bbox)
        if add_image_summaries:
            tf.summary.image('image_with_bounding_boxes', image_with_box)

        # 1.random crop
        distorted_image, distorted_bbox = distorted_bounding_box_crop(
            image,
            bbox,
            min_object_covered=0.90,
            aspect_ratio_range=(0.75, 1.33),
            area_range=(0.90, 1.0), )

        # Restore the shape since the dynamic slice based upon the bbox_size loses
        # the third dimension.
        distorted_image.set_shape([None, None, 1])
        image_with_distorted_box = tf.image.draw_bounding_boxes(
            tf.expand_dims(image, 0), distorted_bbox)
        if add_image_summaries:
            tf.summary.image('images_with_distorted_bounding_box',
                             image_with_distorted_box)

        # This resizing operation may distort the images because the aspect
        # ratio is not respected. We select a resize method in a round robin
        # fashion based on the thread number.
        # Note that ResizeMethod contains 4 enumerated resizing methods.

        # We select only 1 case for fast_mode bilinear.
        # todo siamese use nearest
        num_resize_cases = 1 if fast_mode else 4
        distorted_image = apply_with_random_selector(
            distorted_image,
            lambda x, method: tf.image.resize_images(x, [height, width], method),
            num_cases=num_resize_cases)

        if add_image_summaries:
            tf.summary.image('cropped_resized_image',
                             tf.expand_dims(distorted_image, 0))

        # 2.Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # 3.旋转
        # distorted_image = tf.contrib.image.rotate(
        #     distorted_image,
        #     angles=tf.random.uniform(shape=(1,), minval=-0.18, maxval=0.18, )[0]
        # )

        # 4.brightness
        distorted_image = tf.image.random_brightness(distorted_image, max_delta=30)

        # 5.mean
        if mean_tf_func is None:
            mean_tf_func = tf_image_mean_inception

        distorted_image = mean_tf_func(distorted_image)

        # todo 1.rotate
        # todo 2.shift

        if add_image_summaries:
            tf.summary.image('final_distorted_image',
                             tf.expand_dims(distorted_image, 0))
        return distorted_image


def _whale_gray_preprocess_for_eval(image, height, width, bbox=None, scope=None, mean_tf_func=None):
    """Prepare one image for evaluation.

    If height and width are specified it would output an image with that size by
    applying resize_bilinear.

    If central_fraction is specified it would crop the central fraction of the
    input image.

    Args:
      mean_tf_func (func):
      image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
        [0, 1], otherwise it would converted to tf.float32 assuming that the range
        is [0, MAX], where MAX is largest positive representable number for
        int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
      height: integer
      width: integer
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged
        as [ymin, xmin, ymax, xmax].
      scope: Optional scope for name_scope.
    Returns:
      3-D float Tensor of prepared image.
    """
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        if image.shape[-1] == 3:
            image = tf.image.rgb_to_grayscale(image)
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        if bbox is not None:
            image, distorted_bbox = distorted_bounding_box_crop(
                image, bbox,
                min_object_covered=1.0,
                aspect_ratio_range=(0.75, 1.33),
                area_range=(0.99, 1.0),
            )
            image.set_shape([None, None, 1])

        if height and width:
            # Resize the image to the specified height and width.
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [height, width],
                                             align_corners=False)  # todo siamese use nearest
            image = tf.squeeze(image, [0])

        # 5.mean
        if mean_tf_func is None:
            mean_tf_func = tf_image_mean_inception

        image = mean_tf_func(image)

        return image


def _whale_rgb_preprocess_for_train(image, height, width, bbox, fast_mode=True, easy_train: bool = False,
                                    scope=None, add_image_summaries=True, mean_tf_func=None):
    """Distort one image for training a network.

    Distorting images provides a useful technique for augmenting the data
    set during training in order to make the network invariant to aspects
    of the image that do not effect the label.

    Additionally it would create image_summaries to display the different
    transformations applied to the image.

    Args:
      mean_tf_func (func):
      image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
        [0, 1], otherwise it would converted to tf.float32 assuming that the range
        is [0, MAX], where MAX is largest positive representable number for
        int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
      height: integer
      width: integer
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged
        as [ymin, xmin, ymax, xmax].
      fast_mode: Optional boolean, if True avoids slower transformations (i.e.
        bi-cubic resizing, random_hue or random_contrast).
      scope: Optional scope for name_scope.
      add_image_summaries: Enable image summaries.
    Returns:
      3-D float Tensor of distorted image used for training with range [-1, 1].
    """

    with tf.name_scope(scope, 'distort_image', [image, height, width, bbox]):
        if bbox is None:
            bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].
        image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), bbox)
        if add_image_summaries:
            tf.summary.image('image_with_bounding_boxes', image_with_box)

        if easy_train:
            rotate_angle, min_cover = 0.01, 0.90
        else:
            rotate_angle, min_cover = 0.18, 0.30

        # 1. random rotate
        image = tf.contrib.image.rotate(
            image,
            angles=tf.random.uniform(shape=(1,), minval=-rotate_angle, maxval=rotate_angle, )[0],  # -10 ~ 10
            interpolation="BILINEAR",
        )

        # 2. random crop
        distorted_image, distorted_bbox = distorted_bounding_box_crop(
            image,
            bbox,
            min_object_covered=min_cover,
            aspect_ratio_range=(0.75, 1.33),
            area_range=(min_cover, 1.0), )

        # Restore the shape since the dynamic slice based upon the bbox_size loses
        # the third dimension.
        distorted_image.set_shape([None, None, 3])
        image_with_distorted_box = tf.image.draw_bounding_boxes(
            tf.expand_dims(image, 0), distorted_bbox)
        if add_image_summaries:
            tf.summary.image('images_with_distorted_bounding_box', image_with_distorted_box)

        # This resizing operation may distort the images because the aspect
        # ratio is not respected. We select a resize method in a round robin
        # fashion based on the thread number.
        # Note that ResizeMethod contains 4 enumerated resizing methods.

        # We select only 1 case for fast_mode bilinear.
        # todo siamese use nearest
        num_resize_cases = 1 if fast_mode else 4
        distorted_image = apply_with_random_selector(
            distorted_image,
            lambda x, method: tf.image.resize_images(x, [height, width], method),
            num_cases=num_resize_cases)

        if add_image_summaries:
            tf.summary.image('cropped_resized_image', tf.expand_dims(distorted_image, 0))

        # 3.Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # 4.brightness
        distorted_image = tf.image.random_brightness(distorted_image, max_delta=30)

        # 5.mean
        if mean_tf_func is None:
            mean_tf_func = tf_image_mean_inception

        distorted_image = mean_tf_func(distorted_image)

        if add_image_summaries:
            tf.summary.image('final_distorted_image', tf.expand_dims(distorted_image, 0))

        return distorted_image


def _whale_rgb_preprocess_for_eval(image, height, width, bbox=None, scope=None, mean_tf_func=None):
    """Prepare one image for evaluation.

    If height and width are specified it would output an image with that size by
    applying resize_bilinear.

    If central_fraction is specified it would crop the central fraction of the
    input image.

    Args:
      mean_tf_func (func):
      image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
        [0, 1], otherwise it would converted to tf.float32 assuming that the range
        is [0, MAX], where MAX is largest positive representable number for
        int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
      height: integer
      width: integer
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged
        as [ymin, xmin, ymax, xmax].
      scope: Optional scope for name_scope.
    Returns:
      3-D float Tensor of prepared image.
    """

    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        if bbox is not None:
            image, distorted_bbox = distorted_bounding_box_crop(
                image, bbox,
                min_object_covered=1.0,
                aspect_ratio_range=(0.75, 1.33),
                area_range=(0.99, 1.0),
            )
            image.set_shape([None, None, 3])

        if height and width:
            # Resize the image to the specified height and width.
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [height, width], align_corners=False)  # todo siamese use nearest
            image = tf.squeeze(image, [0])

        # 5.mean
        if mean_tf_func is None:
            mean_tf_func = tf_image_mean_inception

        image = mean_tf_func(image)

        return image


def tf_image_mean_inception(image, ):
    """
        Test Pass, equal to `whale_siamese_image_mean_np`
    Args:
        image:

    Returns:

    """
    # rescale the image to [-1,1] from [0,1]
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def tf_image_mean_vgg(image, ):
    """
    Args:
        image:

    Returns:

    """
    means = [_VGG_R_MEAN, _VGG_G_MEAN, _VGG_B_MEAN]

    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')

    num_channels = image.get_shape().as_list()[-1]

    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)


def whale_siamese_image_mean_np(img: np.ndarray) -> np.ndarray:
    """
        Test Pass, equal to `whale_siamese_image_mean_tf`
    Args:
        img:

    Returns:

    """
    img -= np.mean(img, keepdims=True)
    img /= np.std(img, keepdims=True) + K.epsilon()
    return img


def whale_siamese_image_mean_tf(image, ):
    """
        Test Pass, equal to `whale_siamese_image_mean_np`
    Args:
        image:

    Returns:

    """
    # image = tf.image.per_image_standardization(image)
    image = tf.subtract(image, tf.reduce_mean(image, keepdims=True))

    dev_squared = tf.square(image - tf.reduce_mean(image, keepdims=True))
    std_image = tf.sqrt(tf.reduce_mean(dev_squared, keepdims=True))

    return tf.divide(image, std_image + K.epsilon())


def whale_gray_preprocess_image(image, height, width,
                                is_training=False,
                                bbox=None,
                                fast_mode=True,
                                add_image_summaries=True, mean_tf_func=None):
    """Pre-process one image for training or evaluation.

    Args:
      mean_tf_func:
      image: 3-D Tensor [height, width, channels] with the image. If dtype is
        tf.float32 then the range should be [0, 1], otherwise it would converted
        to tf.float32 assuming that the range is [0, MAX], where MAX is largest
        positive representable number for int(8/16/32) data type (see
        `tf.image.convert_image_dtype` for details).
      height: integer, image expected height.
      width: integer, image expected width.
      is_training: Boolean. If true it would transform an image for train,
        otherwise it would transform it for evaluation.
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as
        [ymin, xmin, ymax, xmax].
      fast_mode: Optional boolean, if True avoids slower transformations.
      add_image_summaries: Enable image summaries.

    Returns:
      3-D float Tensor containing an appropriately scaled image

    Raises:
      ValueError: if user does not provide bounding box
    """
    if is_training:
        return _whale_gray_preprocess_for_train(
            image, height, width, bbox, fast_mode,
            add_image_summaries=add_image_summaries, mean_tf_func=mean_tf_func)
    else:
        return _whale_gray_preprocess_for_eval(image, height, width, bbox, mean_tf_func=mean_tf_func)


def whale_rgb_preprocess_image(image, height, width, is_training=False, bbox=None, easy_train: bool = True,
                               fast_mode=True, add_image_summaries=True, mean_tf_func=tf_image_mean_vgg):
    """Pre-process one image for training or evaluation.

    Args:
      mean_tf_func:
      image: 3-D Tensor [height, width, channels] with the image. If dtype is
        tf.float32 then the range should be [0, 1], otherwise it would converted
        to tf.float32 assuming that the range is [0, MAX], where MAX is largest
        positive representable number for int(8/16/32) data type (see
        `tf.image.convert_image_dtype` for details).
      height: integer, image expected height.
      width: integer, image expected width.
      is_training: Boolean. If true it would transform an image for train,
        otherwise it would transform it for evaluation.
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as
        [ymin, xmin, ymax, xmax].
      fast_mode: Optional boolean, if True avoids slower transformations.
      add_image_summaries: Enable image summaries.

    Returns:
      3-D float Tensor containing an appropriately scaled image

    Raises:
      ValueError: if user does not provide bounding box
    """
    if is_training:
        return _whale_rgb_preprocess_for_train(
            tf_image_random_gray(image, gray_prob=0.2), height, width, bbox, fast_mode,
            add_image_summaries=add_image_summaries, mean_tf_func=mean_tf_func, easy_train=easy_train)
    else:
        return _whale_rgb_preprocess_for_eval(image, height, width, bbox, mean_tf_func=mean_tf_func)


def siamese_net_preprocess_image_np(image_file: str, size=(384, 384)) -> np.ndarray:
    img = Image.open(image_file).convert("L")
    if size:
        img = img.resize(size)
    img = img_to_array(img)
    return whale_siamese_image_mean_np(img)


def whale_siamese_preprocess_tf(image, height, width, scope=None):
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        if image.shape[-1] == 3:
            image = tf.image.rgb_to_grayscale(image)
        if image.dtype != tf.float32:
            # dtype_max = image.dtype.max
            # image = tf.multiply(tf.image.convert_image_dtype(image, dtype=tf.float32), dtype_max)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        if height and width:
            # Resize the image to the specified height and width.
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_images(image, [height, width],
                                           method=ResizeMethod.NEAREST_NEIGHBOR)
            image = tf.squeeze(image, [0])

        return whale_siamese_image_mean_tf(image)


def tf_image_random_gray(image, gray_prob: float = 0.2):
    def gray():
        return tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))

    def rgb():
        return image

    # Uniform variable in [0,1)
    p_order = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    return tf.cond(tf.less(p_order, gray_prob), true_fn=gray, false_fn=rgb)


__all__ = ("whale_gray_preprocess_image", "siamese_net_preprocess_image_np", "whale_siamese_preprocess_tf",
           "whale_siamese_image_mean_np", "whale_siamese_image_mean_tf", "tf_image_random_gray",
           "whale_rgb_preprocess_image", "tf_image_mean_vgg")
