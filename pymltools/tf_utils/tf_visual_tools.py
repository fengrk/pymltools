# -*- coding:utf-8 -*-
from __future__ import absolute_import

import logging
import pathlib
import queue
import threading

import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import tempfile
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from .basic_tools import get_mini_train_set
from pyxtools import show_hot_graph, get_image, image_to_array, np_image_to_pil_image, create_guid, get_md5, \
    get_pretty_float


class PropVisualTools(object):
    def __init__(self, image_file_path, sess, file_input, prob_logit, label_map: dict, inner_layer=None,
                 square_shape: tuple = None, square_color=np.asarray([127, 127, 127], np.uint8),
                 one_of_count: int = 10):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.label_map = label_map
        self.prob_logit = prob_logit
        self.inner_layer = inner_layer
        self.file_input = file_input
        self.sess = sess
        self.square_color = square_color
        self.square_shape = square_shape
        self.image_file_path = image_file_path
        self.predict_result = None
        self.xyz_list = ([], [], [])
        self.one_of_ = one_of_count
        self.inner_result = None

    def raw_prob(self) -> float:
        if self.inner_layer is None:
            x = self.prob_logit.eval(feed_dict={self.file_input: self.image_file_path})
        else:
            inner_result, x = self.sess.run((self.inner_layer, self.prob_logit),
                                            feed_dict={self.file_input: self.image_file_path})
            self.inner_result = {
                "raw_output": inner_result
            }
        self.predict_result = {
            "class": int(x.argmax()),
            "prob": float(x.max()),
            "class_name": self.label_map[x.argmax()]
        }
        self.logger.info("result of {} is: {}(prob: {})".format(
            self.image_file_path,
            self.predict_result["class_name"],
            self.predict_result["prob"])
        )
        return self.predict_result["prob"]

    def _predict_prob(self, image_file, ) -> float:
        return float(self.prob_logit.eval(feed_dict={self.file_input: image_file})[0][self.predict_result["class"]])

    def _predict_prob_with_inner_layer(self, inner_output, ) -> float:
        """ inner_output: 1x7x7x320 """
        return float(self.prob_logit.eval(feed_dict={self.inner_layer: inner_output})[0][self.predict_result["class"]])

    def _run_expr_without_inner_layer(self, tmp_dir):
        pipeline = queue.Queue()

        def mask_image(_image_array, _points_list, _square_value):
            blank_image_array = _image_array.copy()
            for point in _points_list:
                blank_image_array[point[0]][point[1]][:] = _square_value
            return np_image_to_pil_image(blank_image_array)

        def prepare_image(result_queue, image_file_path, square_shape, square_color):
            image = get_image(image_file_path)
            image_array = image_to_array(image)  # shape 0 y, height; shape 1 x, width
            if square_shape is None:
                square_shape = (int(image_array.shape[1] / self.one_of_), int(image_array.shape[0] / self.one_of_))
            x_length = int(image_array.shape[1] / square_shape[0])  # shape 0 y, height; shape 1 x, width
            y_length = int(image_array.shape[0] / square_shape[1])

            for y_p in range(y_length):
                for x_p in range(x_length):
                    _points_list = []
                    if y_p == y_length - 1:
                        y_end = image_array.shape[0]
                    else:
                        y_end = y_p * square_shape[1] + square_shape[1]
                    if x_p == x_length - 1:
                        x_end = image_array.shape[1]
                    else:
                        x_end = x_p * square_shape[0] + square_shape[0]

                    for y_y in range(y_p * square_shape[1], y_end):
                        for x_x in range(x_p * square_shape[0], x_end):
                            _points_list.append([y_y, x_x])  # shape 0 y, height; shape 1 x, width

                    new_image = mask_image(image_array, _points_list, square_color)
                    new_image_file = os.path.join(
                        tmp_dir,
                        "{}.jpg".format(create_guid())
                    )
                    new_image.save(new_image_file)
                    result_queue.put((new_image_file, _points_list))

            result_queue.put(None)

        def list_image_file():
            while True:
                content = pipeline.get()
                if content is None:
                    break
                yield content

        _thread = threading.Thread(target=prepare_image,
                                   args=(pipeline, self.image_file_path, self.square_shape, self.square_color))
        _thread.daemon = True
        _thread.start()

        for image_file, points_list in list_image_file():
            prob = self._predict_prob(image_file)
            for (x, y) in points_list:
                # shape 0 y, height; shape 1 x, width
                self.xyz_list[0].append(y)
                self.xyz_list[1].append(x)
                self.xyz_list[2].append(prob)
            os.remove(image_file)

        try:
            os.removedirs(tmp_dir)
        except Exception as e:
            self.logger.error(e)

    def _run_expr_with_inner_layer(self, tmp_dir):
        pipeline = queue.Queue()

        def prepare_image(result_queue, image_file_path, square_shape, square_color):
            raw_inner_output = np.squeeze(self.inner_result["raw_output"], axis=0)  # 7x7x320
            image = get_image(image_file_path)
            image_array = image_to_array(image)  # shape 0 y, height; shape 1 x, width
            image_array_shape = image_array.shape
            blank_value = 0.0

            x_length = image_array_shape[1] // raw_inner_output.shape[1]  # shape 0 y, height; shape 1 x, width
            y_length = image_array_shape[0] // raw_inner_output.shape[0]

            for y_p in range(raw_inner_output.shape[1]):
                for x_p in range(raw_inner_output.shape[0]):
                    _points_list = []
                    if y_p == raw_inner_output.shape[1] - 1:
                        y_end = image_array_shape[0]
                    else:
                        y_end = y_p * y_length + y_length
                    if x_p == raw_inner_output.shape[0] - 1:
                        x_end = image_array_shape[1]
                    else:
                        x_end = x_p * x_length + x_length

                    for y_y in range(y_p * y_length, y_end):
                        for x_x in range(x_p * x_length, x_end):
                            _points_list.append([y_y, x_x])  # shape 0 y, height; shape 1 x, width

                    new_inner_output = raw_inner_output.copy()
                    new_inner_output[x_p][y_p][:] = blank_value
                    result_queue.put((
                        new_inner_output.reshape(
                            (1,
                             new_inner_output.shape[0],
                             new_inner_output.shape[1],
                             new_inner_output.shape[2])),
                        _points_list))

            result_queue.put(None)

        def list_inner_output():
            while True:
                content = pipeline.get()
                if content is None:
                    break
                yield content

        _thread = threading.Thread(target=prepare_image,
                                   args=(pipeline, self.image_file_path, self.square_shape, self.square_color))
        _thread.daemon = True
        _thread.start()

        for inner_output, points_list in list_inner_output():
            prob = self._predict_prob_with_inner_layer(inner_output)
            for (x, y) in points_list:
                # shape 0 y, height; shape 1 x, width
                self.xyz_list[0].append(y)
                self.xyz_list[1].append(x)
                self.xyz_list[2].append(prob)

    def run_expr(self):
        if self.predict_result is None:
            self.raw_prob()

        tmp_dir = os.path.join(tempfile.gettempdir(), get_md5(self.image_file_path.encode("utf-8")))
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)

        if self.inner_layer is None:
            self._run_expr_without_inner_layer(tmp_dir)
        else:
            # self._run_expr_with_inner_layer(tmp_dir)
            #  todo 错误方法, 唯一收获是实现inner_layer充当input_tensor
            self._run_expr_without_inner_layer(tmp_dir)

    def create_hot_graph(self, save_image_file: str):
        if self.xyz_list and self.xyz_list[0]:
            show_hot_graph(xyz_list=self.xyz_list, image_save_file=save_image_file)
            self.logger.info("export hot graph to {}".format(save_image_file))
        else:
            raise ValueError("xyz_list is invalid!")

    def _get_inner_output_with_mask(self, tmp_dir):
        assert self.inner_layer is not None
        image = get_image(self.image_file_path)
        image_array = image_to_array(image).copy()  # shape 0 y, height; shape 1 x, width
        for x in range(image_array.shape[0]):
            for y in range(image_array.shape[1]):
                image_array[x][y][:] = self.square_color
        new_image = np_image_to_pil_image(image_array)
        new_image_file = os.path.join(
            tmp_dir,
            "{}.jpg".format(create_guid())
        )
        new_image.save(new_image_file)

        return self.inner_layer.eval(feed_dict={self.file_input: new_image_file})

    def debug(self):
        if self.predict_result is None:
            self.raw_prob()

        self.logger.info("prob is {}".format(
            self._predict_prob_with_inner_layer(self.inner_result["raw_output"]))
        )

        # todo 测试后发现, inner_output 与 原始图像不具有唯一对应性
        tmp_dir = os.path.join(tempfile.gettempdir(), get_md5(self.image_file_path.encode("utf-8")))
        inner_output = np.squeeze(self._get_inner_output_with_mask(tmp_dir), axis=0)
        for x in range(inner_output.shape[0]):
            for y in range(inner_output.shape[1]):
                self.logger.info("first5 is {}, last5 is {}".format(inner_output[x][y][:5], inner_output[x][y][-5:]))
                # todo 每个位置的数据都不一样


class EmbeddingVisualTools(object):
    """
        可视化向量空间
    """
    tmp_dir = "/tmp/tensorboard"

    def __init__(self, ):
        pass

    @classmethod
    def prepare_dir(cls):
        if os.path.exists(cls.tmp_dir):
            shutil.rmtree(cls.tmp_dir)

    @classmethod
    def show_embeddings(cls, embeddings: np.ndarray, labels: list, sprite_file_name: str = None,
                        sprite_size: list = None, tmp_dir: str = None):
        if tmp_dir is None:
            tmp_dir = cls.tmp_dir

        tf.logging.info("Embeddings shape: {}".format(embeddings.shape))

        # Visualize test embeddings
        with tf.Graph().as_default() as graph:
            embedding_var = tf.Variable(embeddings, name='embedding')

            summary_writer = tf.summary.FileWriter(tmp_dir)

            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name

            # Specify where you find the sprite (we will create this later)
            # Copy the embedding sprite image to the eval directory
            if sprite_file_name:
                if not sprite_size:
                    sprite_size = [2, 2]
                shutil.copy2(sprite_file_name, tmp_dir)
                embedding.sprite.image_path = pathlib.Path(sprite_file_name).name
                embedding.sprite.single_image_dim.extend(sprite_size)

            # Specify where you find the metadata
            # Save the metadata file needed for Tensorboard projector
            metadata_filename = "metadata.tsv"
            with open(os.path.join(tmp_dir, metadata_filename), 'w') as f:
                for i in range(len(labels)):
                    f.write('{}\n'.format(labels[i]))
            embedding.metadata_path = metadata_filename

            # Say that you want to visualise the embeddings
            projector.visualize_embeddings(summary_writer, config)

            saver = tf.train.Saver()
            with tf.Session(graph=graph) as sess:
                sess.run(embedding_var.initializer)
                saver.save(sess, os.path.join(tmp_dir, "embeddings.ckpt"))

            tf.logging.info("save log in {}".format(tmp_dir))

    @classmethod
    def show_mnist_embeddings(cls):
        """
            todo with error
        :return:
        """

        def create_sprite_image(images):
            """
                Returns a sprite image consisting of images passed as argument.
                Images should be count x width x height
            """
            if isinstance(images, list):
                images = np.array(images)
            img_h = images.shape[1]
            img_w = images.shape[2]
            n_plots = int(np.ceil(np.sqrt(images.shape[0])))

            spriteimage = np.ones((img_h * n_plots, img_w * n_plots))

            for i in range(n_plots):
                for j in range(n_plots):
                    this_filter = i * n_plots + j
                    if this_filter < images.shape[0]:
                        this_img = images[this_filter]
                        spriteimage[i * img_h:(i + 1) * img_h, j * img_w:(j + 1) * img_w] = this_img

            return spriteimage

        def vector_to_matrix_mnist(mnist_digits):
            """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
            return np.reshape(mnist_digits, (-1, 28, 28))

        def invert_grayscale(mnist_digits):
            """ Makes black white, and white black """
            return 1 - mnist_digits

        mnist = get_mini_train_set(one_hot=False)
        batch_xs, batch_ys = mnist.train.next_batch(500)

        # spr
        to_visualise = vector_to_matrix_mnist(batch_xs)
        to_visualise = invert_grayscale(to_visualise)
        sprite_image = create_sprite_image(to_visualise)

        path_for_mnist_sprites = "./x.png"
        plt.imsave(path_for_mnist_sprites, sprite_image, cmap='gray')
        plt.imshow(sprite_image, cmap='gray')

        cls.show_embeddings(embeddings=batch_xs, labels=[y for y in batch_ys],
                            sprite_file_name=path_for_mnist_sprites,
                            sprite_size=[2, 2], tmp_dir="/tmp/mnist")

    @classmethod
    def show_random_embeddings(cls):
        embeddings = np.random.random((10, 8)).astype(np.float32)
        labels = [1, 2] * 5

        cls.show_embeddings(embeddings=embeddings,
                            labels=labels,
                            sprite_file_name=None,
                            tmp_dir="/tmp/random")


def show_embedding(feature_list, labels: list, log_dir: str = "./tmp/logs"):
    # embeddings
    embeddings = feature_list
    if isinstance(feature_list, list):
        # feature_list shape: [(1, self.d), (1, self.d)]
        embeddings = np.vstack(feature_list).reshape((len(feature_list), feature_list[0].shape[-1]))

    tf.logging.info("show_embedding, first 5 is {}".format(embeddings[:5]))
    tf.logging.info("show_embedding, first 5 sum is {}".format(np.sum(embeddings[:5], axis=1)))
    tf.logging.info("show_embedding, end 5 is {}".format(embeddings[-5:]))
    tf.logging.info("show_embedding, end 5 sum is {}".format(np.sum(embeddings[-5:], axis=1)))

    # label_info
    label_info = {}
    for label in labels:
        if label not in label_info:
            label_info[label] = 0
        label_info[label] += 1
    label_count_list = sorted(label_info.items(), key=lambda x: x[1], reverse=True)

    tf.logging.info("show_embedding, label with multi instance is {}".format(
        {label: label_count for (label, label_count) in label_count_list if label_count > 1}))

    EmbeddingVisualTools.show_embeddings(embeddings=embeddings, labels=labels, tmp_dir=log_dir)
    tf.logging.info("show show_embedding: tensorboard --logdir={}".format(os.path.abspath(log_dir)))


def show_distance_dense_plot(distance_arr: np.ndarray, save_file: str = None) -> (float, float):
    distance_mean = np.mean(distance_arr)
    distance_std = np.std(distance_arr)

    plt.hist(distance_arr, bins=10, rwidth=0.9, normed=True)
    plt.title('Distance distribution')
    plt.xlabel('Distance')
    plt.ylabel('Probability')
    plt.title(r'Histogram : $\mu={}$,$\sigma={}$'.format(
        get_pretty_float(float(distance_mean)), get_pretty_float(float(distance_std))))

    # plt.vlines(distance_mean, 0, distance_max, colors="c", linestyles="dashed")

    if save_file:
        plt.savefig(save_file)
    else:
        plt.show()

    return float(distance_mean), float(distance_std)


__all__ = ("EmbeddingVisualTools", "show_embedding",
           "PropVisualTools", "show_distance_dense_plot")
