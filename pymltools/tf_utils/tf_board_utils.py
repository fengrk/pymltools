# -*- coding:utf-8 -*-
from __future__ import absolute_import

import logging

import os
import tensorflow as tf


def list_all_var() -> list:
    return tf.trainable_variables(scope=None)


class TensorboardUtils(object):
    @classmethod
    def variable_summaries(cls, var, scope=None):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope(scope, default_name=var.name.split(":")[0] if var.name.find(":") > 0 else var.name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    @classmethod
    def image_summaries(cls, var, scope=None):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope(scope, default_name=var.name.split(":")[0] if var.name.find(":") > 0 else var.name):
            tf.summary.image("image", var)

    @classmethod
    def variable_list_summaries(cls, var_list=None):
        """

        :type var_list: list | None
        """
        if var_list is None:
            # todo would ignore some var in new model part
            # var_list = tf.contrib.framework.get_model_variables()
            var_list = list_all_var()

        for var in var_list:
            cls.variable_summaries(var, scope=None)

    @classmethod
    def image_list_summaries(cls, var_list=None):
        """
        可视化images
        :type var_list: list | None
        """
        if var_list is None:
            var_list = list_all_var()

        for var in var_list:
            cls.image_summaries(var, scope=None)


def list_trainable_var(train_scope):
    """

    :rtype: list|None
    """
    if not train_scope:
        return None

    trainable_var_scope = [scope.strip() for scope in train_scope.split(',') if len(scope.strip()) > 0]

    var_list = tf.trainable_variables(scope=None)

    trainable_var_list = []
    for v in var_list:
        is_trainable = False
        for _var_scope in trainable_var_scope:
            if v.name.startswith(_var_scope):
                is_trainable = True
                break
        if is_trainable:
            trainable_var_list.append(v)

    if trainable_var_list:
        logging.info("fc trainable var is {}!".format(", ".join([var.name for var in trainable_var_list])))
    else:
        logging.info("fc trainable var is all vars!")

    return trainable_var_list if trainable_var_list else None


def save_model_from_ckpt_file(pretrained_ckpt_file: str, model_dir: str):
    """

    Args:
        pretrained_ckpt_file: path of XX.pb file or last_checkpoint
        model_dir: dir to save log
    """
    with tf.Session() as sess:
        if os.path.exists(pretrained_ckpt_file) and os.path.isfile(pretrained_ckpt_file):
            with tf.gfile.FastGFile(pretrained_ckpt_file, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def)
        else:
            saver = tf.train.import_meta_graph("{}.meta".format(pretrained_ckpt_file))
            saver.restore(sess, pretrained_ckpt_file)

        train_writer = tf.summary.FileWriter(model_dir, sess.graph)
        train_writer.close()


__all__ = ("list_all_var", "TensorboardUtils",
           "list_trainable_var", "save_model_from_ckpt_file")
