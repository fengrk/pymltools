# -*- coding:utf-8 -*-
from __future__ import absolute_import

import logging
import platform
import time

import os
import shutil
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import DEFAULT_SOURCE_URL, dtypes, read_data_sets

from pyxtools import global_init_logger, set_time_zone, singleton, compress_by_tar, get_base_name_of_file, list_files
from .tf_ckpt_utils import tf_export_checkpoint


def init_logger(log_file="tf.log"):
    set_time_zone()
    global_init_logger(logging.INFO, log_file=log_file, reset_logger_name_list=["tensorflow"])


def get_wsl_path(file_path: str) -> str:
    if platform.system().lower().find("windows") > -1:
        return file_path

    if file_path.find(":") == 1:
        return "/mnt/" + file_path.split(":")[0].lower() + file_path[2:]

    return file_path


def save_file(file_list: list, target_dir: str = "/content/drive/ai", use_uid: bool = False, daemon: bool = True,
              only_save_latest_checkpoint: bool = False, logger=logging):
    logger.info("saving file ...")

    def _do_job(_file_list, _target_dir):
        for file_name in _file_list:
            _time_start = time.time()
            logger.info("trying to save file: {}".format(file_name))
            if only_save_latest_checkpoint and os.path.isdir(file_name) and tf.train.latest_checkpoint(file_name):
                new_file_name = file_name + "_backup"
                if os.path.exists(new_file_name) and os.path.isdir(new_file_name):
                    shutil.rmtree(new_file_name)

                tf_export_checkpoint(file_name, new_file_name)
                if os.path.exists(new_file_name) and tf.train.latest_checkpoint(new_file_name):
                    file_name = new_file_name
                    logger.info("trying to save train dir: {}".format(file_name))

            _name = get_base_name_of_file(file_name)
            if use_uid:
                tar_file_name = "{}.{}.tar.gz".format(_name, int(time.time()))
            else:
                tar_file_name = "{}.tar.gz".format(_name)

            compress_by_tar(source=file_name, target_file_name=tar_file_name, absolute_dir=False)
            if not os.path.exists(tar_file_name):
                logger.error("fail to compress/save {}".format(file_name))
                continue

            logger.info("success to compress file!")
            shutil.copy(tar_file_name, os.path.join(_target_dir, tar_file_name))

            logger.info(
                "success to save {} to {}, time cost {}s".format(file_name, target_dir, time.time() - _time_start))

    import threading
    _thread = threading.Thread(target=_do_job, args=(file_list, target_dir))
    _thread.daemon = daemon
    _thread.start()


@singleton
def is_gpu_available() -> bool:
    """

    :rtype: bool
    """
    return tf.test.is_gpu_available()


def get_mini_train_set(fake_data=False,
                       one_hot=False,
                       dtype=dtypes.float32,
                       reshape=True,
                       validation_size=5000,
                       seed=None,
                       source_url=DEFAULT_SOURCE_URL):
    mnist_path = get_wsl_path("E:/frkhit/Download/AI/data-set/MNIST_data")
    if not os.path.exists(mnist_path):
        os.mkdir(mnist_path)

    tar_path = get_wsl_path("E:/frkhit/Download/AI/data-set/mnist")
    if os.path.exists(tar_path):
        tar_file_list = [file_name for file_name in list_files(tar_path) if file_name.endswith(".gz")]
        for file_name in tar_file_list:
            dst_file_name = os.path.join(mnist_path, os.path.basename(file_name))
            if not os.path.exists(dst_file_name):
                shutil.copy(file_name, dst_file_name)

    return read_data_sets(mnist_path,
                          fake_data=fake_data,
                          one_hot=one_hot,
                          dtype=dtype,
                          reshape=reshape,
                          validation_size=validation_size,
                          seed=seed,
                          source_url=source_url)


__all__ = ("get_wsl_path", "init_logger", "save_file", "is_gpu_available", "get_mini_train_set")
