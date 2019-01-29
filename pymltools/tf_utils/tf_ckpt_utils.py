# -*- coding:utf-8 -*-
from __future__ import absolute_import

import glob
import logging

import os
import shutil
import tensorflow as tf


def remove_old_checkpoint_file(train_dir):
    def parse_checkpoint_file(checkpoint):
        with open(checkpoint, "r") as fr:
            _keep_cpt_list = []
            for line in fr.readlines():
                if line.find('all_model_checkpoint_paths: "') == 0:
                    _keep_cpt_list.append(line.split('"')[1])

            if _keep_cpt_list:
                return "-".join(_keep_cpt_list[0].split("-")[:-1]), _keep_cpt_list

        return None, None

    def get_cpt_path(_file_name, _prefix_path):
        """
            返回 MnistBP-xxx
        :param _file_name: str, 文件名称
        :param _prefix_path: str, MnistBP
        :rtype: str|None
        """
        lines = _file_name.split(".")
        if len(lines) != 2:
            return None

        prefix_lines = lines[0].split("-")
        if len(prefix_lines) == 2 and prefix_lines[0] == _prefix_path:
            try:
                int(prefix_lines[1])
                return lines[0]
            except:
                pass
        return None

    prefix_path, keep_cpt_list = parse_checkpoint_file(os.path.join(train_dir, "checkpoint"))
    if prefix_path is None:
        return

    keep_cpt_set = set(keep_cpt_list)

    for file_name in os.listdir(train_dir):
        cpt_path = get_cpt_path(file_name, prefix_path)
        if cpt_path and cpt_path not in keep_cpt_set:
            os.remove(os.path.join(train_dir, file_name))


def get_last_meta_path(save_path):
    path = "/".join(save_path.split("/")[:-1])
    model_name = save_path.split("/")[-1]

    meta_file_info = {}
    for file_name in os.listdir(path):
        if file_name.find(model_name) == 0 and len(file_name) > 5 and file_name[-5:] == ".meta":
            step_str = file_name[:-5].split("-")[-1]
            try:
                meta_file_info[int(step_str)] = os.path.join(path, file_name)
            except ValueError as e:
                logging.error(e, exc_info=1)
                meta_file_info[0] = os.path.join(path, file_name)

    if not meta_file_info:
        return None

    meta_keys = list(meta_file_info.keys())
    meta_keys.sort()
    return meta_file_info[meta_keys[-1]]


def get_saver_and_last_step(meta_path, sess):
    if meta_path is None:
        return None, -1
    else:
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, meta_path[:-5])
        try:
            return saver, int(meta_path[:-5].split("-")[-1])
        except ValueError as e:
            logging.error(e, exc_info=1)
            return saver, -1


def tf_export_checkpoint(src_ckpt_dir: str, target_ckpt_dir: str):
    if not os.path.exists(target_ckpt_dir):
        os.mkdir(target_ckpt_dir)

    ckpt = tf.train.latest_checkpoint(src_ckpt_dir)
    if ckpt is None:
        raise ValueError("no checkpoint found in {}".format(src_ckpt_dir))

    ckpt_file_list = glob.glob("{}*".format(ckpt))
    for _file in ckpt_file_list:
        shutil.copy(_file, os.path.join(target_ckpt_dir, os.path.basename(_file)))

    shutil.copy(os.path.join(src_ckpt_dir, "checkpoint"), os.path.join(target_ckpt_dir, "checkpoint"))


__all__ = ("remove_old_checkpoint_file", "get_last_meta_path", "get_saver_and_last_step", "tf_export_checkpoint")
