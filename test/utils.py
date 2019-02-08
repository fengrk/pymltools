# -*- coding:utf-8 -*-
from __future__ import absolute_import

import logging

import os

from pyxtools import list_files, get_md5_for_file


def list_jpg_file() -> list:
    return [img_file for img_file in list_files(os.path.dirname(__file__)) if img_file.endswith(".jpg")]


def compare_two_dir(dir1: str, dir2: str) -> bool:
    # is dir
    if not os.path.isdir(dir1) or not os.path.isdir(dir2):
        logging.error("{}, {} is not dir".format(dir1, dir2))
        return False

    # file is same
    file_list_1 = list_files(dir1)
    file_list_2 = list_files(dir2)
    if len(file_list_1) != len(file_list_2):
        logging.error("file count not equal!")
        return False

    md5_list_1 = [get_md5_for_file(_file) for _file in file_list_1]
    md5_list_2 = [get_md5_for_file(_file) for _file in file_list_2]
    logging.info("md5 list of {} is {}".format(
        dir1, [(file_list_1[index], md5_list_1[index]) for index in range(len(file_list_1))]))

    logging.info("md5 list of {} is {}".format(
        dir2, [(file_list_2[index], md5_list_2[index]) for index in range(len(file_list_2))]))
    return set(md5_list_1) == set(md5_list_2)
