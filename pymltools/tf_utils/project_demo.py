# -*- coding:utf-8 -*-
from __future__ import absolute_import

import logging
import time

import os
import tensorflow as tf
from enum import Enum

from pyxtools import get_base_name_of_file
from .basic_tools import save_file


def mkdir_path(path):
    if not tf.gfile.Exists(path):
        tf.gfile.MakeDirs(path)


class ProcessMode(Enum):
    train = 0
    test = 1


class OptimizerType(Enum):
    adam = 0
    sgd = 1


class Params(object):
    def __init__(self, _dict):
        self.__dict__.update(_dict)


def estimator_iter_process(loop_process, iter_stop_time: float = 1e12, loop_process_min_epoch: int = 1,
                           end_process_func=None, loop_process_max_epoch: int = 10000,
                           ignore_error_in_loop_process: bool = True, loop_process_start_epoch: int = 0,
                           logger=logging.getLogger("estimator_iter_process")):
    """

    Args:
        loop_process: func, with args (total_epochs, epoch_nums).
        iter_stop_time: float, timestamp in second, hope stop before this time.
        loop_process_min_epoch: int,
        end_process_func: func, call before exit
        loop_process_max_epoch: int, max epoch allowed to run loop_process
        loop_process_start_epoch: int, start epoch
        ignore_error_in_loop_process: bool, whether ignore error when running loop_process func
        logger: logging.Logger

    Returns:

    """

    def _sum_time(time_cost_list: list) -> (float, int):
        _total_time, _total_epoch = 0.0, 0
        for (_time, _epoch) in time_cost_list:
            _total_epoch += _epoch
            _total_time += _time

        return _total_time, _total_epoch

    assert loop_process_min_epoch > 0
    _total_time_start = time.time()
    if _total_time_start > iter_stop_time:
        raise ValueError("iter_stop_time must greater current time {}".format(_total_time_start))
    if loop_process is None:
        raise ValueError("loop_process cannot be null!")

    if loop_process_max_epoch <= loop_process_min_epoch:
        epoch_list = [loop_process_max_epoch]
    else:
        epoch_list = [loop_process_min_epoch] * int(loop_process_max_epoch / loop_process_min_epoch) + \
                     [loop_process_max_epoch % loop_process_min_epoch]

    time_cost_of_loop_process_list = []

    for epoch_nums in epoch_list:
        # time cost
        loop_time_start = time.time()
        total_time_cost, total_epochs = _sum_time(time_cost_of_loop_process_list)
        if total_epochs > 0:
            time_cost_per_epoch = total_time_cost / total_epochs
            _time_cost_per_epoch_list = []
            for (_time_cost, _epoch_count) in time_cost_of_loop_process_list:
                if _epoch_count > 0:
                    _time_cost_per_epoch_list.append(_time_cost / _epoch_count)

            time_cost_per_epoch = max(time_cost_per_epoch, max(_time_cost_per_epoch_list))
            epoch_nums = min(epoch_nums, int((iter_stop_time - loop_time_start) / time_cost_per_epoch))
        else:
            total_epochs = loop_process_start_epoch

        if epoch_nums <= 0:
            logger.warning("give up running loop_process for timeout!")
            break

        try:
            logger.info("try to run {} from No.{}!".format(epoch_nums, total_epochs))
            loop_process(total_epochs, epoch_nums)
            logger.info("success to run {} from No.{}, time cost {}s!".format(
                epoch_nums, total_epochs, round(time.time() - loop_time_start, 2)))
        except Exception as e:
            logger.error("fail to run {} from No.{}, error is {}\n".format(epoch_nums, total_epochs, e), exc_info=True)
            if not ignore_error_in_loop_process:
                logger.error("give up running loop_process for error!")
                break

        # record time, ignore error
        time_cost_of_loop_process_list.append((time.time() - loop_time_start, epoch_nums))

        # double check
        if (iter_stop_time - time.time()) < 600:
            logger.warning("give up running loop_process for timeout!")
            break

    # end logger of loop process
    _, loop_process_epoch_count = _sum_time(time_cost_of_loop_process_list)
    _total_time_cost = time.time() - _total_time_start
    logger.info("time cost {}s/{}h for run {} epoch of loop process".format(
        round(_total_time_cost, 2), round(_total_time_cost / 3600, 2), loop_process_epoch_count))

    # run end_process_func
    if end_process_func:
        _time_start = time.time()
        logger.info("running end_process_func...")
        end_process_func()
        logger.info("success to run end_process_func, time cost {}s".format(round(time.time() - _time_start, 2)))

    # logger
    _total_time_cost = time.time() - _total_time_start
    logger.info("success to run estimator_iter_process, time cost {}s/{}h".format(
        round(_total_time_cost, 2), round(_total_time_cost / 3600, 2)))


def colab_save_file_func(train_dir: str, logger=logging, only_save_latest_checkpoint: bool = True,
                         daemon: bool = False):
    try:
        _file_list = []
        if os.path.exists("tf.log"):
            _file_list.append("tf.log")
        if os.path.exists(get_base_name_of_file(train_dir)):
            _file_list.append(get_base_name_of_file(train_dir))
        if _file_list:
            save_file(file_list=_file_list, daemon=daemon, logger=logger,
                      only_save_latest_checkpoint=only_save_latest_checkpoint)
        else:
            logger.warning("no file to save!")
    except Exception as ex:
        logger.error(ex, exc_info=True)


__all__ = ("mkdir_path", "Params", "ProcessMode", "estimator_iter_process",
           "colab_save_file_func", "OptimizerType",)
