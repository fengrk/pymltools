# -*- coding:utf-8 -*-
from __future__ import absolute_import

import logging
import math

import os
import tensorflow as tf
import tensorflow.contrib.framework as framework
import tensorflow.contrib.slim as slim

from pyxtools import get_md5
from .tf_board_utils import list_all_var


def get_init_fn(pretrained_model_file, checkpoint_exclude_scopes=None):
    """
    This function is copied from TF slim.

    Returns a function run by the chief worker to warm-start the training.

    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
      An init function run by the supervisor.
    """
    tf.logging.info('Use pretrained model %s' % pretrained_model_file)

    exclusions = []
    if checkpoint_exclude_scopes:
        exclusions = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]

    variables_to_restore = []
    for var in slim.get_model_variables():  # todo maybe list_all_var() is better
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(
        pretrained_model_file,
        variables_to_restore,
        ignore_missing_vars=True)


class TensorSortUtils(object):
    def __init__(self, model_var_dict: dict, ckpt_var_dict: dict, index_add: int = 0, root_name: str = None):
        self.ckpt_var_dict = ckpt_var_dict
        self.model_var_dict = model_var_dict
        self.index_add = index_add
        self.root_name = root_name

    def get_var_info(self, tensor):
        var_name = self._get_var_name(tensor)
        main_name = var_name.split("/")[0]
        if self.root_name and self.root_name == main_name:
            main_name = var_name.split("/")[1]

        main_name_list = main_name.split("_")
        try:
            index = int(main_name_list[-1])
            base_name = "_".join(main_name_list[:-1])
        except Exception:
            index = 0
            base_name = main_name

        # shape
        s = tensor.get_shape()
        shape_hash = get_md5(":".join([str(s[i].value) for i in range(len(s))]).encode("utf-8"))

        # pattern
        if self.root_name and self.root_name == var_name.split("/")[0]:
            name_pattern = "/".join(var_name.split("/")[1:]).replace(main_name, base_name + "_{}", 1)
        else:
            name_pattern = var_name.replace(main_name, base_name + "_{}", 1)
        return var_name, base_name, index, shape_hash, name_pattern

    def get_var_name(self, model_tensor):
        name, base_name, index, _, name_pattern = self.get_var_info(model_tensor)
        if self.index_add == 0:
            return name

        return name_pattern.format(index + self.index_add)

    @staticmethod
    def _get_var_name(tensor) -> str:
        return tensor.name.split(":")[0]

    def get_tensor_in_checkpoint(self, model_tensor) -> (str, str):
        var_name = self.get_var_name(model_tensor)

        if var_name not in self.ckpt_var_dict:
            var_name = os.path.join(var_name, 'ExponentialMovingAverage')
            if var_name not in self.ckpt_var_dict:
                return "", 'Skip init {}(name in model) because it is not in ckpt file, current var name is {}'.format(
                    self._get_var_name(model_tensor), var_name)

        if not model_tensor.get_shape().is_compatible_with(self.ckpt_var_dict[var_name]):
            return "", 'Skip init {}(name in model) from {} in ckpt file because shape dismatch: {} vs {}'.format(
                self._get_var_name(model_tensor), var_name, model_tensor.get_shape(), self.ckpt_var_dict[var_name])

        return var_name, ""


def get_init_fn_v2(pretrained_model_file: str, exclusion_list: list, logger=logging.getLogger("get_init_fn_v2"),
                   index_add: int = 0, model_root_name: str = None):
    checkpoint_reader = framework.load_checkpoint(
        pretrained_model_file)
    ckpt_variable_shape_map = checkpoint_reader.get_variable_to_shape_map()

    model_variable_to_restore = list_all_var()

    # Variable filtering by given exclude_scopes.
    model_variables_should_be_restore = {}
    for v in model_variable_to_restore:
        excluded = False
        for exclusion in exclusion_list:
            if v.name.startswith(exclusion):
                excluded = True
                break

        var_name = v.name.split(':')[0]
        if not excluded:
            model_variables_should_be_restore[var_name] = v
        else:
            logger.warning('Skip init {}(name in model) because it is excluded by user.'.format(var_name))

    # Final filter by checking shape matching and skipping variables that
    # are not in the checkpoint.
    ckpt_variables_used = {}
    model_var_skip_list = []

    util = TensorSortUtils(model_var_dict=model_variables_should_be_restore, ckpt_var_dict=ckpt_variable_shape_map,
                           index_add=index_add, root_name=model_root_name)
    for var_name, var_tensor in model_variables_should_be_restore.items():
        ckpt_tensor_name, error_info = util.get_tensor_in_checkpoint(var_tensor)
        if not ckpt_tensor_name:
            logger.warning(error_info)
            model_var_skip_list.append(var_name)
        else:
            ckpt_variables_used[ckpt_tensor_name] = var_tensor

    if model_var_skip_list:
        logger.warning("var(name in ckpt file) skip with unexpect: {}".format(model_var_skip_list))
        logger.info("var in ckpt file is : {}".format([var_name for var_name in ckpt_variable_shape_map]))
        logger.info("var(name in ckpt file) restored is : {}".format([var_name for var_name in ckpt_variables_used]))

    return framework.assign_from_checkpoint_fn(
        pretrained_model_file, ckpt_variables_used)


class InitFromPretrainedCheckpointHook(tf.train.SessionRunHook):
    """
        pretrained model
    """

    def __init__(self, pretrained_checkpoint_file: str, exclusion_list: list, index_add: int = 0,
                 root_name: str = None):
        self.logger = logging.getLogger(self.__class__.__name__)

        if pretrained_checkpoint_file is None:
            raise ValueError('pretrained_checkpoint_dir must be specified.')
        if not isinstance(exclusion_list, (tuple, list)):
            raise ValueError("exclusion_list must be list!")

        self._pretrained_checkpoint_file = pretrained_checkpoint_file
        self.exclusion_list = exclusion_list
        self._index_add = index_add
        self._root_name = root_name
        self._init_fn = None

    def begin(self):
        self._init_fn = get_init_fn_v2(
            self._pretrained_checkpoint_file, self.exclusion_list, logger=self.logger, index_add=self._index_add,
            model_root_name=self._root_name)

    def after_create_session(self, session, coord):
        self.logger.info('Restoring weights from {}...'.format(self._pretrained_checkpoint_file))
        self._init_fn(session)
        self.logger.info('Done restoring weights from {}!'.format(self._pretrained_checkpoint_file))


class InitHookForNetWithMultiSamePart(InitFromPretrainedCheckpointHook):
    """
         pretrained model
    """

    def __init__(self, pretrained_checkpoint_file: str, exclusion_list: list):
        super(InitHookForNetWithMultiSamePart, self).__init__(pretrained_checkpoint_file, exclusion_list)

    def begin(self):
        checkpoint_reader = framework.load_checkpoint(self._pretrained_checkpoint_file)
        # var tensor in ckpt
        ckpt_variable_shape_map = checkpoint_reader.get_variable_to_shape_map()
        # name to value
        ckpt_variable_tensor_map = {var_name: checkpoint_reader.get_tensor(var_name)
                                    for var_name in ckpt_variable_shape_map}
        # all var tensor in model
        variable_to_restore = framework.get_model_variables()

        # Variable filtering by given exclude_scopes.
        filtered_variables_to_restore = {}  # var tensor to restore
        for v in variable_to_restore:
            excluded = False
            for exclusion in self.exclusion_list:
                if self.is_same_part(exclusion, v):
                    excluded = True  # not restore
                    break

            var_name = v.name.split(':')[0]
            if not excluded:
                filtered_variables_to_restore[var_name] = v
            else:
                self.logger.warning('Skip init [%s] because it is excluded by user.', var_name)

        # Final filter by checking shape matching and skipping variables that
        # are not in the checkpoint.
        final_map_to_value = {}
        for var_name, var_tensor in filtered_variables_to_restore.items():
            ckpt_var_name = self.get_part_name(var_name)

            if ckpt_var_name not in ckpt_variable_shape_map:
                self.logger.warning("Skip init [%s] because [%s] not in ckpt file", var_name, var_name)
                continue

            if not var_tensor.get_shape().is_compatible_with(ckpt_variable_shape_map[ckpt_var_name]):
                # Skip init variable from ckpt if shape dismatch.
                self.logger.warning(
                    'Skip init [%s] from [%s] in ckpt because shape dismatch: %s vs %s',
                    var_tensor.name, ckpt_var_name,
                    var_tensor.get_shape(), ckpt_variable_shape_map[ckpt_var_name])
                continue

            final_map_to_value[var_name] = ckpt_variable_tensor_map[ckpt_var_name]

        # restore
        self.logger.info("var restore: {}".format(",".join([var for var in final_map_to_value])))
        self._init_fn = framework.assign_from_values_fn(final_map_to_value)

    @staticmethod
    def is_same_part(scope, net_var_tensor):
        """

        :rtype: bool
        """
        net_var_name = net_var_tensor.name
        var_list = net_var_name.split("/")
        var_list[0] = "_".join(var_list[0].split("_")[:-1])

        return bool(scope == "/".join(var_list))

    @staticmethod
    def get_part_name(var_name):
        var_list = var_name.split("/")
        var_list[0] = "_".join(var_list[0].split("_")[:-1])
        return "/".join(var_list)


class LossLoggingHookForTrain(tf.train.SessionRunHook):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def after_run(self, run_context, run_values):
        losses = run_context.session.graph.get_collection("losses")
        self.logger.info("loss is {}".format(run_context.session.run(losses)))


class LossStepHookForTrain(tf.train.SessionRunHook):
    NAF_LOSS = 1e10

    def __init__(self, log_after_run: bool = True, log_after_end: bool = False, show_log_per_steps: int = 10,
                 run_after_run_per_steps: int = 1):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._loss_list = []
        self._step_list = []
        self._store_loss_list = []
        self._global_step_tensor = None
        self._log_after_end = log_after_end
        self._log_after_run = log_after_run
        self._run_after_run_per_steps = run_after_run_per_steps

        self._show_log_per_steps = max(1, show_log_per_steps)
        self._show_log_per_steps_count = 0
        self._global_step = 0

    def begin(self):
        self._global_step_tensor = tf.train.get_global_step()

    def after_run(self, run_context, run_values):
        self._global_step += 1
        if self._log_after_run and self._global_step % self._run_after_run_per_steps == 0:
            _loss_tensor = run_context.session.graph.get_collection("losses")
            losses, step = run_context.session.run([_loss_tensor, self._global_step_tensor])
            loss = losses[0]  # todo len of losses == 1
            self._loss_list.append(loss)
            self._step_list.append(step)
            self._store_loss_list.append((loss, step))

            self._show_log_per_steps_count += 1
            if self._show_log_per_steps_count % self._show_log_per_steps == 0:
                _step_loss_list = self._loss_list[-self._show_log_per_steps:]
                if _step_loss_list:
                    self.logger.info("current loss is {}, average loss after last show is {}, current steps is {}".
                                     format(loss, sum(_step_loss_list) / len(_step_loss_list), step))
                else:
                    self.logger.info("current loss is {}, current steps is {}".format(loss, step))

    def list_loss(self) -> list:
        return self._loss_list

    def list_steps(self) -> list:
        return self._step_list

    def pop_loss_and_steps(self) -> (list, list):
        loss_list = list(self._loss_list)
        step_list = list(self._step_list)

        self._loss_list.clear()
        self._step_list.clear()

        return loss_list, step_list

    def get_average_loss(self) -> float:
        count = len(self._loss_list)
        if count > 10:
            _loss_list = list(self._loss_list)
            _loss_list.sort()
            return sum(_loss_list[1:-1]) / (count - 2)
        elif count > 0:
            return sum(self._loss_list) / count
        else:
            return self.NAF_LOSS

    def list_loss_of_each_epoch(self, steps_per_epoch: int) -> list:
        assert steps_per_epoch > 0
        if not self._store_loss_list:
            return []

        loss_of_epoch_list = [(0.0, 0)] * math.ceil(self._store_loss_list[-1][1] / steps_per_epoch)

        for (loss, step) in self._store_loss_list:
            _epoch = int(step / steps_per_epoch)
            loss_of_epoch_list[_epoch][0] += loss
            loss_of_epoch_list[_epoch][1] += 1

        for i in range(len(loss_of_epoch_list)):
            if loss_of_epoch_list[i][1] == 0:
                loss_of_epoch_list[i][0] = self.NAF_LOSS
                loss_of_epoch_list[i][1] = 1

        return [loss / step for (loss, step) in loss_of_epoch_list]

    def end(self, session):
        # todo error
        if self._log_after_end:
            _loss_tensor = session.graph.get_collection("losses")
            losses, step = session.run([_loss_tensor, self._global_step_tensor])
            self._loss_list.extend(losses)
            self._step_list.append(step)
            self.logger.info("loss is {}, steps is {}".format(losses[0], step))


__all__ = ("LossLoggingHookForTrain", "InitFromPretrainedCheckpointHook", "InitHookForNetWithMultiSamePart",
           "get_init_fn", "get_init_fn_v2", "LossStepHookForTrain",)
