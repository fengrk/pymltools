# -*- coding:utf-8 -*-
from __future__ import absolute_import

import logging
import time

import cv2
import h5py
import os
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
from enum import Enum
from tensorflow.python import debug as tf_debug

from pyxtools import get_base_name_of_file, SingletonMixin, FileCache, download_big_file, list_files, read_image, \
    save_image
from .basic_tools import save_file
from .init_hook_utils import InitFromPretrainedCheckpointHook, get_init_fn
from .tf_ckpt_utils import remove_old_checkpoint_file
from .tf_dataset import DatasetUtils
from .tf_record_utils import TfRecordUtils


def mkdir_path(path):
    if not tf.gfile.Exists(path):
        tf.gfile.MakeDirs(path)


class DataPreProcess(SingletonMixin):
    """
        处理数据集
    """
    IMAGE_ID_SEP = DatasetUtils.IMAGE_ID_SEP
    SPLIT_TRAIN = DatasetUtils.SPLIT_TRAIN
    SPLIT_VALIDATION = DatasetUtils.SPLIT_VALIDATION
    SPLIT_TEST = DatasetUtils.SPLIT_TEST
    SPLIT_PREDICT = DatasetUtils.SPLIT_PREDICT
    NUM_SHARDS = 5
    logger = logging.getLogger("DataPreProcess")

    def __init__(self, ):
        self.image_dir = ""
        self.tf_record_dir = ""
        self._target_size = None
        self.IMAGE_ORIGIN_DIRNAME = "images"
        self.IMAGE_RESIZED_DIRNAME = "images_resized"
        self.tf_record_utils = DatasetUtils("", self.NUM_SHARDS, [], (), None)
        self.image_pk_vs_class_id = {}
        self.image_id_vs_image_pk = {}
        self.cache = None

    @classmethod
    def instance(cls):
        """

        :rtype: DataPreProcess
        """
        return super(DataPreProcess, cls).instance()

    def prepare(self, root_path, target_size=None, cache_file=None):
        # record dir
        self.image_dir = os.path.join(root_path, "CUB_200_2011")
        self.tf_record_dir = os.path.join(self.image_dir, "tf")
        self.tf_record_utils.tf_record_dir = self.tf_record_dir
        mkdir_path(self.tf_record_dir)

        # cache
        if cache_file is None:
            cache_file = "{}.pkl".format(self.__class__.__name__)
        if not os.path.exists(cache_file):
            try:
                download_big_file(
                    "https://raw.githubusercontent.com/frkhit/file_servers/master/{}".format(cache_file),
                    cache_file
                )
            except Exception as e:
                self.logger.error(e)
        self.cache = FileCache(cache_file)

        # data init
        self._target_size = target_size
        if target_size is None:
            self.IMAGE_RESIZED_DIRNAME = self.IMAGE_ORIGIN_DIRNAME

        raw_image_list = list_files(os.path.join(self.image_dir, self.IMAGE_ORIGIN_DIRNAME))
        all_image = self.resize_image(raw_image_list)
        self.image_pk_vs_class_id = self._parse_info_file(os.path.join(self.image_dir, "image_class_labels.txt"))
        image_pk_vs_file = self._parse_info_file(os.path.join(self.image_dir, "images.txt"))
        image_pk_vs_to_train = self._parse_info_file(os.path.join(self.image_dir, "train_test_split.txt"))
        self.image_id_vs_image_pk = {self.get_image_id(image_file): image_pk
                                     for image_pk, image_file in image_pk_vs_file.items()}

        # get class id func
        def get_class_id_func(image_file):
            return self.get_class_id(image_file)

        self.tf_record_utils.get_class_id_func = get_class_id_func

        # train, validate, test, predict dataset
        training_filenames, validation_filenames, test_filenames, predict_filenames = \
            self.cache.get(self.SPLIT_TRAIN), self.cache.get(self.SPLIT_VALIDATION), \
            self.cache.get(self.SPLIT_TEST), self.cache.get(self.SPLIT_PREDICT)

        if training_filenames is None or validation_filenames is None or test_filenames is None or predict_filenames is None:
            predict_filenames = []
            other_file_list = []
            for image_file_name in all_image:
                _image_id = self.get_image_id(image_file_name)
                _image_pk = self.image_id_vs_image_pk[_image_id]
                _class_id = int(self.image_pk_vs_class_id[_image_pk]) - 1
                _to_train = bool(int(image_pk_vs_to_train[_image_pk]))
                if _to_train:
                    other_file_list.append(image_file_name)
                else:
                    predict_filenames.append(image_file_name)

            training_filenames, validation_filenames, test_filenames, _ = \
                self.tf_record_utils.create_file_list([0.75, 0.15, 0.1, 0], other_file_list, get_class_id_func)
            self.cache.set(self.SPLIT_TRAIN, training_filenames)
            self.cache.set(self.SPLIT_VALIDATION, validation_filenames)
            self.cache.set(self.SPLIT_TEST, test_filenames)
            self.cache.set(self.SPLIT_PREDICT, predict_filenames)

        self.tf_record_utils.set_file_name_list(
            [training_filenames, validation_filenames, test_filenames, predict_filenames])

        # label_id_name_tuple
        class_info = self._parse_info_file(os.path.join(self.image_dir, "classes.txt"))
        label_id_name_tuple = [(int(class_id) - 1, class_name) for class_id, class_name in class_info.items()]
        self.tf_record_utils.set_label_id_name(tuple(label_id_name_tuple))

        # build tf record file
        self.tf_record_utils.convert_images_with_labels()

        # show dataset info
        self.logger.info("image count info: train[{}], validate[{}], test[{}], predict[{}]".format(
            len(training_filenames),
            len(validation_filenames),
            len(test_filenames),
            len(predict_filenames)
        ))
        self.logger.info("net info: classes[{}], shard_of_tf_record[{}]".format(
            self.num_of_class, self.NUM_SHARDS
        ))

    def get_class_id(self, image_file_name):
        """

        :rtype: int|None
        :type image_file_name: str
        """
        return int(self.image_pk_vs_class_id[self.image_id_vs_image_pk[self.get_image_id(image_file_name)]]) - 1

    @classmethod
    def get_image_id(cls, image_file_name):
        """

        :rtype: str
        :type image_file_name: str
        """
        return ".".join(os.path.basename(image_file_name).split(".")[:-1])

    @staticmethod
    def _parse_info_file(file_name):
        """
        解析数据集文件
        :rtype: dict
        """
        result_dict = {}
        with open(file_name, "r", encoding="utf-8") as fr:
            for line in fr:
                if line and len(line) > 2:
                    _list = line.strip().split(" ")
                    assert len(_list) == 2
                    result_dict[_list[0]] = _list[1]

        return result_dict

    def list_tf_record_files(self, split_name):
        """

        :rtype: list of str
        """
        return self.tf_record_utils.list_tf_record_files(split_name)

    def resize_image(self, file_list: list) -> list:
        if self._target_size is None:
            self.logger.info("no need to resize images!")
            return file_list

        self.logger.info("resizing images...")

        def resize_image(image):
            old_height, old_width, _ = image.shape
            if old_height >= old_width:
                return cv2.resize(image,
                                  (int(old_width * self._target_size / old_height), self._target_size),
                                  interpolation=cv2.INTER_CUBIC)
            else:
                return cv2.resize(image,
                                  (self._target_size, int(old_height * self._target_size / old_width)),
                                  interpolation=cv2.INTER_CUBIC)

        def change_file(full_file_list) -> list:
            new_full_file_list = []
            for full_name in full_file_list:
                new_file_name = full_name.replace(
                    "{}CUB_200_2011{}{}{}".format(os.sep, os.sep, self.IMAGE_ORIGIN_DIRNAME, os.sep),
                    "{}CUB_200_2011{}{}{}".format(os.sep, os.sep, self.IMAGE_RESIZED_DIRNAME, os.sep))
                new_full_file_list.append(new_file_name)

                if not os.path.exists(new_file_name):
                    resized_image = resize_image(read_image(full_name))
                    if not os.path.exists(os.path.dirname(new_file_name)):
                        os.makedirs(os.path.dirname(new_file_name))
                    save_image(new_file_name, resized_image)

            return new_full_file_list

        new_file_list = change_file(file_list)

        self.logger.info("success to resize images!")

        return new_file_list

    def debug_tf_parser(self):
        return self.tf_record_utils.debug_tf_parser()

    @property
    def num_of_class(self):
        return self.tf_record_utils.num_of_class


class AbstractEstimator(object):
    def __init__(self, model_name, train_ckpt_dir, pretrained_ckpt_file=None):
        self.logger = logging.getLogger(self.__class__.__name__)

        # basic var
        self.model_name = model_name
        self.PRETRAINED_CKPT_FILE = pretrained_ckpt_file
        self.TRAIN_CKPT_DIR = train_ckpt_dir
        self._classifier = None
        self.cache = FileCache(os.path.join(self.TRAIN_CKPT_DIR, "pickle.pkl"))
        self.params = Params({})
        self.prepare_experiment_env()

        # learning var
        self.batch_size = 8
        self.train_data_size = 0
        self.learning_rate = 1e-4
        self.learning_rate_decay_steps = 500
        self.learning_rate_decay_epoch_num = 4

        # training var
        self.train_loss_hook = None
        self.global_debug_hook = None
        self.train_restore_hook = None
        self._checkpoint_exist = bool(tf.train.latest_checkpoint(self.TRAIN_CKPT_DIR) is not None)

    @classmethod
    def create_debug_hook(cls):
        return tf_debug.LocalCLIDebugHook()

    def get_epoch_num(self) -> int:
        _epoch = self.cache.get("epoch")
        return 0 if _epoch is None else _epoch

    def set_epoch_num(self, count: int):
        self.cache.set("epoch", count)

    def prepare_experiment_env(self):
        """
            准备实验环境
        :return:
        """
        # 准备目录
        self.mkdir_path(self.TRAIN_CKPT_DIR)

        # 准备 pre-trained model
        if self.PRETRAINED_CKPT_FILE is not None:
            if os.path.exists(self.PRETRAINED_CKPT_FILE):
                if os.path.isdir(self.PRETRAINED_CKPT_FILE):
                    ckpt_file = tf.train.latest_checkpoint(self.PRETRAINED_CKPT_FILE)
                    if ckpt_file:
                        self.PRETRAINED_CKPT_FILE = ckpt_file
                        return
            else:
                ckpt_file = tf.train.latest_checkpoint(os.path.dirname(self.PRETRAINED_CKPT_FILE))
                if ckpt_file:
                    self.PRETRAINED_CKPT_FILE = ckpt_file
                    return
            try:
                if tf.contrib.framework.load_checkpoint(self.PRETRAINED_CKPT_FILE):
                    return
            except Exception as e:
                self.logger.warning(e)

            self.logger.warning("checkpoint not exists!")

    @staticmethod
    def mkdir_path(path):
        return mkdir_path(path)

    def get_dataset_func(self, split_name, num_epochs=1, shuffle=True, batch_size=64, num_parallel_calls=2,
                         prefetch_size=2, shuffle_size=4):
        raise NotImplementedError

    def model_fun(self, ):
        """ 返回func """
        raise NotImplementedError

    def get_classifier(self):
        """

        :rtype: tf.estimator.Estimator
        """
        if self._classifier is None:
            config = tf.estimator.RunConfig(
                model_dir=self.TRAIN_CKPT_DIR,
                save_checkpoints_secs=120,
                keep_checkpoint_max=2
            )

            self._classifier = tf.estimator.Estimator(model_fn=self.model_fun(), config=config, params=self.params)
        return self._classifier

    @staticmethod
    def create_learning_rate_op(learning_rate: float):
        return tf.train.exponential_decay(
            learning_rate,  # Base learning rate.
            tf.train.get_global_step(),  # Current index into the dataset.
            100,  # Decay step.
            0.995,  # Decay rate.
            staircase=True)

    def get_restore_hook(self):
        if self._checkpoint_exist:
            return

        self._checkpoint_exist = bool(tf.train.latest_checkpoint(self.TRAIN_CKPT_DIR) is not None)
        if self._checkpoint_exist:
            return

        if self.train_restore_hook:
            return self.train_restore_hook

        if self.PRETRAINED_CKPT_FILE:
            self.train_restore_hook = InitFromPretrainedCheckpointHook(self.PRETRAINED_CKPT_FILE,
                                                                       exclusion_list=["global_step"])
            return self.train_restore_hook

        return

    def train(self, batch_size=64, num_epochs=100, shuffle=True, steps=500):
        """
            训练
        :return:
        """
        # remove old ckpt file
        if os.path.exists(self.TRAIN_CKPT_DIR):
            try:
                remove_old_checkpoint_file(self.TRAIN_CKPT_DIR)
            except Exception as e:
                self.logger.error("fail to remove old checkpoint file: {}".format(e))

        # train
        split_name = DataPreProcess.SPLIT_TRAIN

        classifier = self.get_classifier()

        train_hooks = []
        restore_hook = self.get_restore_hook()
        if restore_hook:
            train_hooks.append(restore_hook)
        if self.train_loss_hook:
            train_hooks.append(self.train_loss_hook)
        if self.global_debug_hook:
            train_hooks.append(self.global_debug_hook)

        _epoch = self.get_epoch_num()

        if steps is None:
            self.logger.info(
                "start to train No.{}~{} epoch... ".format(_epoch + 1, _epoch + num_epochs)
            )
        else:
            self.logger.info("start to train No.{} epoch... ".format(_epoch + 1))

        classifier.train(
            input_fn=self.get_dataset_func(
                split_name=split_name, num_epochs=num_epochs, shuffle=shuffle, batch_size=batch_size
            ),
            hooks=None if not train_hooks else train_hooks,
            steps=steps,
        )

        if steps is None:
            _epoch += num_epochs
            self.set_epoch_num(_epoch)
            self.logger.info("success to train No.{} epoch! ".format(_epoch))

    def evaluate(self, num_epochs=1, batch_size=128, steps=None):
        """
        # Fine-tune only the new layers for 1000 steps.
        # Run evaluation.
        :return:
        """
        return self._evaluate_model(split_name=DataPreProcess.SPLIT_VALIDATION,
                                    num_epochs=num_epochs,
                                    batch_size=batch_size,
                                    steps=steps)

    def _evaluate_model(self, split_name, num_epochs=1, batch_size=128, steps=None, ):
        """
        # Fine-tune only the new layers for 1000 steps.
        # Run evaluation.
        :return:
        """
        classifier = self.get_classifier()

        _epoch = self.get_epoch_num()

        self.logger.info("start to {} after {} epochs... ".format(split_name, _epoch))

        train_hooks = []
        restore_hook = self.get_restore_hook()
        if restore_hook:
            train_hooks.append(restore_hook)
        if self.global_debug_hook:
            train_hooks.append(self.global_debug_hook)

        return classifier.evaluate(
            input_fn=self.get_dataset_func(
                split_name=split_name, num_epochs=num_epochs, shuffle=False, batch_size=batch_size
            ),
            steps=steps,
            hooks=train_hooks if train_hooks else None
        )

    def test(self, batch_size=128):
        """
            预测结果
        :return: dict
        """

        return self._evaluate_model(split_name=DataPreProcess.SPLIT_TEST,
                                    num_epochs=1,
                                    batch_size=batch_size,
                                    steps=None)

    def predict(self, batch_size=128, call_back_func=None):
        """
            预测结果
        :return: dict
        """

        split_name = DataPreProcess.SPLIT_PREDICT
        classifier = self.get_classifier()

        train_hooks = []
        restore_hook = self.get_restore_hook()
        if restore_hook:
            train_hooks.append(restore_hook)
        if self.global_debug_hook:
            train_hooks.append(self.global_debug_hook)

        scores = classifier.predict(
            input_fn=self.get_dataset_func(
                split_name=split_name, num_epochs=1, shuffle=False, batch_size=batch_size),
            hooks=train_hooks if train_hooks else None
        )

        if call_back_func:
            return call_back_func(scores)
        else:
            return self.parse_predict_result(scores)

    def parse_predict_result(self, result_list, ):
        # for score in result_list:
        #     image_id = DataPreProcess.get_image_id(score["filename"].decode("utf-8"))
        #     _result = {"class_id": int(score["class"]), "prob": score["prob"].tolist()}
        #     result_info[image_id] = _result
        #
        # self.logger.info("result is {}".format(result_list))
        self.logger.info("result is {}".format(result_list))

    def train_with_evaluate(self, safe_max_batch_size=32, evaluate_every_epoch=2, max_epoch=100, save_every_train=1,
                            save_func=None):
        epoch_list = [evaluate_every_epoch] * int(max_epoch / evaluate_every_epoch) + [max_epoch % evaluate_every_epoch]

        num_of_train = 0
        for epoch_nums in epoch_list:
            self.train(batch_size=safe_max_batch_size, num_epochs=epoch_nums, shuffle=True, steps=None)
            num_of_train += 1
            if save_func and num_of_train % save_every_train == 0:
                save_func()

            self.evaluate(num_epochs=1, batch_size=safe_max_batch_size, steps=None)

    def get_learning_rate(self, ):
        if self.train_data_size > 0 and self.batch_size > 0 and self.learning_rate_decay_epoch_num > 0:
            decay_steps = int(self.learning_rate_decay_epoch_num * self.train_data_size / self.batch_size)
        else:
            decay_steps = self.learning_rate_decay_steps

        return tf.train.exponential_decay(
            self.learning_rate,  # Base learning rate.
            tf.train.get_global_step(),  # Current index into the dataset.
            decay_steps,  # Decay step.
            0.995,  # Decay rate.
            staircase=True)


class ProcessMode(Enum):
    train = 0
    test = 1


class OptimizerType(Enum):
    adam = 0
    sgd = 1


class Params(object):
    def __init__(self, _dict):
        self.__dict__.update(_dict)


class FeatureExtractor(object):
    logger = logging.getLogger("FeatureExtractor")

    def __init__(self, model_name, pretrained_model_dir, output_layer_name, num_train, arg_scope, preprocess_image_fn,
                 net_model_fn, feature_dir="./", ):
        self.model_name = model_name
        self.output_layer_name = output_layer_name
        self.num_train = num_train
        self.feature_dir = feature_dir
        self.pretrained_model_dir = pretrained_model_dir

        self._arg_scope = arg_scope
        self._preprocess_image_fn = preprocess_image_fn
        self._net_model_fn = net_model_fn

        self.feature_train_file_key_1 = 'feature_list'
        self.feature_train_file_key_2 = 'file_list'
        self.feature_train_file_key_3 = 'label_list'
        self._split_name_list = [DataPreProcess.SPLIT_TRAIN, DataPreProcess.SPLIT_VALIDATION,
                                 DataPreProcess.SPLIT_PREDICT]

        # model net
        self.image_size = self._net_model_fn.default_image_size
        self._features_images_key = "images"
        self._features_filename_key = "filenames"
        self.NUM_OF_CLASS = 1000

    def extract_feature(self, batch_size=64):
        # feature file list 判断
        feature_file_list = [os.path.join(self.feature_dir, "{}.h5".format(split_name))
                             for split_name in self._split_name_list]
        feature_file_exists = True
        for feature_file in feature_file_list:
            if not os.path.exists(feature_file):
                feature_file_exists = False
                break

        # 生成feature
        if feature_file_exists is False:
            for split_name in self._split_name_list:
                start_time = time.time()
                self.predict(split_name, batch_size=batch_size)
                self.logger.info(
                    "extracted feature for {}: cost {} seconds!".format(split_name, time.time() - start_time)
                )

    def load_dataset(self, split_name):
        feature_file = os.path.join(self.feature_dir, "{}.h5".format(split_name))
        if not os.path.exists(feature_file):
            raise Exception("{}.h5 not exists!".format(split_name))

        with h5py.File(feature_file, 'r') as h5f:
            feature_list = h5f[self.feature_train_file_key_1][:]
            label_list = h5f[self.feature_train_file_key_2][:]
            file_name_list = h5f[self.feature_train_file_key_3][:]

        return feature_list, file_name_list, label_list

    def predict(self, split_name, batch_size=64):
        """
            训练
        :return:
        """

        def eval_decode(example):
            return TfRecordUtils.decode(example, False, self._preprocess_image_fn, self.image_size, self.image_size)

        def train_decode(example):
            return TfRecordUtils.decode(example, True, self._preprocess_image_fn, self.image_size, self.image_size)

        def get_eval_feature_and_label():

            tf_record_files_list = DataPreProcess.instance().list_tf_record_files(split_name)
            dataset = tf.data.TFRecordDataset(tf_record_files_list)

            dataset = dataset.map(eval_decode, num_parallel_calls=2)

            dataset = dataset.batch(batch_size)

            iterator = dataset.make_one_shot_iterator()

            return iterator.get_next()

        def get_train_feature_and_label():

            tf_record_files_list = DataPreProcess.instance().list_tf_record_files(split_name)
            dataset = tf.data.TFRecordDataset(tf_record_files_list)

            dataset = dataset.map(train_decode, num_parallel_calls=2)

            dataset = dataset.batch(batch_size)

            dataset = dataset.repeat(self.num_train)

            iterator = dataset.make_one_shot_iterator()

            return iterator.get_next()

        def calc_features(get_feature_and_label_fn, ):
            with tf.Graph().as_default() as graph:
                features, labels = get_feature_and_label_fn()
                input_images = tf.reshape(features[self._features_images_key],
                                          [-1, self.image_size, self.image_size, 3])
                files = features[self._features_filename_key]

                # load net
                with slim.arg_scope(self._arg_scope()):
                    net, end_points = self._net_model_fn(input_images,
                                                         num_classes=self.NUM_OF_CLASS, is_training=False)
                    feature_layer = self.create_feature_layer(net, end_points)

                with tf.Session(graph=graph) as sess:
                    # init
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())
                    init_func = get_init_fn(self.pretrained_model_dir)
                    init_func(sess)

                    # calc feature
                    _feature_list, _file_list, _label_list = [], [], []
                    try:
                        while True:
                            feature_list, file_list, label_list = sess.run([feature_layer, files, labels])
                            _feature_list.extend(feature_list)
                            _label_list.extend(label_list)
                            _file_list.extend(file_list)
                    except tf.errors.OutOfRangeError:
                        pass

                    return _feature_list, _file_list, _label_list

        all_feature_list, all_file_list, all_label_list = calc_features(get_eval_feature_and_label)

        if split_name == DataPreProcess.SPLIT_TRAIN:
            temp_feature_list, temp_file_list, temp_label_list = calc_features(get_train_feature_and_label)
            all_feature_list.extend(temp_feature_list)
            all_file_list.extend(temp_file_list)
            all_label_list.extend(temp_label_list)

        # shuffle
        shuffle_list = list(range(len(all_label_list)))
        random.shuffle(shuffle_list)
        shuffle_feature_list = [all_feature_list[index] for index in shuffle_list]
        shuffle_file_list = [all_file_list[index] for index in shuffle_list]
        shuffle_label_list = [all_label_list[index] for index in shuffle_list]

        # save feature
        feature_file = os.path.join(self.feature_dir, "{}.h5".format(split_name))
        with h5py.File(feature_file, "w") as h5f:
            h5f.create_dataset(self.feature_train_file_key_1, data=shuffle_feature_list)
            h5f.create_dataset(self.feature_train_file_key_2, data=shuffle_label_list)
            h5f.create_dataset(self.feature_train_file_key_3, data=shuffle_file_list)

    def create_feature_layer(self, net, end_points):
        return end_points[self.output_layer_name]


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


__all__ = ("mkdir_path", "Params", "ProcessMode", "estimator_iter_process", "FeatureExtractor",
           "colab_save_file_func", "OptimizerType", "DataPreProcess", "AbstractEstimator")
