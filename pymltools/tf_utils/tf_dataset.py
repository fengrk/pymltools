# -*- coding:utf-8 -*-
from __future__ import absolute_import

import logging
import math
import pickle
import sys

import os
import random
import tempfile
import tensorflow as tf

from .tf_record_utils import TfRecordUtils
from pyxtools import random_choice, show_images, get_image, np_image_to_pil_image


def get_int_count(float_count: float) -> int:
    int_part = int(float_count)
    if random.random() <= (float_count - int_part):
        int_part += 1
    return int_part


def balance_class_dict(info: dict, target_count: int) -> dict:
    new_info = {}
    for class_id, class_list in info.items():
        class_count = len(class_list)
        if class_count >= target_count:
            new_info[class_id] = random_choice(class_list, k=target_count, unique=True)
        else:
            new_info[class_id] = list(class_list)
            new_info[class_id].extend(random_choice(class_list, k=target_count - class_count, unique=False))

    return new_info


class DatasetUtils(object):
    """
        处理数据集
    """
    logger = logging.getLogger("DatasetUtils")
    tf_record_file_pattern = "TF_%s_%05d-of-%05d.tfrecord"

    IMAGE_ID_SEP = "---"
    SPLIT_TRAIN = "train"
    SPLIT_VALIDATION = "validation"
    SPLIT_TEST = "test"
    SPLIT_PREDICT = "predict"

    def __init__(self, tf_record_dir: str, num_shards: int, file_name_list: list,
                 label_id_name_tuple: tuple, get_class_id_func):
        self.tf_record_dir = tf_record_dir
        self.NUM_SHARDS = num_shards
        self.get_class_id_func = get_class_id_func
        self.file_list_dict = {}
        self.dict_class_id_vs_name = {}
        self.dict_class_name_vs_id = {}

        self.set_file_name_list(file_name_list)
        self.set_label_id_name(label_id_name_tuple)

    def set_file_name_list(self, file_name_list: list):
        if file_name_list and len(file_name_list) != 4:
            raise ValueError("file_name_list is invalid: length !=4 !")

        if file_name_list:
            self.file_list_dict = {
                split_name: file_name_list[index]
                for index, split_name in
                enumerate([self.SPLIT_TRAIN, self.SPLIT_VALIDATION, self.SPLIT_TEST, self.SPLIT_PREDICT])
            }

    def set_label_id_name(self, label_id_name_tuple: tuple):
        self.dict_class_id_vs_name.clear()
        self.dict_class_name_vs_id.clear()
        for label_id, label_name in label_id_name_tuple:
            class_id = int(label_id)
            self.dict_class_id_vs_name[class_id] = label_name
            self.dict_class_name_vs_id[label_name] = class_id

    @property
    def num_of_class(self) -> int:
        assert len(self.dict_class_id_vs_name) == len(self.dict_class_name_vs_id)
        return len(self.dict_class_id_vs_name)

    def get_class_id(self, file_name: str) -> int:
        return self.get_class_id_func(file_name)

    def get_class_name(self, class_id: int) -> str:
        return self.dict_class_id_vs_name[class_id]

    def _get_dataset_tf_record_filename(self, split_name, shard_id) -> str:
        output_filename = self.tf_record_file_pattern % (
            split_name, shard_id, self.NUM_SHARDS)
        return os.path.join(self.tf_record_dir, output_filename)

    def _dataset_exists(self, ) -> bool:
        for split_name in [self.SPLIT_TRAIN, self.SPLIT_VALIDATION, self.SPLIT_TEST, self.SPLIT_PREDICT]:
            if len(self.file_list_dict[split_name]) > 0:
                for shard_id in range(self.NUM_SHARDS):
                    output_filename = self._get_dataset_tf_record_filename(split_name, shard_id)
                    if not tf.gfile.Exists(output_filename):
                        return False
        return True

    def convert_images_with_labels(self, ):
        """
            将训练集数据转为tf record文件
        :return:
        """
        if not tf.gfile.Exists(self.tf_record_dir):
            tf.gfile.MakeDirs(self.tf_record_dir)

        if not self.need_to_rebuild_dataset_file():
            return

        def _convert_dataset(split_name, filenames):
            assert split_name in [self.SPLIT_TRAIN, self.SPLIT_VALIDATION, self.SPLIT_TEST, self.SPLIT_PREDICT]

            self.logger.info("converting {} data({}) to tf record...".format(split_name, len(filenames)))

            if len(filenames) == 0:
                return

            num_per_shard = int(math.ceil(len(filenames) / float(self.NUM_SHARDS)))

            with tf.Graph().as_default():
                with tf.Session(''):
                    for shard_id in range(self.NUM_SHARDS):
                        output_filename = self._get_dataset_tf_record_filename(split_name, shard_id)
                        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                            start_ndx = shard_id * num_per_shard
                            end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
                            for i in range(start_ndx, end_ndx):
                                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                                    i + 1, len(filenames), shard_id))
                                sys.stdout.flush()

                                _class_id = self.get_class_id(filenames[i])
                                example = TfRecordUtils.encode(filenames[i], _class_id)
                                tfrecord_writer.write(example.SerializeToString())

                            sys.stdout.write('\n')
                            sys.stdout.flush()

        # First, convert the training, validation sets, test sets, predict sets
        _convert_dataset(self.SPLIT_TRAIN, self.file_list_dict[self.SPLIT_TRAIN])
        _convert_dataset(self.SPLIT_VALIDATION, self.file_list_dict[self.SPLIT_VALIDATION])
        _convert_dataset(self.SPLIT_TEST, self.file_list_dict[self.SPLIT_TEST])
        _convert_dataset(self.SPLIT_PREDICT, self.file_list_dict[self.SPLIT_PREDICT])

        self.logger.info('\nFinished converting the dataset!')

    def list_tf_record_files(self, split_name: str) -> list:
        """

        :rtype: list of str
        """
        import fnmatch
        file_pattern = self.tf_record_file_pattern[:self.tf_record_file_pattern.find("%s")] + \
                       split_name + "*." + self.tf_record_file_pattern.split(".")[-1]

        result_list = [os.path.join(self.tf_record_dir, fn) for fn in
                       fnmatch.filter(os.listdir(self.tf_record_dir), file_pattern)]
        return result_list

    def debug_tf_parser(self):

        def show_image(_features, _labels):
            image_list = []
            for index, image in enumerate(_features["images"]):
                file_name = _features["filenames"][index].decode("utf-8")
                self.logger.info("image file is {}, class is {}, class name is correct? {}"
                                 .format(file_name,
                                         self.get_class_name(int(_labels[index])),
                                         bool(int(_labels[index] == self.get_class_id(file_name)))))
                image_list.append([
                    np_image_to_pil_image(image.reshape(200, 200, 3), mode="RGB"),
                    get_image(file_name, 200, 200)
                ])

            show_images(image_list)

        def preprocess_image(image):
            image_content = tf.image.decode_jpeg(image, channels=3)

            return image_content

        def decode(example):
            _features, _labels = TfRecordUtils.decode_v2(example, process_image_fn=preprocess_image)
            _features["images"] = tf.image.resize_images(_features["images"], [200, 200])
            return _features, _labels

        tf_record_files_list = self.list_tf_record_files(self.SPLIT_TEST)
        dataset = tf.data.TFRecordDataset(tf_record_files_list)
        dataset = dataset.map(decode, num_parallel_calls=1)
        dataset = dataset.batch(2)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()

        with tf.get_default_graph().as_default():
            with tf.Session() as sess:
                count = 5
                while count > 1:
                    show_image(*sess.run([features, labels]))
                    input("")
                    count -= 1

    def need_to_rebuild_dataset_file(self) -> bool:
        if self._dataset_exists():
            self.logger.info('Dataset files already exist. Exiting without re-creating them.')
            return False
        return True

    @classmethod
    def create_file_list(cls, percent_list: list, file_name_list: list, get_class_id_func,
                         default_count_per_class: int = None) -> [list, list, list, list]:
        """

        :param percent_list:
        :param file_name_list:
        :param get_class_id_func:
        :param default_count_per_class: 数据不均衡, 对数据过少的类补全数据至default_count_per_class
        :return:
        """
        percent_train, percent_validate, percent_test, percent_predict = percent_list

        # total image
        total_photo_filename_info = {}
        for image_file in file_name_list:
            _class_id = get_class_id_func(image_file)
            total_photo_filename_info.setdefault(_class_id, []).append(image_file)

        # predict image
        predict_filenames = []
        for _class_id, file_list in total_photo_filename_info.items():
            random.seed()
            random.shuffle(file_list)
            random.shuffle(file_list)
            _predict_index = get_int_count(len(file_list) * percent_predict)
            predict_filenames.extend(file_list[:_predict_index])

        # train validate test dataset
        photo_filename_info = {}
        for image_file in list(set(file_name_list) - set(predict_filenames)):
            _class_id = get_class_id_func(image_file)
            photo_filename_info.setdefault(_class_id, []).append(image_file)

        # 样本均衡: 过采样的方法
        class_count_list = [len(image_file) for image_file in photo_filename_info.values()]
        class_count_list.sort()
        max_count = class_count_list[-1]
        if default_count_per_class is None:
            default_count_per_class = max_count
        min_count = class_count_list[0]
        cls.logger.info("count of each class is {}! max count = {}, min count = {}!".format(
            class_count_list, max_count, min_count
        ))
        if min_count != max_count and (max_count - min_count) / min_count > 0.1:
            cls.logger.warning("样本不均衡! 对样本进行过采样处理!")
            for _class_id, _image_file_list in photo_filename_info.items():
                count = default_count_per_class - len(_image_file_list)
                while count > 0:
                    count -= 1
                    _image_file_list.append(random.choice(_image_file_list))

        training_filenames = []
        validation_filenames = []
        test_filenames = []

        # new percent
        new_percent_train = percent_train / (percent_train + percent_validate + percent_test)
        new_percent_validate = percent_validate / (percent_train + percent_validate + percent_test)
        for _class_id, file_list in photo_filename_info.items():
            random.seed()
            random.shuffle(file_list)
            random.shuffle(file_list)
            length_file_list = len(file_list)
            _train_index = get_int_count(length_file_list * new_percent_train)
            _validation_index = get_int_count(length_file_list * (new_percent_validate + new_percent_train))
            training_filenames.extend(file_list[:_train_index])
            validation_filenames.extend(file_list[_train_index:_validation_index])
            test_filenames.extend(file_list[_validation_index:])

        # shuffle
        random.seed(0)
        if training_filenames:
            random.shuffle(training_filenames)
        if validation_filenames:
            random.shuffle(validation_filenames)
        if test_filenames:
            random.shuffle(test_filenames)
        if predict_filenames:
            random.shuffle(predict_filenames)

        return [training_filenames, validation_filenames, test_filenames, predict_filenames]


def get_readable_names_for_imagenet_labels() -> dict:
    pkf = os.path.join(tempfile.gettempdir(), "get_readable_names_for_imagenet_labels.pkl")

    if os.path.exists(pkf):
        with open(pkf, "rb") as fr:
            label_maps = pickle.load(fr)

        if label_maps:
            return label_maps

    from ml_tools.datasets import imagenet
    label_maps = imagenet.create_readable_names_for_imagenet_labels()
    with open(pkf, "wb") as fw:
        pickle.dump(label_maps, fw)
    return label_maps


__all__ = ("DatasetUtils", "get_readable_names_for_imagenet_labels", "get_int_count", "balance_class_dict")
