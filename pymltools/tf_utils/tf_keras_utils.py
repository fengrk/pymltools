# -*- coding:utf-8 -*-
from __future__ import absolute_import

import logging

import numpy as np
import os
import random
import shutil
import tempfile
import tensorflow as tf
from tensorflow.python.estimator.estimator import WarmStartSettings
from tensorflow.python.keras import losses, optimizers, utils
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import load_model

from .init_hook_utils import InitFromPretrainedCheckpointHook
from pyxtools import create_guid


class MnistKerasDemo(object):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.keras_model_file = "./keras-model.model"
        self.tf_dir = "./tf"

        self.batch_size = 128
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 28, 28, 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 28, 28, 1)

        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255
        self.logger.info('x_train shape: {}'.format(self.x_train.shape))
        self.logger.info('train samples: {}'.format(self.x_train.shape[0]))
        self.logger.info('test samples: {}'.format(self.x_test.shape[0]))

        # convert class vectors to binary class matrices
        self.y_train = utils.to_categorical(self.y_train, 10)
        self.y_test = utils.to_categorical(self.y_test, 10)

    def get_bone_net(self):
        def model_net():
            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(10, activation='softmax'))
            return model

        return model_net

    def print_tf_estimator(self, init_mode: str = "warm") -> float:
        def model_func(features, labels, mode):
            branch_model = self.get_bone_net()()
            inputs = tf.reshape(features["image"], [-1, 28, 28, 1])
            output = branch_model(inputs)

            # Compute predictions.
            if mode == tf.estimator.ModeKeys.PREDICT:
                # Compute loss.
                predictions = {
                    "label": features["label"],  # predict have no label
                    "predict": output,
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

            # Compute loss.
            loss = tf.losses.mean_squared_error(labels=labels, predictions=output)

            # Create training op.
            if mode == tf.estimator.ModeKeys.TRAIN:
                # 全网络 优化器
                learning_rate = 1e-4
                # fc 优化器
                fc_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                fc_train_op = fc_optimizer.minimize(loss,
                                                    global_step=tf.train.get_global_step(),
                                                    var_list=None)

                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=fc_train_op, scaffold=None)

            # Compute evaluation metrics.
            eval_metric_ops = {
                'accuracy': tf.metrics.accuracy(labels=labels, predictions=output),
            }
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=eval_metric_ops)

        if os.path.exists(self.tf_dir):
            shutil.rmtree(self.tf_dir)

        assert not os.path.exists(self.tf_dir)
        os.mkdir(self.tf_dir)

        log_dir = os.path.join(self.tf_dir, "tmp")
        checkpoint_path = keras_convert_model_to_estimator_ckpt(keras_model_path=self.keras_model_file, log_dir=log_dir,
                                                                logger=self.logger)
        if not checkpoint_path:
            raise ValueError("checkpoint_path cannot be null!")

        if init_mode == "warm":
            classifier = tf.estimator.Estimator(model_fn=model_func, model_dir=self.tf_dir,
                                                warm_start_from=keras_warm_start(checkpoint_path))
            scores = classifier.predict(input_fn=tf.estimator.inputs.numpy_input_fn(
                x={"image": self.x_test, "label": self.y_test},
                batch_size=self.batch_size, num_epochs=1, shuffle=False, num_threads=1
            ))
        else:
            classifier = tf.estimator.Estimator(model_fn=model_func, model_dir=self.tf_dir)
            scores = classifier.predict(input_fn=tf.estimator.inputs.numpy_input_fn(
                x={"image": self.x_test, "label": self.y_test},
                batch_size=self.batch_size, num_epochs=1, shuffle=False, num_threads=1
            ), hooks=[InitFromPretrainedCheckpointHook(checkpoint_path, exclusion_list=["global_step"])])

        result_list = []
        for score in scores:
            result_list.append(bool(np.argmax(score["label"]) == np.argmax(score["predict"])))

        accuracy = sum(result_list) / len(result_list)
        self.logger.info('TF Test accuracy: {}'.format(accuracy))
        return accuracy

    def keras_train(self, ):
        model = self.get_bone_net()()
        model.compile(loss=losses.categorical_crossentropy,
                      optimizer=optimizers.Adadelta(),
                      metrics=['accuracy'])

        if os.path.exists(self.keras_model_file):
            self.logger.info("load weight from {}".format(os.path.abspath(self.keras_model_file)))
            model.set_weights(load_model(self.keras_model_file).get_weights())

        model.fit(self.x_train, self.y_train,
                  batch_size=self.batch_size,
                  epochs=1,
                  verbose=1,
                  validation_data=(self.x_test, self.y_test))
        model.save(self.keras_model_file)

        self.print_keras_model(model=model)

    def print_keras_model(self, model=None) -> float:
        if model is None:
            model = self.get_bone_net()()
            model.compile(loss=losses.categorical_crossentropy,
                          optimizer=optimizers.Adadelta(),
                          metrics=['accuracy'])

        self.logger.info("load weight from {}".format(os.path.abspath(self.keras_model_file)))
        model.set_weights(load_model(self.keras_model_file).get_weights())
        score = model.evaluate(self.x_test, self.y_test, verbose=0)
        self.logger.info('Test loss: {}'.format(score[0]))
        self.logger.info('Test accuracy: {}'.format(score[1]))
        return score[1]

    def test_equal(self, count: int = 1, tf_mode: str = "warm"):
        self.logger.info("testing: count {}, mode {}...".format(count, tf_mode))
        x_train = self.x_train.copy()
        y_train = self.y_train.copy()
        _train_length = len(x_train) // 128

        for i in range(count):
            _count = random.choice([ii for ii in range(_train_length)])
            self.x_train = x_train[_count * 128:(_count + 1) * 128]
            self.y_train = y_train[_count * 128:(_count + 1) * 128]
            self.keras_train()
            assert self.print_tf_estimator(init_mode=tf_mode) == self.print_keras_model()

        self.logger.info("testing success: count {}, mode {}...".format(count, tf_mode))

    def test_main(self):
        """
            todo 测试无法通过: print_tf_estimator 只能执行一次
        Returns:

        """
        for count in [1, 3]:
            for mode in ["warm", "init"]:
                try:
                    self.test_equal(count=count, tf_mode=mode)
                except Exception as e:
                    self.logger.error(e)


def keras_warm_start(ckpt_file: str, ) -> WarmStartSettings:
    return WarmStartSettings(ckpt_to_initialize_from=ckpt_file)


def keras_convert_model_to_estimator_ckpt(keras_model_path: str, keras_model=None, log_dir: str = None,
                                          logger=logging.getLogger(__name__), ):
    if log_dir is None:
        log_dir = os.path.join(tempfile.gettempdir(), "keras-{}".format(create_guid()))

    if not os.path.exists(keras_model_path) or not os.path.isfile(keras_model_path):
        raise ValueError("keras_model_or_file {} not exist!".format(keras_model_path))

    keras_dir = os.path.join(log_dir, "keras")
    if tf.train.latest_checkpoint(keras_dir):
        logger.warning("checkpoint file exist before before! Just return it!")
        return tf.train.latest_checkpoint(keras_dir)

    _config = tf.estimator.RunConfig(model_dir=log_dir)
    logger.info("loading weight from tensorflow.python.keras model: {}".format(os.path.abspath(keras_model_path)))
    try:
        if keras_model is None:
            tf.keras.estimator.model_to_estimator(keras_model_path=keras_model_path, config=_config)
        else:
            keras_model.set_weights(load_model(keras_model_path).get_weights())
            tf.keras.estimator.model_to_estimator(keras_model=keras_model, config=_config)

        logger.info("success to load weight from tensorflow.python.keras model!")
        return tf.train.latest_checkpoint(keras_dir)
    except Exception as e:
        logger.error("fail to load weight from tensorflow.python.keras model, error is {}".format(e), exc_info=True)

    return None


__all__ = ("keras_convert_model_to_estimator_ckpt", "MnistKerasDemo", "keras_warm_start")
