# -*- coding:utf-8 -*-
from __future__ import absolute_import

import logging
import math
import unittest

import numpy as np
from sklearn.preprocessing import normalize

from pyxtools import global_init_logger, NormType, calc_distance_pairs

global_init_logger()


class TestNumpy(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def testL2Norm(self):
        # ref: https://www.itread01.com/content/1541469428.html
        input_data = np.asarray([[1.0, 2, 3], [4.0, 5, 6], [7.0, 8, 9]])
        self.logger.info("shape of input data is {}".format(input_data.shape))

        result = np.linalg.norm(input_data, ord=2, axis=0)
        self.logger.info("result is {}".format(result))

        normalized = normalize(input_data, norm='l2', axis=0)  # L2
        self.logger.info("normalized is {}".format(normalized))

        self.assertEqual(normalized.shape, input_data.shape)
        self.assertEqual(normalized[0][0], input_data[0][0] / result[0])
        self.assertEqual(normalized[0][1], input_data[0][1] / result[1])

    def testL2NormV2(self):
        input_data = np.asarray([[1.0, 2, 3, 1], [4.0, 5, 6, 4], [7.0, 8, 9, 7]])
        self.assertTrue(input_data.shape == (3, 4))

        # l2 mxd, have d norm
        # norm np
        norm_list = [np.sqrt(np.dot(input_data[:, i], input_data[:, i].T)) for i in range(input_data.shape[-1])]
        norm_arr = np.linalg.norm(input_data, ord=2, axis=0)
        self.assertTrue((abs(np.asarray(norm_list) - norm_arr) < 1e-4).all())

        norm_np = np.zeros(shape=input_data.shape)
        for i in range(input_data.shape[-1]):
            norm_np[:, i] = input_data[:, i] / norm_list[i]

        self.assertTrue((abs(input_data / norm_arr - norm_np) < 1e-4).all())
        self.logger.info("norm np is {}".format(norm_np))

        # normalize
        normalize_result = normalize(input_data, norm="l2", axis=0)
        self.logger.info("normalize_result is {}".format(normalize_result))
        self.assertTrue((abs(norm_np - normalize_result) < 1e-4).all())

        # l2: mxd, have 1 norm
        # norm np
        total_norm = np.sqrt(np.sum(np.power(input_data, 2)))
        self.assertEqual(total_norm.shape, ())
        norm_np = input_data / total_norm
        self.logger.info("norm np is {}".format(norm_np))

        # normalize
        # todo not pass
        # normalize_result = normalize(input_data, norm="l2", axis=1)
        # self.logger.info("normalize_result is {}".format(normalize_result))
        #
        # self.assertEqual(normalize_result.shape, norm_np.shape)
        # self.assertTrue((abs(norm_np - normalize_result) < 1e-4).all())

    def testL2NormV3(self):
        input_data = np.asarray([[1.0, 2, 3, 1], [4.0, 5, 6, 4], [7.0, 8, 9, 7]])
        self.assertTrue(input_data.shape == (3, 4))

        # l2 mxd, have d norm
        # norm np
        norm_list = [np.sqrt(np.dot(input_data[:, i], input_data[:, i].T)) for i in range(input_data.shape[-1])]
        norm_np = np.zeros(shape=input_data.shape)
        for i in range(input_data.shape[-1]):
            norm_np[:, i] = input_data[:, i] / norm_list[i]

        self.logger.info("norm np is {}".format(norm_np))
        self.assertTrue((abs(NormType.l2.normalize(input_data) - norm_np) < 1e-4).all())

        # l2: mxd, have 1 norm
        # norm np
        total_norm = np.sqrt(np.sum(np.power(input_data, 2)))
        self.assertEqual(total_norm.shape, ())
        norm_np = input_data / total_norm
        self.logger.info("norm np is {}".format(norm_np))
        self.assertTrue((abs(norm_np - NormType.all.normalize(input_data)) < 1e-4).all())

    def testDistance(self):
        m, n, k = 5, 512, 25
        vec1 = np.random.random((m, n))
        for i in range(n):
            vec1[0][i] = 1
            vec1[1][i] = 0

        vec2 = np.concatenate((vec1, np.random.random((k, n))), axis=0)

        self.assertTrue(vec1.shape == (m, n))
        self.assertTrue(vec2.shape == (m + k, n))
        self.assertTrue((vec1 == vec2[:m]).all())
        self.assertTrue(np.sum(vec1[0]) == n)
        self.assertTrue(np.sum(vec1[1]) == 0)

        # dumpy
        distance_dumpy = np.zeros(shape=(m, m + k), dtype=np.float32)
        for i in range(m):
            for j in range(m + k):
                distance_dumpy[i][j] = np.sqrt(np.sum(np.square(vec1[i] - vec2[j])))
        self.assertTrue(distance_dumpy.shape == (m, m + k))
        self.assertTrue(distance_dumpy[0][0] == 0)
        self.assertTrue(distance_dumpy[1][1] == 0)
        self.assertTrue(abs(distance_dumpy[0][1] - math.sqrt(n)) < 1e-4)
        self.assertTrue(abs(distance_dumpy[1][0] - math.sqrt(n)) < 1e-4)

        # distance = np.zeros((m, m + k))
        distance = calc_distance_pairs(vec1, vec2)

        self.assertTrue(distance.shape == (m, m + k))
        self.assertTrue((abs(distance_dumpy - distance) < 1e-4).all())

    def testLen(self):
        a = np.random.random((5, 4))
        self.assertTrue(a.shape == (5, 4))
        self.assertTrue(len(a) == a.shape[0])

        a = np.random.random((5, 1))
        self.assertTrue(a.shape == (5, 1))
        self.assertTrue(len(a) == a.shape[0])

        a = np.random.random((5,))
        self.assertTrue(a.shape == (5,))
        self.assertTrue(len(a) == a.shape[0])
