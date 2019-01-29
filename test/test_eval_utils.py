# -*- coding:utf-8 -*-
from __future__ import absolute_import
import unittest

try:
    from pymltools.tf_utils import map_per_image, map_per_set
except ImportError:
    from pymltools.pymltools.tf_utils import map_per_image, map_per_set


class TestEvalUtils(unittest.TestCase):
    def testMAP(self):
        self.assertTrue(map_per_image('x', []) == 0.0)
        self.assertTrue(map_per_image('x', ['y']) == 0.0)
        self.assertTrue(map_per_image('x', ['x']) == 1.0)
        self.assertTrue(map_per_image('x', ['x', 'y', 'z']) == 1.0)
        self.assertTrue(map_per_image('x', ['y', 'x']) == 0.5)
        self.assertTrue(map_per_image('x', ['y', 'x', 'x']) == 0.5)
        self.assertTrue(map_per_image('x', ['y', 'z']) == 0.0)
        self.assertTrue(map_per_image('x', ['y', 'z', 'x']) == 1 / 3)
        self.assertTrue(map_per_image('x', ['y', 'z', 'a', 'b', 'c']) == 0.0)
        self.assertTrue(map_per_image('x', ['x', 'z', 'a', 'b', 'c']) == 1.0)
        self.assertTrue(map_per_image('x', ['y', 'z', 'a', 'b', 'x']) == 1 / 5)
        self.assertTrue(map_per_image('x', ['y', 'z', 'a', 'b', 'c', 'x']) == 0.0)

        self.assertTrue(map_per_set(['x'], [['x', 'y']]) == 1.0)
        self.assertTrue(map_per_set(['x', 'z'], [['x', 'y'], ['x', 'y']]) == 1 / 2)
        self.assertTrue(map_per_set(['x', 'z'], [['x', 'y'], ['x', 'y', 'z']]) == 2 / 3)
        self.assertTrue(map_per_set(['x', 'z', 'k'], [['x', 'y'], ['x', 'y', 'z'], ['a', 'b', 'c', 'd', 'e']]) == 4 / 9)
