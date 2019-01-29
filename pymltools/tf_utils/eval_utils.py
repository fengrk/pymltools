# -*- coding:utf-8 -*-
from __future__ import absolute_import

""" 
MAP@5 
ref: [Explanation of MAP5 scoring metric](https://www.kaggle.com/pestipeti/explanation-of-map5-scoring-metric)
Author: [Peter](https://www.kaggle.com/pestipeti)
"""
import numpy as np


def map_per_image(label, predictions, k: int = 5):
    """Computes the precision score of one image.

    Parameters
    ----------
    label : string
            The true label of the image
    predictions : list
            A list of predicted elements (order does matter, k predictions allowed per image)

    k : int
            MAP@k

    Returns
    -------
    score : double
    """
    try:
        return 1 / (predictions[:k].index(label) + 1)
    except ValueError:
        return 0.0


def map_per_set(labels, predictions, k: int = 5):
    """Computes the average over multiple images.

    Parameters
    ----------
    labels : list
             A list of the true labels. (Only one true label per images allowed!)
    predictions : list of list
             A list of predicted elements (order does matter, k predictions allowed per image)
    k : int
            MAP@k

    Returns
    -------
    score : double
    """
    return np.mean([map_per_image(l, p, k) for l, p in zip(labels, predictions)])


__all__ = ("map_per_image", "map_per_set")
