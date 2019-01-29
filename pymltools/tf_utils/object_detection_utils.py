# -*- coding:utf-8 -*-
from __future__ import absolute_import

from collections import namedtuple

import numpy as np
from PIL import Image

detection_data = namedtuple(
    "detection_data",
    ["filename", "xmin", "ymin", "xmax", "ymax", "prob"]
)


def parse_result(image: Image, file_name: str, output_dict: dict, min_score_thresh: float = 0.5) -> list:
    width, height = image.size
    result_list = []
    for i, boxes in enumerate(output_dict['detection_boxes']):
        if output_dict['detection_scores'][i] >= min_score_thresh:
            ymin, xmin, ymax, xmax = boxes
            result_list.append(
                detection_data(
                    file_name,
                    int(xmin * width), int(ymin * height),
                    int(xmax * width), int(ymax * height),
                    float(output_dict['detection_scores'][i])
                ))
        else:
            break

    return result_list


def load_image_into_numpy_array(_image):
    (im_width, im_height) = _image.size
    return np.array(_image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)
