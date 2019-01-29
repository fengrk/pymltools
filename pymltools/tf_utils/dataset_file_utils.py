# -*- coding:utf-8 -*-
from __future__ import absolute_import

import h5py


def load_data_from_h5file(h5_file, key_list):
    with h5py.File(h5_file, 'r') as h5f:
        return [h5f[key][:] for key in key_list]


def store_data_in_h5file(h5_file, data_list, key_list):
    """

    Args:
        h5_file:
        data_list: list. If str, must be ascii
        key_list:

    Returns:

    """
    with h5py.File(h5_file, "w") as h5f:
        for index, data in enumerate(data_list):
            h5f.create_dataset(key_list[index], data=data)


__all__ = ("load_data_from_h5file", "store_data_in_h5file")
