# -*- coding:utf-8 -*-
from __future__ import absolute_import

import os

from pyxtools import get_md5


class PathSource(object):
    def __init__(self, path: str, can_write: bool = True, not_exist_callable=None):
        self.path = path
        self.can_write = can_write
        self.not_exist_callable = not_exist_callable

    def init_path(self):
        if not os.path.exists(self.path) and self.not_exist_callable is not None:
            self.not_exist_callable()

    def create_path(self, *paths) -> str:
        path = os.path.join(self.path, *paths)
        if not os.path.exists(path) and not self.can_write:
            raise ValueError("not allow to create path!")
        return path

    def __hash__(self):
        return self.hash_path(self.path)

    @classmethod
    def hash_path(cls, path: str) -> str:
        return get_md5(path.encode("utf-8"))


class PathCollection(object):
    def __init__(self):
        self.paths = {}

    def add(self, *path: PathSource):
        hash_path = hash(path)
        self.paths[hash_path] = path

    def get_path_source(self, path: str) -> PathSource:
        hash_path = PathSource.hash_path(path)
        return self.paths.get(hash_path, PathSource(path=path))

    def get_path(self, path: str) -> str:
        path_obj = self.get_path_source(path)
        path_obj.init_path()
        if not os.path.exists(path_obj.path) and not path_obj.can_write:
            raise ValueError("path fail to create!")

        return path_obj.path

    def init_paths(self):
        [path.init_path() for path in self.paths.values() if path]


__all__ = ("PathSource", "PathCollection")
