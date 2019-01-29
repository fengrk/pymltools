# -*- coding:utf-8 -*-
from __future__ import absolute_import

import logging

import numpy as np

from pyxtools import get_md5, create_guid


class ClassifierConfig(object):
    def __init__(self, classifier_name: str, weight, result_dict: dict):
        self.classifier_name = classifier_name
        self.weight = weight
        self.result_dict = result_dict

    @property
    def result_len(self):
        return len(self.result_dict)

    @property
    def result_key_md5(self):
        keys_list = list(self.result_dict.keys())
        keys_list.sort()
        return get_md5(",".join([str(key) for key in keys_list]).encode("utf-8"))


class ClassifierVoter(object):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.classifier_results = {}
        self.final_result = {}
        self._key_md5 = None

    def add_classifier_result(self, result_dict, weight: int = 1, classifier: str = None):
        if not isinstance(weight, int) or weight < 0:
            raise ValueError("weight must be int and >= 0!")

        if classifier is None:
            classifier = get_md5(create_guid().encode("utf-8"))

        if classifier in self.classifier_results:
            self.logger.warning("result of {} is replaced".format(classifier))

        classifier_result = ClassifierConfig(classifier_name=classifier,
                                             weight=weight,
                                             result_dict=result_dict)

        result_key_md5 = classifier_result.result_key_md5

        if self._key_md5 is None:
            self._key_md5 = result_key_md5

        if result_key_md5 != self._key_md5:
            raise ValueError("keys in result_dict not same as other classifier result!")

        self.classifier_results[classifier] = classifier_result

    def voting(self):
        if not self.classifier_results:
            raise Exception("not classifier found!")

        key_list = None
        for classifier_result in self.classifier_results.values():
            key_list = classifier_result.result_dict.keys()
            break

        final_result = {}
        for key in key_list:
            voter_result = []
            for classifier_result in self.classifier_results.values():
                voter_result.extend([classifier_result.result_dict[key]] * classifier_result.weight)
            final_result[key] = max(voter_result, key=voter_result.count)

        self.final_result = final_result
        return self.final_result


class ProbVoter(object):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.classifier_results = {}
        self.final_result = {}
        self._key_md5 = None

    def add_classifier_result(self, result_dict, weight: float = 1.0, classifier: str = None):
        if not isinstance(weight, float) or weight < 0:
            raise ValueError("weight must be float and >= 0!")

        if classifier is None:
            classifier = get_md5(create_guid().encode("utf-8"))

        if classifier in self.classifier_results:
            self.logger.warning("result of {} is replaced".format(classifier))

        classifier_result = ClassifierConfig(classifier_name=classifier,
                                             weight=weight,
                                             result_dict=result_dict)

        result_key_md5 = classifier_result.result_key_md5

        if self._key_md5 is None:
            self._key_md5 = result_key_md5

        if result_key_md5 != self._key_md5:
            raise ValueError("keys in result_dict not same as other classifier result!")

        self.classifier_results[classifier] = classifier_result

    def voting(self):
        if not self.classifier_results:
            raise Exception("not classifier found!")

        key_list = None
        for classifier_result in self.classifier_results.values():
            key_list = classifier_result.result_dict.keys()
            break

        final_result = {}
        for key in key_list:
            voter_result = []
            for classifier_result in self.classifier_results.values():
                voter_result.append([prob * classifier_result.weight for prob in classifier_result.result_dict[key]])

            final_result[key] = np.average(np.array(voter_result), axis=0).tolist()

        self.final_result = final_result
        return self.final_result


class KaggleVoterUtils(object):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def classifier_voter(self, input_file_list, target_output_file, weight_list=None, ):
        head_line = None
        with open(input_file_list[0], "r", encoding="utf-8") as fr:
            for line in fr:
                head_line = line.rstrip()
                break

        if head_line is None:
            raise Exception("{} is wrong!".format(input_file_list[0]))

        if weight_list is None:
            weight_list = [1] * len(input_file_list)

        voter_tool = ClassifierVoter()

        for index, input_file in enumerate(input_file_list):
            voter_tool.add_classifier_result(weight=weight_list[index],
                                             result_dict=self._parse_classifier_result(input_file))

        result_dict = voter_tool.voting()

        with open(target_output_file, "w", encoding="utf-8") as fw:
            fw.write(head_line + "\n")
            all_line = ("{},{}".format(key, value) for key, value in result_dict.items())
            fw.write("\n".join(all_line) + "\n")

        self.logger.info("success to combine result to {}".format(target_output_file))

    @staticmethod
    def _parse_classifier_result(result_csv):
        with open(result_csv, "r", encoding="utf-8") as fw:
            fw.readline()
            result = {}

            for line in fw:
                if line:
                    list_line = line.rstrip().split(",")
                    result[list_line[0]] = list_line[1]

            return result

    def prob_voter(self, input_file_list, target_output_file, weight_list=None, ):
        # head title
        head_line = None
        with open(input_file_list[0], "r", encoding="utf-8") as fr:
            for line in fr:
                head_line = line.rstrip()
                break

        if head_line is None:
            raise Exception("{} is wrong!".format(input_file_list[0]))

        # init
        if weight_list is None:
            weight_list = [1.0] * len(input_file_list)

        voter_tool = ProbVoter()

        for index, input_file in enumerate(input_file_list):
            voter_tool.add_classifier_result(weight=weight_list[index],
                                             result_dict=self._parse_classifier_result(input_file))
        # voting
        result_dict = voter_tool.voting()

        # output
        with open(target_output_file, "w", encoding="utf-8") as fw:
            fw.write(head_line + "\n")

            all_line = ("{},{}".format(key, ",".join([str(prob) for prob in prob_list]))
                        for key, prob_list in result_dict.items())
            fw.write("\n".join(all_line) + "\n")

        self.logger.info("success to combine result to {}".format(target_output_file))

    @staticmethod
    def _parse_prob_result(result_csv):
        with open(result_csv, "r", encoding="utf-8") as fw:
            fw.readline()
            result = {}

            for line in fw:
                if line:
                    list_line = line.rstrip().split(",")
                    key = list_line.pop(0)
                    result[key] = [float(value) for value in list_line]

            return result


__all__ = ("ClassifierConfig", "ClassifierVoter", "ProbVoter", "KaggleVoterUtils")
