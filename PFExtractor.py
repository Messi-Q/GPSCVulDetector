"""
Author: Messi-Q

Date: Created on 3:11 PM 2021-02-25
"""

import numpy as np
from parser import parameter_parser
from models.FFNN import FFNN

args = parameter_parser()


def pattern_feature_extract():
    train_total_name_path = "./graph_feature/reentrancy/contract_name_train.txt"
    valid_total_name_path = "./graph_feature/reentrancy/contract_name_valid.txt"
    pattern_feature_path = "./pattern_feature/original_pattern_feature/reentrancy/"

    final_pattern_feature_train = []  # pattern feature train
    pattern_feature_train_label_path = "./pattern_feature/original_pattern_feature/reentrancy/label_by_extractor_train.txt"
    final_pattern_feature_valid = []  # pattern feature valid
    pattern_feature_test_label_path = "./pattern_feature/original_pattern_feature/reentrancy/label_by_extractor_valid.txt"

    f_train = open(train_total_name_path, 'r')
    lines = f_train.readlines()
    for line in lines:
        line = line.strip('\n').split('.')[0]
        tmp_feature = np.loadtxt(pattern_feature_path + line + '.txt')
        tmp_feature = [y for x in tmp_feature for y in x]
        final_pattern_feature_train.append(tmp_feature)

    f_test = open(valid_total_name_path, 'r')
    lines = f_test.readlines()
    for line in lines:
        line = line.strip('\n').split('.')[0]
        tmp_feature = np.loadtxt(pattern_feature_path + line + '.txt')
        tmp_feature = [y for x in tmp_feature for y in x]
        final_pattern_feature_valid.append(tmp_feature)

    # labels of extractor
    label_by_extractor_train = []
    f_train_label_extractor = open(pattern_feature_train_label_path, 'r')
    labels = f_train_label_extractor.readlines()
    for label in labels:
        label_by_extractor_train.append(label.strip('\n'))

    label_by_extractor_valid = []
    f_test_label_extractor = open(pattern_feature_test_label_path, 'r')
    labels = f_test_label_extractor.readlines()
    for label in labels:
        label_by_extractor_valid.append(label.strip('\n'))

    pattern_train = np.array(final_pattern_feature_train)
    pattern_test = np.array(final_pattern_feature_valid)

    # The label of certain contract in training set
    y_train = []
    for i in range(len(label_by_extractor_train)):
        y_train.append(int(label_by_extractor_train[i]))
    y_train = np.array(y_train)

    # The label of certain contract in testing set
    y_test = []
    for i in range(len(label_by_extractor_valid)):
        y_test.append(int(label_by_extractor_valid[i]))
    y_test = np.array(y_test)

    model = FFNN(pattern_train, pattern_test, y_train, y_test)
    model.pattern_train_feature_extract()  # extract pattern train feature
    model.pattern_test_feature_extract()  # extract pattern test feature


if __name__ == "__main__":
    pattern_feature_extract()
