import os
import numpy as np

test_total_name_path = "../graph_feature/timestamp/contract_name_valid.txt"
pattern_feature_path = "./feature_zeropadding/timestamp/"
label_path = "./label_by_extractor/timestamp/"
outputFinalFeature = "./feature_zeropadding/pattern_valid.txt"
outputFinalLabelByExtractor = "./feature_zeropadding/label_by_extractor_valid.txt"
final_pattern_feature = []

f = open(test_total_name_path, 'r')
lines = f.readlines()

for line in lines:
    line = line.strip('\n').split('.')[0]
    tmp_feature = np.loadtxt(pattern_feature_path + line + '.txt')
    final_pattern_feature.append(tmp_feature)

    # f_label = open(label_path + line + '.sol', 'r')
    # label = f_label.readline().strip('\n')
    # f_label_out = open(outputFinalLabelByExtractor, 'a')
    # f_label_out.write(label + '\n')


np.savetxt(outputFinalFeature, final_pattern_feature, fmt="%.6f")


