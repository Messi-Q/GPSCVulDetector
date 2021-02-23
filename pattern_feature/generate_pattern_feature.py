import numpy as np

train_total_name_path = "../graph_feature/reentrancy/contract_name_train.txt"
valid_total_name_path = "../graph_feature/reentrancy/contract_name_valid.txt"
label_path = "./label_by_autoextractor/reentrancy/"
outputFinalLabelByExtractor = "./featurezeropadding/reentrancy/label_by_extractor_train.txt"

f = open(train_total_name_path, 'r')
lines = f.readlines()

for line in lines:
    line = line.strip('\n').split('.')[0]
    f_label = open(label_path + line + '.c', 'r')
    label = f_label.readline().strip('\n')
    f_label_out = open(outputFinalLabelByExtractor, 'a')
    f_label_out.write(label + '\n')


