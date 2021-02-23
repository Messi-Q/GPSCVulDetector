import os
import torch
import numpy as np

"""
Here is the method for extracting security patterns of timestamp dependence.
"""

patterns = {"pattern1": 1, "pattern2": 2, "pattern3": 3}

patterns_flag = {"100", "010", "001"}


# split all functions of contracts
def split_function(filepath):
    function_list = []
    f = open(filepath, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    flag = -1

    for line in lines:
        text = line.strip()
        if len(text) > 0 and text != "\n":
            if text.split()[0] == "function" or text.split()[0] == "function()":
                function_list.append([text])
                flag += 1
            elif len(function_list) > 0 and ("function" in function_list[flag][0]):
                function_list[flag].append(text)

    return function_list


# Position the call.value to generate the graph
def extract_pattern(filepath):
    allFunctionList = split_function(filepath)  # Store all functions
    timeStampList = []  # Store all W functions that call call.value
    otherFunctionList = []  # Store functions other than W functions
    pattern_list = []

    # Store other functions without W functions (with block.timestamp)
    for i in range(len(allFunctionList)):
        flag = 0
        for j in range(len(allFunctionList[i])):
            text = allFunctionList[i][j]
            if 'block.timestamp' in text:
                timeStampList.append(allFunctionList[i])
                flag += 1
        if flag == 0:
            otherFunctionList.append(allFunctionList[i])

    ################   pattern 1: timestampInvocation  #######################
    if len(timeStampList) != 0:
        pattern_list.append(1)
    else:
        pattern_list.append(0)
        pattern_list.append(0)
        pattern_list.append(0)

    ################   pattern 2: timestampAssign      #######################
    for i in range(len(timeStampList)):
        TimestampFlag1 = 0
        VarTimestamp = None

        if len(pattern_list) > 1:
            break

        for j in range(len(timeStampList[i])):
            text = timeStampList[i][j]
            if 'block.timestamp' in text:
                TimestampFlag1 += 1
                VarTimestamp = text.split("=")[0]
            elif TimestampFlag1 != 0:
                if VarTimestamp != " " or "":
                    if VarTimestamp in text:
                        pattern_list.append(1)
                        break
                    elif j + 1 == len(timeStampList[i]) and len(pattern_list) == 1:
                        pattern_list.append(0)
                else:
                    pattern_list.append(0)
                    break

    ################  pattern 3: timestampContamination  #######################
    for i in range(len(timeStampList)):
        TimestampFlag2 = 0
        VarTimestamp = None

        if len(pattern_list) > 2:
            break

        for j in range(len(timeStampList[i])):
            text = timeStampList[i][j]
            if 'block.timestamp' in text:
                VarTimestamp = text.split("=")[0]
                TimestampFlag2 += 1
                if 'return' in text:
                    pattern_list.append(1)
                    break
            elif TimestampFlag2 != 0:
                if VarTimestamp in text and 'return' in text:
                    pattern_list.append(1)
                    break
                elif j + 1 == len(timeStampList[i]) and len(pattern_list) == 2:
                    pattern_list.append(0)

    return pattern_list


def extract_feature_by_fnn(outputPathFC, pattern1, pattern2, pattern3):
    pattern1 = torch.Tensor(pattern1)
    pattern2 = torch.Tensor(pattern2)
    pattern3 = torch.Tensor(pattern3)
    model = FFNNP(4, 100, 250)

    pattern1FC = model(pattern1).detach().numpy().tolist()
    pattern2FC = model(pattern2).detach().numpy().tolist()
    pattern3FC = model(pattern3).detach().numpy().tolist()
    pattern_final = np.array([pattern1FC, pattern2FC, pattern3FC])

    np.savetxt(outputPathFC, pattern_final, fmt="%.6f")


if __name__ == "__main__":
    # pattern1 = [1, 0, 0]
    # pattern2 = [0, 1, 0]
    # pattern3 = [0, 0, 1]
    # label1 = None
    # test_contract = "../data/timestamp/source_code/15.sol"
    # pattern_list = extract_pattern(test_contract)
    # if len(pattern_list) == 3:
    #     if pattern_list[0] == 1:
    #         if pattern_list[1] == 0 and pattern_list[2] == 0:
    #             label1 = 0
    #         else:
    #             label1 = 1
    #     else:
    #         label1 = 0
    # else:
    #     print("The extracted patterns are error!")
    #
    # pattern1.append(pattern_list[0])
    # pattern2.append(pattern_list[1])
    # pattern3.append(pattern_list[2])

    label = None
    inputFileDir = "../data/timestamp/source_code/"
    outputfeatureDir = "../pattern_feature/featurezeropadding/timestamp/"
    outputlabelDir = "../pattern_feature/label_by_autoextractor/timestamp/"
    dirs = os.listdir(inputFileDir)
    for file in dirs:
        pattern1 = [1, 0, 0]
        pattern2 = [0, 1, 0]
        pattern3 = [0, 0, 1]

        print(file)
        inputFilePath = inputFileDir + file
        name = file.split(".")[0]
        pattern_list = extract_pattern(inputFilePath)
        if len(pattern_list) == 3:
            if pattern_list[0] == 1:
                if pattern_list[1] == 0 and pattern_list[2] == 0:
                    label = 0
                else:
                    label = 1
            else:
                label = 0
        else:
            print("The extracted patterns are error!")

        pattern1.append(pattern_list[0])
        pattern2.append(pattern_list[1])
        pattern3.append(pattern_list[2])

        pattern1 = np.array(pattern1)
        pattern1 = np.array(np.pad(pattern1, (0, 246), 'constant'))
        pattern2 = np.array(pattern2)
        pattern2 = np.array(np.pad(pattern2, (0, 246), 'constant'))
        pattern3 = np.array(pattern3)
        pattern3 = np.array(np.pad(pattern3, (0, 246), 'constant'))

        pattern_final = np.array([pattern1, pattern2, pattern3])
        outputPath = outputfeatureDir + name + ".txt"
        np.savetxt(outputPath, pattern_final, fmt="%.6f")

        outputlabelPath = outputlabelDir + file
        f_outlabel = open(outputlabelPath, 'a')
        f_outlabel.write(str(label))
