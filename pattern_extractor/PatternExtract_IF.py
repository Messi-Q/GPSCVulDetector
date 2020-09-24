import os
import re
import torch
import numpy as np
from Simple_FC import SimpleFC

"""
Here is the method for extracting security patterns of infinite loop.
"""

# function return type
function_return_list = ['int8', 'int16', 'int32', 'int64', 'int128', 'int256', 'uint8', 'uint16', 'uint32', 'uint64',
                        'uint128', 'uint256', 'void', 'bool', 'string', 'address', "$_()", "_()"]


# split all functions of contracts
def split_function(filepath):
    function_list = []
    f = open(filepath, 'r', encoding="utf-8")
    lines = f.readlines()
    f.close()
    flag = -1

    for line in lines:
        count = 0
        text = line.rstrip()
        if len(text) > 0 and text != "\n":
            if "uint" in text.split()[0] and text.startswith("uint"):
                function_list.append([text])
                flag += 1
                continue
            elif len(function_list) > 0 and ("uint" in function_list[flag][0]):
                for types in function_return_list:
                    if text.startswith(types):
                        count += 1
                if count == 0:
                    function_list[flag].append(text)
                    continue
            if "int" in text.split()[0] and text.startswith("int"):
                function_list.append([text])
                flag += 1
                continue
            elif len(function_list) > 0 and ("int" in function_list[flag][0]):
                for types in function_return_list:
                    if text.startswith(types):
                        count += 1
                if count == 0:
                    function_list[flag].append(text)
                    continue
            if "void" in text and text.startswith("void"):
                function_list.append([text])
                flag += 1
                continue
            elif len(function_list) > 0 and ("void" in function_list[flag][0]):
                for types in function_return_list:
                    if text.startswith(types):
                        count += 1
                if count == 0:
                    function_list[flag].append(text)
                    continue
            if "bool" in text and text.startswith("bool"):
                function_list.append([text])
                flag += 1
                continue
            elif len(function_list) > 0 and ("bool" in function_list[flag][0]):
                for types in function_return_list:
                    if text.startswith(types):
                        count += 1
                if count == 0:
                    function_list[flag].append(text)
                    continue
            if "string" in text and text.startswith("string"):
                function_list.append([text])
                flag += 1
                continue
            elif len(function_list) > 0 and ("string" in function_list[flag][0]):
                for types in function_return_list:
                    if text.startswith(types):
                        count += 1
                if count == 0:
                    function_list[flag].append(text)
                    continue
            if "address" in text and text.startswith("address"):
                function_list.append([text])
                flag += 1
                continue
            elif len(function_list) > 0 and ("address" in function_list[flag][0]):
                for types in function_return_list:
                    if text.startswith(types):
                        count += 1
                if count == 0:
                    function_list[flag].append(text)
                    continue
            if "$_()" in text and text.startswith("$_()"):
                function_list.append([text])
                flag += 1
                continue
            elif len(function_list) > 0 and ("$_()" in function_list[flag][0]):
                for types in function_return_list:
                    if text.startswith(types):
                        count += 1
                if count == 0:
                    function_list[flag].append(text)
                    continue
            if "_()" in text and text.startswith("_()"):
                function_list.append([text])
                flag += 1
                continue
            elif len(function_list) > 0 and ("_()" in function_list[flag][0]):
                for types in function_return_list:
                    if text.startswith(types):
                        count += 1
                if count == 0:
                    function_list[flag].append(text)
                    continue

    return function_list


# Position the call.value to generate the graph
def extract_pattern(filepath):
    allFunctionList = split_function(filepath)  # Store all functions
    functionNameList = []  # Store all functions' name
    otherFunctionList = []
    pattern_list = []
    fallbackList = []
    LoopFlag1 = 0
    LoopFlag2 = 0
    LoopFlag3 = 0
    selfcallflag = 0

    for i in range(len(allFunctionList)):
        tmp = re.compile(".*?(?=\\()")
        funTypeAndName = tmp.match(allFunctionList[i][0]).group()
        if funTypeAndName != "$_" and funTypeAndName != "_":
            result = funTypeAndName.split(" ")
            functionNameList.append(result[1])
        else:
            functionNameList.append("Fallback")

    # label_by_extractor node_list
    for i in range(len(functionNameList)):
        if functionNameList[i] == "Fallback":
            fallbackList.append(allFunctionList[i])
        else:
            otherFunctionList.append(allFunctionList[i])

    ################   pattern 1  #######################
    if len(fallbackList) != 0:
        for i in range(len(fallbackList[0])):
            text = fallbackList[0][i]
            for k in range(len(functionNameList)):
                if functionNameList[k] in text:
                    LoopFlag1 += 1
                    pattern_list.append(1)
            if LoopFlag1 == 0 and i + 1 == len(fallbackList[0]) and len(pattern_list) == 0:
                pattern_list.append(0)
    else:
        pattern_list.append(0)
        print("There is no fallback funnction!")

    ################   pattern 2  #######################
    for i in range(len(allFunctionList)):
        currentProcessedFunctionName = functionNameList[i]  # current function name
        if selfcallflag != 0:
            break
        for j in range(1, len(allFunctionList[i])):
            text = allFunctionList[i][j]
            text = text.replace(" ", "")
            if currentProcessedFunctionName + "(" in text:
                pattern_list.append(1)
                break
    if selfcallflag == 0:
        pattern_list.append(0)

    ################   pattern 3  #######################
    for i in range(len(otherFunctionList)):
        ForFlag = 0

        if len(pattern_list) > 2:
            break

        for j in range(len(otherFunctionList[i])):
            text = allFunctionList[i][j]
            text_value = re.findall('[a-zA-Z0-9]+', text)
            if "for" in text_value:
                ForFlag += 1
                result = re.findall('[(](.*?)[)]', text)[0].split(";")
                result_value = re.sub("\D", "", result[1])

                if (("<" or "<=") in result[1]) and (("--" or "-=") in result[2]):
                    LoopFlag2 += 1
                    pattern_list.append(1)
                    break
                elif ((">" or ">=") in result[1]) and (("++" or "+=") in result[2]):
                    LoopFlag2 += 1
                    pattern_list.append(1)
                    break
                elif (result[0] == "" or " ") and (result[1] == "" or " ") and (result[2] == "" or " "):
                    LoopFlag2 += 1
                    pattern_list.append(1)
                    break
                # uint8: the max value is 255, uint16: the max value is 65535; the max value is 4294967295
                elif result_value != "":
                    if "uint8" in result[0] and int(result_value) > 255:
                        LoopFlag2 += 1
                        pattern_list.append(1)
                        break
                    elif "uint16" in result[0] and int(result_value) > 65535:
                        LoopFlag2 += 1
                        pattern_list.append(1)
                        break
                    elif "uint32" in result[0] and int(result_value) > 4294967295:
                        LoopFlag2 += 1
                        pattern_list.append(1)
                        break
                elif j + 1 == len(otherFunctionList[i]) and len(pattern_list) == 1:
                    pattern_list.append(0)
            elif ForFlag == 0 and j + 1 == len(otherFunctionList[i]):
                pattern_list.append(0)

    ################   pattern 4  #######################
    for i in range(len(otherFunctionList)):
        WhileFlag = 0
        WhileVaraible = None
        ResultValue = None

        if len(pattern_list) > 3:
            break

        for j in range(len(otherFunctionList[i])):
            text = allFunctionList[i][j]
            text_value = re.findall('[a-zA-Z0-9]+', text)
            if "while" in text:
                WhileFlag += 1
                result = re.findall('[(](.*?)[)]', text)
                WhileVaraible = result[0]
                ResultValue = re.findall('[a-zA-Z0-9]+', result[0])

                if "True" == result[0]:
                    LoopFlag3 += 1
                    pattern_list.append(1)
                    break
                elif "==" or "!=" in result:
                    LoopFlag3 += 1
                    pattern_list.append(1)
                    break
                elif j + 1 == len(otherFunctionList[i]) and len(pattern_list) == 2:
                    pattern_list.append(0)

            elif LoopFlag3 == 0 and j + 1 == len(otherFunctionList[i]):
                pattern_list.append(0)

            elif WhileFlag != 0:
                if "<" in WhileVaraible or "<=" in WhileVaraible:
                    if (ResultValue[0] + "--" or ResultValue[0] + "-=") in text.replace(" ", ""):
                        LoopFlag3 += 1
                        pattern_list.append(1)
                        break
                    elif (ResultValue[1] + "++" or ResultValue[0] + "+=") in text.replace(" ", ""):
                        LoopFlag3 += 1
                        pattern_list.append(1)
                        break
                elif ">" in WhileVaraible or ">=" in WhileVaraible:
                    if (ResultValue[0] + "++" or ResultValue[0] + "+=") in text.replace(" ", ""):
                        LoopFlag3 += 1
                        pattern_list.append(1)
                        break
                    elif (ResultValue[1] + "--" or ResultValue[0] + "-=") in text.replace(" ", ""):
                        LoopFlag3 += 1
                        pattern_list.append(1)
                        break

    print(pattern_list)

    return pattern_list


def extract_feature_with_fc(outputPathFC, pattern1, pattern2, pattern3, pattern4):
    pattern1 = torch.Tensor(pattern1)
    pattern2 = torch.Tensor(pattern2)
    pattern3 = torch.Tensor(pattern3)
    pattern4 = torch.Tensor(pattern4)
    model = SimpleFC(5, 100, 200)

    pattern1FC = model(pattern1).detach().numpy().tolist()
    pattern2FC = model(pattern2).detach().numpy().tolist()
    pattern3FC = model(pattern3).detach().numpy().tolist()
    pattern4FC = model(pattern4).detach().numpy().tolist()
    patter_final = np.array([pattern1FC, pattern2FC, pattern3FC, pattern4FC])

    np.savetxt(outputPathFC, patter_final, fmt="%.6f")


if __name__ == "__main__":
    pattern1 = [1, 0, 0, 0]
    pattern2 = [0, 1, 0, 0]
    pattern3 = [0, 0, 1, 0]
    pattern4 = [0, 0, 0, 1]

    label1 = None
    test_contract = "../data/loops/loopwhile77.c"
    pattern_list = extract_pattern(test_contract)
    if len(pattern_list) == 4:
        if sum(pattern_list) == 0:
            label1 = 0
        else:
            label1 = 1
    else:
        print("The extracted patterns are error!")

    pattern1.append(pattern_list[0])
    pattern2.append(pattern_list[1])
    pattern3.append(pattern_list[2])
    pattern4.append(pattern_list[3])

    label = None
    inputFileDir = "../data/loops/"
    outputfeatureDir = "../pattern_feature/feature_zeropadding/loops/"
    outputfeatureFCDir = "../pattern_feature/feature_FNN/loops/"
    outputlabelDir = "../pattern_feature/label_by_extractor/loops/"
    dirs = os.listdir(inputFileDir)
    for file in dirs:
        pattern1 = [1, 0, 0, 0]
        pattern2 = [0, 1, 0, 0]
        pattern3 = [0, 0, 1, 0]
        pattern4 = [0, 0, 0, 1]

        print(file)
        inputFilePath = inputFileDir + file
        name = file.split(".")[0]
        pattern_list = extract_pattern(inputFilePath)
        if len(pattern_list) == 4:
            if sum(pattern_list) == 0:
                label = 0
            else:
                label = 1
        else:
            print("The extracted patterns are error!")

        pattern1.append(pattern_list[0])
        pattern2.append(pattern_list[1])
        pattern3.append(pattern_list[2])
        pattern4.append(pattern_list[3])

        outputPathFC = outputfeatureFCDir + name + ".txt"
        extract_feature_with_fc(outputPathFC, pattern1, pattern2, pattern3, pattern4)

        pattern1 = np.array(pattern1)
        pattern1 = np.array(np.pad(pattern1, (0, 195), 'constant'))
        pattern2 = np.array(pattern2)
        pattern2 = np.array(np.pad(pattern2, (0, 195), 'constant'))
        pattern3 = np.array(pattern3)
        pattern3 = np.array(np.pad(pattern3, (0, 195), 'constant'))
        pattern4 = np.array(pattern4)
        pattern4 = np.array(np.pad(pattern4, (0, 195), 'constant'))
        print(len(pattern4))

        patter_final = np.array([pattern1, pattern2, pattern3, pattern4])
        outputPath = outputfeatureDir + name + ".txt"
        np.savetxt(outputPath, patter_final, fmt="%.6f")

        outputlabelPath = outputlabelDir + file
        f_outlabel = open(outputlabelPath, 'a')
        f_outlabel.write(str(label))
