import re
import numpy as np

# keywords of solidity; immutable set
keywords = frozenset(
    {'bool', 'break', 'case', 'catch', 'const', 'continue', 'default', 'do', 'double', 'struct', 'else', 'enum',
     'payable', 'function', 'modifier', 'emit', 'export', 'extern', 'false', 'constructor', 'float', 'if', 'contract',
     'int', 'long', 'string', 'super', 'or', 'protected', 'return', 'returns', 'assert', 'event', 'indexed', 'using',
     'require', 'uint', 'transfer', 'Transfer', 'Transaction', 'switch', 'pure', 'view', 'this', 'throw', 'true', 'try',
     'revert', 'bytes', 'bytes4', 'bytes32', 'internal', 'external', 'union', 'constant', 'while', 'for', 'NULL',
     'uint256', 'uint128', 'uint8', 'uint16', 'address', 'call', 'msg', 'sender', 'public', 'private', 'mapping'})

# map user-defined variables to symbolic names
var_list = ['balances[msg.sender]', 'participated[msg.sender]', 'playerPendingWithdrawals[msg.sender]',
            'tokens[msg.sender]', 'accountBalances[msg.sender]', 'creditedPoints[msg.sender]',
            'balances[from]', 'balances[recipient]', 'claimedBonus[recipient]', 'Bal[msg.sender]',
            'Accounts[msg.sender]', 'ExtractDepositTime[msg.sender]', 'Bids[msg.sender]',
            'participated[msg.sender]', 'latestSeriesForUser[msg.sender]', 'payments[msg.sender]',
            'rewardsForA[recipient]', 'userBalance[msg.sender]', 'credit[msg.sender]',
            'credit[to]', 'userPendingWithdrawals[msg.sender]']

# function limit type
function_limit = ['private', 'onlyOwner', 'internal', 'onlyGovernor', 'onlyCommittee', 'onlyAdmin', 'onlyPlayers',
                  'onlyManager', 'onlyHuman', 'onlyCongressMembers', 'preventReentry', 'onlyMembers',
                  'onlyProxyOwner', 'noReentrancy', 'notExecuted', 'noEther', 'notConfirmed']

# Boolean condition expression
var_op_bool = ['!', '~', '**', '*', '!=', '<', '>', '<=', '>=', '==', '<<', '>>', '||', '&&']

# Assignment expressions
var_op_assign = ['|=', '=', '^=', '&=', '<<=', '>>=', '+=', '-=', '*=', '/=', '%=', '++', '--']


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
            if text.split()[0] == "function" or text.split()[0] == "constructor":
                function_list.append([text])
                flag += 1
            elif len(function_list) > 0 and ("function" or "constructor" in function_list[flag][0]):
                function_list[flag].append(text)

    return function_list


# locate the call.value (core node) to generate the contract graph (nodes and edges)
def generate_graph(filepath):
    allFunctionList = split_function(filepath)  # Store all functions
    callValueList = []  # Store all core function that call the call.value
    normalFunction = []  # Store a single normal node function that calls a core function
    NormalFunctions = []  # Store all normal node functions that call core functions
    coreFunctionNameList = []  # Store the core function name that calls call.value
    otherFunctionList = []  # Store functions other than the core function
    node_list = []  # Store all the nodes
    edge_feature_list = []  # Store edge and edge features
    node_feature_list = []  # Store node and node features
    core_count = 0  # Number of core nodes 
    normal_count = 0  # Number of normal nodes
    params = []
    param = []

    # ======================================================================
    # ---------------------------  Handle nodes  ----------------------------
    # ======================================================================
    # Traverse all functions, store the function to invoke the call.value and other functions
    for i in range(len(allFunctionList)):
        flag = 0
        for j in range(len(allFunctionList[i])):
            text = allFunctionList[i][j]
            if '.call.value' in text:
                callValueList.append(allFunctionList[i])
                flag += 1
        if flag == 0:
            otherFunctionList.append(allFunctionList[i])

    # call.value as the core node and the function as the core node
    for i in range(len(callValueList)):
        # get the function name and params. function withdraw(uint amount) public
        tmp = re.compile(r'\b([_A-Za-z]\w*)\b(?:(?=\s*\w+\()|(?!\s*\w+))')
        resultFunction = tmp.findall(callValueList[i][0])
        functionNameTmp = resultFunction[1]
        if functionNameTmp == "payable":
            functionName = functionNameTmp
        else:
            functionName = functionNameTmp + "("
        coreFunctionNameList.append(functionName)

        functionSentence = callValueList[i][0]
        matchingRule = re.compile(r'[(](.*?)[)]', re.S)
        resultParams = re.findall(matchingRule, functionSentence)
        variables = resultParams[0].split(",")
        for n in range(len(variables)):
            params.append([variables[n].strip().split(" ")[-1]])

        # Handling the core function access restrictions
        limit_count = 0
        limitFlag = None
        for k in range(len(function_limit)):
            if function_limit[k] in functionSentence:
                limit_count += 1
                limitFlag = "LimitedAC"
        if limit_count == 0:
            limitFlag = "NoLimit"

        # traverse the call.value function and extract critical variables
        for j in range(1, len(callValueList[i])):
            node_list.append("C" + str(core_count))
            core_count += 1
            text = callValueList[i][j]

            if '.call.value' in text:
                node_list.append("C" + str(core_count))

            else:
                varList = []
                tmp = re.compile("[a-z|\'\.\[\]]+", re.I)
                allVarlist = re.findall(tmp, text)
                for k in range(len(allVarlist)):
                    if allVarlist[i] not in keywords:
                        varList.append(allVarlist[i])



    # Traverse all functions, construct core node and normal node
    for i in range(len(allFunctionList)):
        for j in range(len(allFunctionList[i])):
            text = allFunctionList[i][j]
            if '.call.value' in text:
                node_list.append("C" + str(core_count))
                core_count += 1
                node_list.append("C" + str(core_count))
                callValueList.append([allFunctionList[i], "S", "W" + str(core_count)])

                # get the function name and params
                ss = allFunctionList[i][0]
                pp = re.compile(r'[(](.*?)[)]', re.S)
                result = re.findall(pp, ss)
                result_params = result[0].split(",")

                for n in range(len(result_params)):
                    param.append(result_params[n].strip().split(" ")[-1])

                params.append([param, "S", "W" + str(core_count)])

                # Handling W function access restrictions, which can be used for access restriction properties
                # default that there are C nodes
                limit_count = 0
                for k in range(len(function_limit)):
                    if function_limit[k] in callValueList[core_count][0][0]:
                        limit_count += 1
                        if "address" in text:
                            node_feature_list.append(
                                ["S", "S", "LimitedAC", ["W" + str(core_count)],
                                 2, "INNADD"])
                            node_feature_list.append(
                                ["W" + str(core_count), "W" + str(core_count), "LimitedAC", [],
                                 1, "NULL"])
                            break
                        elif "msg.sender" in text:
                            node_feature_list.append(
                                ["S", "S", "LimitedAC", ["W" + str(core_count)],
                                 2, "MSG"])
                            node_feature_list.append(
                                ["W" + str(core_count), "W" + str(core_count), "LimitedAC", [],
                                 1, "NULL"])
                            break
                        else:
                            param_count = 0
                            for pa in param:
                                if pa in text and pa != "":
                                    param_count += 1
                                    node_feature_list.append(
                                        ["S", "S", "LimitedAC",
                                         ["W" + str(core_count)],
                                         2, "MSG"])
                                    node_feature_list.append(
                                        ["W" + str(core_count), "W" + str(core_count), "LimitedAC", [],
                                         1, "NULL"])
                                    break
                            if param_count == 0:
                                node_feature_list.append(
                                    ["S", "S", "LimitedAC", ["W" + str(core_count)],
                                     2, "INNADD"])
                                node_feature_list.append(
                                    ["W" + str(core_count), "W" + str(core_count), "LimitedAC", [],
                                     1, "NULL"])
                            break
                if limit_count == 0:
                    if "address" in text:
                        node_feature_list.append(
                            ["S", "S", "NoLimit", ["W" + str(core_count)],
                             2, "INNADD"])
                        node_feature_list.append(
                            ["W" + str(core_count), "W" + str(core_count), "NoLimit", [],
                             1, "NULL"])
                    elif "msg.sender" in text:
                        node_feature_list.append(
                            ["S", "S", "NoLimit", ["W" + str(core_count)],
                             2, "MSG"])
                        node_feature_list.append(
                            ["W" + str(core_count), "W" + str(core_count), "NoLimit", [],
                             1, "NULL"])
                    else:
                        param_count = 0
                        for pa in param:
                            if pa in text and pa != "":
                                param_count += 1
                                node_feature_list.append(
                                    ["S", "S", "NoLimit", ["W" + str(core_count)],
                                     2, "MSG"])
                                node_feature_list.append(
                                    ["W" + str(core_count), "W" + str(core_count), "NoLimit", [],
                                     1, "NULL"])
                                break
                        if param_count == 0:
                            node_feature_list.append(
                                ["S", "S", "NoLimit", ["W" + str(core_count)],
                                 2, "INNADD"])
                            node_feature_list.append(
                                ["W" + str(core_count), "W" + str(core_count), "NoLimit", [],
                                 1, "NULL"])

                # For example: function transfer(address _to, uint _value, bytes _data, string _custom_fallback)
                # get function name (transfer)
                tmp = re.compile(r'\b([_A-Za-z]\w*)\b(?:(?=\s*\w+\()|(?!\s*\w+))')
                result_withdraw = tmp.findall(allFunctionList[i][0])
                withdrawNameTmp = result_withdraw[1]
                if withdrawNameTmp == "payable":
                    withdrawName = withdrawNameTmp
                else:
                    withdrawName = withdrawNameTmp + "("
                coreFunctionNameList.append(["W" + str(core_count), withdrawName])

                core_count += 1

    if core_count == 0:
        print("Currently, there is no key word call.value")
        node_feature_list.append(["S", "S", "NoLimit", ["NULL"], 0, "NULL"])
        node_feature_list.append(["W0", "W0", "NoLimit", ["NULL"], 0, "NULL"])
        node_feature_list.append(["C0", "C0", "NoLimit", ["NULL"], 0, "NULL"])
    else:
        # Traverse all functions and find the C function nodes that calls the W function
        # (determine the function call by matching the number of arguments)
        for k in range(len(coreFunctionNameList)):
            w_key = coreFunctionNameList[k][0]
            w_name = coreFunctionNameList[k][1]
            for i in range(len(otherFunctionList)):
                if len(otherFunctionList[i]) > 2:
                    for j in range(1, len(otherFunctionList[i])):
                        text = otherFunctionList[i][j]
                        if w_name in text:
                            p = re.compile(r'[(](.*?)[)]', re.S)
                            result = re.findall(p, text)
                            result_params = result[0].split(",")

                            if result_params[0] != "" and len(result_params) == len(params[k][0]):
                                normalFunction += otherFunctionList[i]
                                NormalFunctions.append(
                                    [w_key, w_name, "C" + str(normal_count), otherFunctionList[i]])
                                node_list.append("C" + str(normal_count))

                                for n in range(len(node_feature_list)):
                                    if w_key in node_feature_list[n][0]:
                                        node_feature_list[n][3].append("C" + str(normal_count))

                                # Handling C function access restrictions
                                limit_count = 0
                                for m in range(len(function_limit)):
                                    if function_limit[m] in normalFunction[0]:
                                        limit_count += 1
                                        node_feature_list.append(
                                            ["C" + str(normal_count), "C" + str(normal_count), "LimitedAC", ["NULL"], 0,
                                             "NULL"])
                                        break
                                if limit_count == 0:
                                    node_feature_list.append(
                                        ["C" + str(normal_count), "C" + str(normal_count), "NoLimit", ["NULL"], 0,
                                         "NULL"])
                                normal_count += 1
                                break

        if normal_count == 0:
            print("There is no C node")
            node_list.append("C0")
            node_feature_list.append(["C0", "C0", "NoLimit", ["NULL"], 0, "NULL"])
            for n in range(len(node_feature_list)):
                if "W" in node_feature_list[n][0]:
                    node_feature_list[n][3] = ["NULL"]

        # ======================================================================
        # ---------------------------  Handle edge  ----------------------------
        # ======================================================================

        # (1) W->S (include: W->VAR, VAR->S, S->VAR)
        for i in range(len(callValueList)):
            flag = 0  # flag: flag = 0, before call.value; flag > 0, after call.value
            before_var_count = 0
            after_var_count = 0
            var_tmp = []
            var_name = []
            var_w_name = []
            for j in range(len(callValueList[i][0])):
                text = callValueList[i][0][j]
                if '.call.value' not in text:
                    if flag == 0:
                        # print("before call.value")
                        # handle W -> VAR
                        for k in range(len(var_list)):
                            if var_list[k] in text:
                                node_list.append("VAR" + str(before_var_count))
                                var_tmp.append("VAR" + str(before_var_count))

                                if len(var_w_name) == 0:
                                    if "assert" in text:
                                        edge_feature_list.append(
                                            [callValueList[i][2], "VAR" + str(before_var_count), callValueList[i][2], 1,
                                             'AH'])
                                    elif "require" in text:
                                        edge_feature_list.append(
                                            [callValueList[i][2], "VAR" + str(before_var_count), callValueList[i][2], 1,
                                             'RG'])
                                    elif j >= 1:
                                        if "if" in callValueList[i][0][j - 1]:
                                            edge_feature_list.append(
                                                [callValueList[i][2], "VAR" + str(before_var_count),
                                                 callValueList[i][2], 1,
                                                 'GN'])
                                        elif "for" in callValueList[i][0][j - 1]:
                                            edge_feature_list.append(
                                                [callValueList[i][2], "VAR" + str(before_var_count),
                                                 callValueList[i][2], 1,
                                                 'FOR'])
                                        elif "else" in callValueList[i][0][j - 1]:
                                            edge_feature_list.append(
                                                [callValueList[i][2], "VAR" + str(before_var_count),
                                                 callValueList[i][2], 1,
                                                 'GB'])
                                        elif j + 1 < len(callValueList[i][0]):
                                            if "if" and "throw" in callValueList[i][0][j] or "if" in \
                                                    callValueList[i][0][j] \
                                                    and "throw" in callValueList[i][0][j + 1]:
                                                edge_feature_list.append(
                                                    [callValueList[i][2], "VAR" + str(before_var_count),
                                                     callValueList[i][2], 1, 'IT'])
                                            elif "if" and "revert" in callValueList[i][0][j] or "if" in \
                                                    callValueList[i][0][
                                                        j] and "revert" in callValueList[i][0][j + 1]:
                                                edge_feature_list.append(
                                                    [callValueList[i][2], "VAR" + str(before_var_count),
                                                     callValueList[i][2], 1, 'RH'])
                                            elif "if" in text:
                                                edge_feature_list.append(
                                                    [callValueList[i][2], "VAR" + str(before_var_count),
                                                     callValueList[i][2], 1, 'IF'])
                                            else:
                                                edge_feature_list.append(
                                                    [callValueList[i][2], "VAR" + str(before_var_count),
                                                     callValueList[i][2], 1, 'FW'])
                                        else:
                                            edge_feature_list.append(
                                                [callValueList[i][2], "VAR" + str(before_var_count),
                                                 callValueList[i][2], 1,
                                                 'FW'])
                                    else:
                                        edge_feature_list.append(
                                            [callValueList[i][2], "VAR" + str(before_var_count), callValueList[i][2], 1,
                                             'FW'])

                                    var_node = 0
                                    var_bool_node = 0
                                    for b in range(len(var_op_bool)):
                                        if var_op_bool[b] in text:
                                            node_feature_list.append(
                                                ["VAR" + str(before_var_count), "VAR" + str(before_var_count),
                                                 callValueList[i][2], 1, 'BOOL'])
                                            var_node += 1
                                            var_bool_node += 1
                                            break

                                    if var_bool_node == 0:
                                        for a in range(len(var_op_assign)):
                                            if var_op_assign[a] in text:
                                                node_feature_list.append(
                                                    ["VAR" + str(before_var_count), "VAR" + str(before_var_count),
                                                     callValueList[i][2], 1, 'ASSIGN'])
                                                var_node += 1
                                                break

                                    if var_node == 0:
                                        node_feature_list.append(
                                            ["VAR" + str(before_var_count), "VAR" + str(before_var_count),
                                             callValueList[i][2], 1, 'NULL'])

                                    var_w_name.append(var_list[k])
                                    var_name.append(var_list[k])
                                    before_var_count += 1
                                else:
                                    var_w_count = 0
                                    for n in range(len(var_w_name)):
                                        if var_list[k] == var_w_name[n]:
                                            var_w_count += 1
                                            var_tmp.append(var_tmp[len(var_tmp) - 1])

                                            var_node = 0
                                            var_bool_node = 0
                                            for b in range(len(var_op_bool)):
                                                if var_op_bool[b] in text:
                                                    node_feature_list.append(
                                                        [var_tmp[len(var_tmp) - 1], var_tmp[len(var_tmp) - 1],
                                                         callValueList[i][2], 1, 'BOOL'])
                                                    var_bool_node += 1
                                                    var_node += 1
                                                    break

                                            if var_bool_node == 0:
                                                for a in range(len(var_op_assign)):
                                                    if var_op_assign[a] in text:
                                                        node_feature_list.append(
                                                            [var_tmp[len(var_tmp) - 1], var_tmp[len(var_tmp) - 1],
                                                             callValueList[i][2], 1, 'ASSIGN'])
                                                        var_node += 1
                                                        break

                                            if var_node == 0:
                                                node_feature_list.append(
                                                    [var_tmp[len(var_tmp) - 1], var_tmp[len(var_tmp) - 1],
                                                     callValueList[i][2], 1, 'NULL'])

                                    if var_w_count == 0:
                                        var_node = 0
                                        var_bool_node = 0
                                        var_tmp.append("VAR" + str(before_var_count))

                                        for b in range(len(var_op_bool)):
                                            if var_op_bool[b] in text:
                                                node_feature_list.append(
                                                    ["VAR" + str(before_var_count), "VAR" + str(before_var_count),
                                                     callValueList[i][2], 1, 'BOOL'])
                                                var_node += 1
                                                var_bool_node += 1
                                                break

                                        if var_bool_node == 0:
                                            for a in range(len(var_op_assign)):
                                                if var_op_assign[a] in text:
                                                    node_feature_list.append(
                                                        ["VAR" + str(before_var_count), "VAR" + str(before_var_count),
                                                         callValueList[i][2], 1, 'ASSIGN'])
                                                    var_node += 1
                                                    break

                                        if var_node == 0:
                                            node_feature_list.append(
                                                ["VAR" + str(before_var_count), "VAR" + str(before_var_count),
                                                 callValueList[i][2], 1, 'NULL'])

                    elif flag != 0:
                        # print("after call.value")
                        # handle S->VAR
                        var_count = 0
                        for k in range(len(var_list)):
                            if var_list[k] in text:
                                if before_var_count == 0:
                                    node_list.append("VAR" + str(after_var_count))
                                    var_tmp.append("VAR" + str(after_var_count))

                                    if "assert" in text:
                                        edge_feature_list.append(
                                            [callValueList[i][1], "VAR" + str(after_var_count), callValueList[i][1], 3,
                                             'AH'])
                                    elif "require" in text:
                                        edge_feature_list.append(
                                            [callValueList[i][1], "VAR" + str(after_var_count), callValueList[i][1], 3,
                                             'RG'])
                                    elif "return" in text:
                                        edge_feature_list.append(
                                            [callValueList[i][1], "VAR" + str(after_var_count), callValueList[i][1], 3,
                                             'RE'])
                                    elif "if" and "throw" in text:
                                        edge_feature_list.append(
                                            [callValueList[i][1], "VAR" + str(after_var_count), callValueList[i][1], 3,
                                             'IT'])
                                    elif "if" and "revert" in text:
                                        edge_feature_list.append(
                                            [callValueList[i][1], "VAR" + str(after_var_count), callValueList[i][1], 3,
                                             'RH'])
                                    elif "if" in text:
                                        edge_feature_list.append(
                                            [callValueList[i][1], "VAR" + str(after_var_count), callValueList[i][1], 3,
                                             'IF'])
                                    else:
                                        edge_feature_list.append(
                                            [callValueList[i][1], "VAR" + str(after_var_count), callValueList[i][1], 3,
                                             'FW'])

                                    var_node = 0
                                    var_bool_node = 0
                                    for b in range(len(var_op_bool)):
                                        if var_op_bool[b] in text:
                                            node_feature_list.append(
                                                ["VAR" + str(after_var_count), "VAR" + str(after_var_count),
                                                 callValueList[i][1], 3, 'BOOL'])
                                            var_node += 1
                                            var_bool_node += 1
                                            break

                                    if var_bool_node == 0:
                                        for a in range(len(var_op_assign)):
                                            if var_op_assign[a] in text:
                                                node_feature_list.append(
                                                    ["VAR" + str(after_var_count), "VAR" + str(after_var_count),
                                                     callValueList[i][1], 3, 'ASSIGN'])
                                                var_node += 1
                                                break

                                    if var_node == 0:
                                        node_feature_list.append(
                                            ["VAR" + str(after_var_count), "VAR" + str(after_var_count),
                                             callValueList[i][1], 3, 'NULL'])

                                    # after_var_count += 1

                                elif before_var_count > 0:
                                    for n in range(len(var_name)):
                                        if var_list[k] == var_name[n]:
                                            var_count += 1
                                            if "assert" in text:
                                                edge_feature_list.append(
                                                    [callValueList[i][1], var_tmp[len(var_tmp) - 1],
                                                     callValueList[i][1], 3,
                                                     'AH'])
                                            elif "require" in text:
                                                edge_feature_list.append(
                                                    [callValueList[i][1], var_tmp[len(var_tmp) - 1],
                                                     callValueList[i][1], 3,
                                                     'RG'])
                                            elif "return" in text:
                                                edge_feature_list.append(
                                                    [callValueList[i][1], var_tmp[len(var_tmp) - 1],
                                                     callValueList[i][1], 3,
                                                     'RE'])
                                            elif "if" and "throw" in text:
                                                edge_feature_list.append(
                                                    [callValueList[i][1], var_tmp[len(var_tmp) - 1],
                                                     callValueList[i][1], 3,
                                                     'IT'])
                                            elif "if" and "revert" in text:
                                                edge_feature_list.append(
                                                    [callValueList[i][1], var_tmp[len(var_tmp) - 1],
                                                     callValueList[i][1], 3,
                                                     'RH'])
                                            elif "if" in text:
                                                edge_feature_list.append(
                                                    [callValueList[i][1], var_tmp[len(var_tmp) - 1],
                                                     callValueList[i][1], 3,
                                                     'IF'])
                                            else:
                                                edge_feature_list.append(
                                                    [callValueList[i][1], var_tmp[len(var_tmp) - 1],
                                                     callValueList[i][1], 3,
                                                     'FW'])

                                            after_var_count += 1

                elif '.call.value' in text:
                    flag += 1

                    if len(var_tmp) > 0:
                        if "assert" in text:
                            edge_feature_list.append(
                                [var_tmp[len(var_tmp) - 1], callValueList[i][1], callValueList[i][2], 2, 'AH'])
                        elif "require" in text:
                            edge_feature_list.append(
                                [var_tmp[len(var_tmp) - 1], callValueList[i][1], callValueList[i][2], 2, 'RG'])
                        elif "return" in text:
                            edge_feature_list.append(
                                [var_tmp[len(var_tmp) - 1], callValueList[i][1], callValueList[i][2], 2, 'RE'])
                        elif j > 1:
                            if "if" in callValueList[i][0][j - 1]:
                                edge_feature_list.append(
                                    [var_tmp[len(var_tmp) - 1], callValueList[i][1], callValueList[i][2], 2, 'GN'])
                            elif "for" in callValueList[i][0][j - 1]:
                                edge_feature_list.append(
                                    [var_tmp[len(var_tmp) - 1], callValueList[i][1], callValueList[i][2], 2, 'FOR'])
                            elif "else" in callValueList[i][0][j - 1]:
                                edge_feature_list.append(
                                    [var_tmp[len(var_tmp) - 1], callValueList[i][1], callValueList[i][2], 2, 'GB'])
                            elif j + 1 < len(callValueList[i][0]):
                                if "if" and "throw" in callValueList[i][0][j] or "if" in callValueList[i][0][j] \
                                        and "throw" in callValueList[i][0][j + 1]:
                                    edge_feature_list.append(
                                        [var_tmp[len(var_tmp) - 1], callValueList[i][1], callValueList[i][2], 2, 'IT'])
                                elif "if" and "revert" in callValueList[i][0][j] or "if" in callValueList[i][0][j] \
                                        and "revert" in callValueList[i][0][j + 1]:
                                    edge_feature_list.append(
                                        [var_tmp[len(var_tmp) - 1], callValueList[i][1], callValueList[i][2], 2, 'RH'])
                                elif "if" in text:
                                    edge_feature_list.append(
                                        [var_tmp[len(var_tmp) - 1], callValueList[i][1], callValueList[i][2], 2, 'IF'])
                                else:
                                    edge_feature_list.append(
                                        [var_tmp[len(var_tmp) - 1], callValueList[i][1], callValueList[i][2], 2, 'FW'])
                            else:
                                edge_feature_list.append(
                                    [var_tmp[len(var_tmp) - 1], callValueList[i][1], callValueList[i][2], 2, 'FW'])
                        else:
                            edge_feature_list.append(
                                [var_tmp[len(var_tmp) - 1], callValueList[i][1], callValueList[i][2], 2, 'FW'])

                    elif len(var_tmp) == 0:
                        if "assert" in text:
                            edge_feature_list.append(
                                [callValueList[i][2], callValueList[i][1], callValueList[i][2], 1, 'AH'])
                        elif "require" in text:
                            edge_feature_list.append(
                                [callValueList[i][2], callValueList[i][1], callValueList[i][2], 1, 'RG'])
                        elif "return" in text:
                            edge_feature_list.append(
                                [callValueList[i][2], callValueList[i][1], callValueList[i][2], 1, 'RE'])
                        elif j > 1:
                            if "if" in callValueList[i][0][j - 1]:
                                edge_feature_list.append(
                                    [callValueList[i][2], callValueList[i][1], callValueList[i][2], 1, 'GN'])
                            elif "for" in callValueList[i][0][j - 1]:
                                edge_feature_list.append(
                                    [callValueList[i][2], callValueList[i][1], callValueList[i][2], 1, 'FOR'])
                            elif "else" in callValueList[i][0][j - 1]:
                                edge_feature_list.append(
                                    [callValueList[i][2], callValueList[i][1], callValueList[i][2], 1, 'GB'])
                            elif j + 1 < len(callValueList[i][0]):
                                if "if" and "throw" in callValueList[i][0][j] or "if" in callValueList[i][0][j] \
                                        and "throw" in callValueList[i][0][j + 1]:
                                    edge_feature_list.append(
                                        [callValueList[i][2], callValueList[i][1], callValueList[i][2], 1, 'IT'])
                                elif "if" and "revert" in callValueList[i][0][j] or "if" in callValueList[i][0][j] \
                                        and "revert" in callValueList[i][0][j + 1]:
                                    edge_feature_list.append(
                                        [callValueList[i][2], callValueList[i][1], callValueList[i][2], 1, 'RH'])
                                elif "if" in text:
                                    edge_feature_list.append(
                                        [callValueList[i][2], callValueList[i][1], callValueList[i][2], 1, 'IF'])
                                else:
                                    edge_feature_list.append(
                                        [callValueList[i][2], callValueList[i][1], callValueList[i][2], 1, 'FW'])
                            else:
                                edge_feature_list.append(
                                    [callValueList[i][2], callValueList[i][1], callValueList[i][2], 1, 'FW'])
                        else:
                            edge_feature_list.append(
                                [callValueList[i][2], callValueList[i][1], callValueList[i][2], 1, 'FW'])

        # (2) handle C->W (include C->VAR, VAR->W)
        for i in range(len(NormalFunctions)):
            for j in range(len(NormalFunctions[i][3])):
                text = NormalFunctions[i][3][j]
                var_flag = 0
                for k in range(len(var_list)):
                    if var_list[k] in text:
                        var_flag += 1

                        var_node = 0
                        var_bool_node = 0
                        for b in range(len(var_op_bool)):
                            if var_op_bool[b] in text:
                                node_feature_list.append(
                                    ["VAR" + str(len(var_tmp)), "VAR" + str(len(var_tmp)),
                                     NormalFunctions[i][2], 1, 'BOOL'])
                                var_node += 1
                                var_bool_node += 1
                                break

                        if var_bool_node == 0:
                            for a in range(len(var_op_assign)):
                                if var_op_assign[a] in text:
                                    node_feature_list.append(
                                        ["VAR" + str(len(var_tmp)), "VAR" + str(len(var_tmp)),
                                         NormalFunctions[i][2], 1, 'ASSIGN'])
                                    var_node += 1
                                    break

                        if var_node == 0:
                            node_feature_list.append(
                                ["VAR" + str(len(var_tmp)), "VAR" + str(len(var_tmp)),
                                 NormalFunctions[i][2], 1, 'NULL'])

                        if "assert" in text:
                            edge_feature_list.append(
                                [NormalFunctions[i][2], "VAR" + str(len(var_tmp)), NormalFunctions[i][2], 1, 'AH'])
                            edge_feature_list.append(
                                ["VAR" + str(len(var_tmp)), NormalFunctions[i][0], NormalFunctions[i][2], 2, 'FW'])
                        elif "require" in text:
                            edge_feature_list.append(
                                [NormalFunctions[i][2], "VAR" + str(len(var_tmp)), NormalFunctions[i][2], 1, 'RG'])
                            edge_feature_list.append(
                                ["VAR" + str(len(var_tmp)), NormalFunctions[i][0], NormalFunctions[i][2], 2, 'FW'])
                        elif "if" and "throw" in text:
                            edge_feature_list.append(
                                [NormalFunctions[i][2], "VAR" + str(len(var_tmp)), NormalFunctions[i][2], 1, 'IT'])
                            edge_feature_list.append(
                                ["VAR" + str(len(var_tmp)), NormalFunctions[i][0], NormalFunctions[i][2], 2, 'FW'])
                        elif "if" and "revert" in text:
                            edge_feature_list.append(
                                [NormalFunctions[i][2], "VAR" + str(len(var_tmp)), NormalFunctions[i][2], 1, 'RH'])
                            edge_feature_list.append(
                                ["VAR" + str(len(var_tmp)), NormalFunctions[i][0], NormalFunctions[i][2], 2, 'FW'])
                        elif "if" in text:
                            edge_feature_list.append(
                                [NormalFunctions[i][2], "VAR" + str(len(var_tmp)), NormalFunctions[i][2], 1, 'IF'])
                            edge_feature_list.append(
                                ["VAR" + str(len(var_tmp)), NormalFunctions[i][0], NormalFunctions[i][2], 2, 'FW'])
                        else:
                            edge_feature_list.append(
                                [NormalFunctions[i][2], "VAR" + str(len(var_tmp)), NormalFunctions[i][2], 1, 'FW'])
                            edge_feature_list.append(
                                ["VAR" + str(len(var_tmp)), NormalFunctions[i][0], NormalFunctions[i][2], 2, 'FW'])
                        break

                if var_flag == 0:
                    if "assert" in text:
                        edge_feature_list.append(
                            [NormalFunctions[i][2], NormalFunctions[i][0], NormalFunctions[i][2], 1, 'AH'])
                    elif "require" in text:
                        edge_feature_list.append(
                            [NormalFunctions[i][2], NormalFunctions[i][0], NormalFunctions[i][2], 1, 'RG'])
                    elif "if" and "throw" in text:
                        edge_feature_list.append(
                            [NormalFunctions[i][2], NormalFunctions[i][0], NormalFunctions[i][2], 1, 'IT'])
                    elif "if" and "revert" in text:
                        edge_feature_list.append(
                            [NormalFunctions[i][2], NormalFunctions[i][0], NormalFunctions[i][2], 1, 'RH'])
                    elif "if" in text:
                        edge_feature_list.append(
                            [NormalFunctions[i][2], NormalFunctions[i][0], NormalFunctions[i][2], 1, 'IF'])
                    else:
                        edge_feature_list.append(
                            [NormalFunctions[i][2], NormalFunctions[i][0], NormalFunctions[i][2], 1, 'FW'])
                    break
                else:
                    print("The C function does not call the corresponding W function")

    # Handling some duplicate elements, the filter leaves a unique
    edge_feature_list = list(set([tuple(t) for t in edge_feature_list]))
    edge_feature_list = [list(v) for v in edge_feature_list]
    node_feature_list_new = []
    [node_feature_list_new.append(i) for i in node_feature_list if not i in node_feature_list_new]
    # node_feature_list = list(set([tuple(t) for t in node_feature_list]))
    # node_feature_list = [list(v) for v in node_feature_list]
    # node_list = list(set(node_list))

    node_feature = sorted(node_feature_list_new, key=lambda x: (x[0]))
    edge_feature = sorted(edge_feature_list, key=lambda x: (x[2], x[3]))

    # Construct Fallback Node
    node_feature, edge_feature = generate_potential_fallback_node(node_feature, edge_feature)

    return node_feature, edge_feature


# generate a potential fallback node
def generate_potential_fallback_node(node_feature, edge_feature):
    node_feature.append(["F", "F", "NoLimit", ["S"], 0, "MSG"])
    edge_feature.append(["S", "F", "S", 0, "FW"])
    edge_feature.append(["F", "W0", "F", 1, "FW"])
    return node_feature, edge_feature


def printResult(file, node_feature, edge_feature):
    main_point = ['S', 'W0', 'W1', 'W2', 'W3', 'W4', 'C0', 'C1', 'C2', 'C3', 'C4', 'F']

    for i in range(len(node_feature)):
        if node_feature[i][0] in main_point:
            for j in range(0, len(node_feature[i][3]), 2):
                if j + 1 < len(node_feature[i][3]):
                    tmp = node_feature[i][3][j] + "," + node_feature[i][3][j + 1]
                elif len(node_feature[i][3]) == 1:
                    tmp = node_feature[i][3][j]

            node_feature[i][3] = tmp

    nodeOutPath = "../../data/reentrancy/node/" + file
    edgeOutPath = "../../data/reentrancy/edge/" + file

    f_node = open(nodeOutPath, 'a')
    for i in range(len(node_feature)):
        result = " ".join(np.array(node_feature[i]))
        f_node.write(result + '\n')
    f_node.close()

    f_edge = open(edgeOutPath, 'a')
    for i in range(len(edge_feature)):
        result = " ".join(np.array(edge_feature[i]))
        print(result)
        f_edge.write(result + '\n')
    f_edge.close()

    return node_feature, edge_feature


if __name__ == "__main__":
    test_contract = "../../data/reentrancy/simple_dao.sol"
    node_feature, edge_feature = generate_graph(test_contract)
    print("node_feature", node_feature)
    print("edge_feature", edge_feature)
