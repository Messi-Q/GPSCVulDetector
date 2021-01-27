import re

# keywords of solidity; immutable set
keywords = frozenset(
    {'bool', 'break', 'case', 'catch', 'const', 'continue', 'default', 'do', 'double', 'struct', 'else', 'enum',
     'payable', 'function', 'modifier', 'emit', 'export', 'extern', 'false', 'constructor', 'float', 'if', 'contract',
     'int', 'long', 'string', 'super', 'or', 'protected', 'return', 'returns', 'assert', 'event', 'indexed', 'using',
     'require', 'uint', 'transfer', 'Transfer', 'Transaction', 'switch', 'pure', 'view', 'this', 'throw', 'true', 'try',
     'revert', 'bytes', 'bytes4', 'bytes32', 'internal', 'external', 'union', 'constant', 'while', 'for', 'NULL',
     'uint256', 'uint128', 'uint8', 'uint16', 'address', 'call', 'msg', 'sender', 'public', 'private', 'mapping'})

test = 'require(msg.sender.call.value(amount)());'
tmp = re.compile("[a-z|\'\.\[\]]+", re.I)
list1 = re.findall(tmp, test)
varAllList = []
varTmpList = []

for i in range(len(list1)):
    if list1[i] not in keywords:
        varTmpList.append(list1[i])
        varAllList.append()
print(list1)
print(varAllList)

varAllList.append('amount')
print(varAllList)
