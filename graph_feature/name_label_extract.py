import json

f = open('./timestamp/train.json', 'r', encoding='utf8')
outputName = open('./timestamp/contract_name_train.txt', 'a')
outputLabel = open('./timestamp/label_by_experts_train.txt', 'a')
json_data = json.load(f)
for d in json_data:
    print(d['contract_name'])
    outputName.write(d['contract_name'] + '\n')
    print(d['targets'])
    outputLabel.write(d['targets'] + '\n')
