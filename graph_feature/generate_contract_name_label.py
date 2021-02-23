import json

f = open('./featurezeropadding/valid.json', 'r', encoding='utf8')
outputName = open('./featurezeropadding/contract_name_valid.txt', 'a')
outputLabel = open('./featurezeropadding/label_by_experts_valid.txt', 'a')
json_data = json.load(f)

for d in json_data:
    print(d['contract_name'])
    outputName.write(d['contract_name'] + '\n')
    print(d['targets'])
    outputLabel.write(d['targets'] + '\n')
