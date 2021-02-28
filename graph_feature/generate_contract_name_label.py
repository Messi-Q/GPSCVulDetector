import json

f = open('./feature_by_zeropadding/valid.json', 'r', encoding='utf8')
outputName = open('./feature_by_zeropadding/contract_name_valid.txt', 'a')
outputLabel = open('./feature_by_zeropadding/label_by_experts_valid.txt', 'a')
json_data = json.load(f)

for d in json_data:
    print(d['contract_name'])
    outputName.write(d['contract_name'] + '\n')
    print(d['targets'])
    outputLabel.write(d['targets'] + '\n')
