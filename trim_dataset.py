# data = None
# with open("C:\Users\rrn210004\Downloads\squad-main\squad-main\data\train-v2.0.json", "rw") as f:
#     data = f.read()

import json
import random

# Open the original JSON file for reading
with open('C:\\Users\\rrn210004\\Downloads\\squad-main\\squad-main\\data\\test-v2.0.json', 'r') as f:
    data = json.load(f)

print("Length of dataset: ", len(data))
print(type(data))
print(data.keys())
print(type(data['data']))
print(data['data'][1].keys())
print(len(data['data'][1]['paragraphs']))
# print(data['data'][1]['paragraphs'][0])

total = 0
tqas = 0

for t in range(len(data['data'])):
    # print("Topic: ", str(t) + ' ' + data['data'][t]['title'])
    total += len(data['data'][t]['paragraphs'])
    for p in range(len(data['data'][t]['paragraphs'])):
        tqas += len(data['data'][t]['paragraphs'][p]['qas'])

print("Total topics: ", len(data['data']))
print("Total paragraphs: ", total)
print("Total questions: ", tqas)

data1 = data['data']
version = data['version']
# print(data1[0])
# Shuffle the data1 randomly
random.shuffle(data1)

# Take the first half of the shuffled data1
half_size = len(data1) // 2
half_data = dict()
half_data['version'] = version
half_data['data'] = data1[:half_size]

# Open a new file for writing
with open('data\\half_test_data.json', 'w') as f:
    json.dump(half_data, f)
