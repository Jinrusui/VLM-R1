import json 

with open(r'E:\VLM-R1\Visual-Spatial-Planning\percive_sft_train.json', 'r') as f:
    data = json.load(f)
    print(len(data))