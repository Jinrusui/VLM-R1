import json 

with open(r'E:\VLM-R1\Visual-Spatial-Planning\plan_grpo_train.json', 'r') as f:
    data = json.load(f)
    print(len(data))