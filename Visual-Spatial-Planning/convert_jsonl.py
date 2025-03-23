import json
import os

# Define input and output file paths
input_file = "e:\\VLM-R1\\Visual-Spatial-Planning\\plan_grpo_train.json"
output_file = "e:\\VLM-R1\\Visual-Spatial-Planning\\plan_grpo_train.jsonl"

# Read the JSON data
with open(input_file, 'r') as f:
    data = json.load(f)

# Create JSONL formatted data
jsonl_data = []
for idx, entry in enumerate(data):
    # Skip entries that don't have required fields
    if not all(key in entry for key in ["image", "problem", "env"]):
        continue
    
    # Handle different image formats (string or list)
    images = entry["image"]
    if not isinstance(images, list):
        images = [images]
    
    # Create the conversation entry
    conversation_entry = {
        "id": idx + 1,
        "image": images,
        "conversations": [
            {"from": "human", "value": f"<image><image>{entry['problem']}"},
            {"from": "gpt", "value": entry['env']}
        ]
    }
    
    jsonl_data.append(conversation_entry)

# Write to JSONL file
with open(output_file, 'w') as f:
    for entry in jsonl_data:
        f.write(json.dumps(entry) + '\n')

print(f"Conversion complete. Created {len(jsonl_data)} entries in {output_file}")