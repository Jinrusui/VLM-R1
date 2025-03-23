import json
import os

# Define input file paths
input_file1 = 'VSP-main/blocks/b_percive_grpo.json'
input_file2 = 'VSP-main/frozenlake/fl_percive_grpo.json'
output_file = 'percive_grpo.json'

# Read the first JSON file
with open(input_file1, 'r') as f:
    data1 = json.load(f)

# Read the second JSON file
with open(input_file2, 'r') as f:
    data2 = json.load(f)

# Concatenate the JSON data (assuming both are lists)
combined_data = data1 + data2

# Write the concatenated data to the output file
with open(output_file, 'w') as f:
    json.dump(combined_data, f, indent=2)

print(f"Successfully concatenated {len(data1)} elements from {input_file1} and {len(data2)} elements from {input_file2}")
print(f"Total of {len(combined_data)} elements written to {output_file}")



# Define input file paths
input_file1 = 'VSP-main/blocks/b_plan_grpo.json'
input_file2 = 'VSP-main/frozenlake/fl_plan_grpo.json'
output_file = 'plan_grpo.json'

# Read the first JSON file
with open(input_file1, 'r') as f:
    data1 = json.load(f)

# Read the second JSON file
with open(input_file2, 'r') as f:
    data2 = json.load(f)

# Concatenate the JSON data (assuming both are lists)
combined_data = data1 + data2

# Write the concatenated data to the output file
with open(output_file, 'w') as f:
    json.dump(combined_data, f, indent=2)

print(f"Successfully concatenated {len(data1)} elements from {input_file1} and {len(data2)} elements from {input_file2}")
print(f"Total of {len(combined_data)} elements written to {output_file}")

# Define input file paths
input_file1 = 'VSP-main/blocks/b_plan_sft.json'
input_file2 = 'VSP-main/frozenlake/fl_plan_sft.json'
output_file = 'plan_sft.json'

# Read the first JSON file
with open(input_file1, 'r') as f:
    data1 = json.load(f)

# Read the second JSON file
with open(input_file2, 'r') as f:
    data2 = json.load(f)

# Concatenate the JSON data (assuming both are lists)
combined_data = data1 + data2

# Write the concatenated data to the output file
with open(output_file, 'w') as f:
    json.dump(combined_data, f, indent=2)

print(f"Successfully concatenated {len(data1)} elements from {input_file1} and {len(data2)} elements from {input_file2}")
print(f"Total of {len(combined_data)} elements written to {output_file}")

# Define input file paths
input_file1 = 'VSP-main/blocks/b_percive_sft.json'
input_file2 = 'VSP-main/frozenlake/fl_percive_sft.json'
output_file = 'percive_sft.json'

# Read the first JSON file
with open(input_file1, 'r') as f:
    data1 = json.load(f)

# Read the second JSON file
with open(input_file2, 'r') as f:
    data2 = json.load(f)

# Concatenate the JSON data (assuming both are lists)
combined_data = data1 + data2

# Write the concatenated data to the output file
with open(output_file, 'w') as f:
    json.dump(combined_data, f, indent=2)

print(f"Successfully concatenated {len(data1)} elements from {input_file1} and {len(data2)} elements from {input_file2}")
print(f"Total of {len(combined_data)} elements written to {output_file}")