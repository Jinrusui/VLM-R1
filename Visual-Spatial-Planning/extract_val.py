import json
import os
import copy

def extract_val(input_path, output_train_path, output_val_path):
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Group examples by task and level
    task_level_examples = {}
    for example in data:
        if 'task_name' not in example or 'level' not in example or 'example_id' not in example:
            continue
            
        task_name = example['task_name']
        level = example['level']
        
        if task_name not in task_level_examples:
            task_level_examples[task_name] = {}
        
        if level not in task_level_examples[task_name]:
            task_level_examples[task_name][level] = []
        
        task_level_examples[task_name][level].append(example)
    
    # Sort examples by example_id and extract the last one for validation
    train_data = []
    val_data = []
    
    for task_name, levels in task_level_examples.items():
        for level, examples in levels.items():
            # Sort examples by example_id
            sorted_examples = sorted(examples, key=lambda x: int(x.get('example_id', 0)))
            
            # Check if we have examples
            if sorted_examples:
                # Last example goes to validation
                val_data.append(sorted_examples[-1])
                
                # All other examples go to training
                train_data.extend(sorted_examples[:-1])
    
    # Write to output files
    with open(output_train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(output_val_path, 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"Train data: {len(train_data)} examples")
    print(f"Validation data: {len(val_data)} examples")
    
    # Print extracted validation examples summary
    print("\nValidation examples:")
    for example in val_data:
        print(f"Task: {example.get('task_name', 'N/A')}, Level: {example.get('level', 'N/A')}, Example ID: {example.get('example_id', 'N/A')}")

if __name__ == "__main__":
    # Process all files
    files = [
        ('Visual-Spatial-Planning/percive_grpo.json', 'Visual-Spatial-Planning/percive_grpo_train.json', 'Visual-Spatial-Planning/percive_grpo_val.json'),
        ('Visual-Spatial-Planning/percive_sft.json', 'Visual-Spatial-Planning/percive_sft_train.json', 'Visual-Spatial-Planning/percive_sft_val.json'),
        ('Visual-Spatial-Planning/plan_grpo.json', 'Visual-Spatial-Planning/plan_grpo_train.json', 'Visual-Spatial-Planning/plan_grpo_val.json'),
        ('Visual-Spatial-Planning/plan_sft.json', 'Visual-Spatial-Planning/plan_sft_train.json', 'Visual-Spatial-Planning/plan_sft_val.json')
    ]
    
    for input_path, output_train_path, output_val_path in files:
        print(f"\nProcessing {input_path}...")
        if os.path.exists(input_path):
            extract_val(input_path, output_train_path, output_val_path)
        else:
            print(f"File not found: {input_path}")