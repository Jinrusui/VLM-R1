import json
import re

def extract_answer(text, tag="<Output>"):
    """Extract answer after specified tag"""
    match = re.search(f'{tag}\\s*(.*?)(?:\\n|$|")', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

# Load the original JSON file
with open("frozenlake_examples_dataset.json", "r") as file:
    data = json.load(file)

# Initialize new data structure
new_data = []

# Process each item in the original JSON
for item in data:
    task_name = item.get("task_name")
    
    # Skip items without a valid task_name
    if not task_name or task_name not in ["task1", "task2", "task3", "task4"]:
        continue
    
    new_item = {
        "task_name": task_name,
        "level": item.get("level", ""),
        "example_id": item.get("example_id", ""),
        "image_path": item.get("image_path", "")
    }
    
    # Configure each task according to the text file
    if task_name == "task1":
        new_item["question"] = item.get("question", "") + " \noutput from {Yes,No}"
        # Extract from answer field after <Output> tag
        if "answer" in item:
            new_item["answer"] = extract_answer(item["answer"])
    
    elif task_name == "task2":
        new_item["question"] = "Determine the relative position of the player with respect to the goal using the directional indicators: {\"Above\", \"Below\", \"Left\", \"Right\"}.\n-Above / Below for row differences\n-Left / Right for column differences"
        # Extract from answer field after <Output> tag
        if "answer" in item:
            new_item["answer"] = extract_answer(item["answer"])
    
    elif task_name == "task3":
        new_item["question"] = "From {A, B, C, D}, select the textual maze representation that exactly matches the given image.\n\nSymbols:\n- @ : Player\n- # : Hole\n- _ : Safe land\n- * : Goal"
        # Extract from analysis field after <Answer> tag
        if "analysis" in item:
            new_item["answer"] = extract_answer(item["analysis"], "<Answer>")
    
    elif task_name == "task4":
        new_item["question"] = f"Determine if the action sequence {item.get('question', '')} is safe (avoids holes) in the maze shown in <TEST-IMAGE>, using the rules:\n - Player (@) moves via L/R/U/D but cannot enter holes (#).\n - Moving off-grid or into a hole fails.\n - Success requires reaching the goal (*).\n Output from {{Yes,No}}"
        # Extract from analysis field after <Output> tag
        if "analysis" in item:
            new_item["answer"] = extract_answer(item["analysis"])
    
    # Add the new item to the list if it has an answer
    if "answer" in new_item:
        new_data.append(new_item)

# Write the new JSON file
with open("converted_tasks.json", "w", encoding="utf-8") as file:
    json.dump(new_data, file, indent=4)

print("Conversion completed successfully!")