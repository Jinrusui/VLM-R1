import json
import re

def extract_answer(text, tag="<Output>"):
    """Extract answer after specified tag"""
    match = re.search(f'{tag}\\s*(.*?)(?:\\n|$|")', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

# Load the original JSON file
with open("VSP-main/blocks/blocks_examples_dataset.json", "r") as file:
    data = json.load(file)

# Initialize new data structure
new_data = []

# Process each item in the original JSON
for item in data:
    task_name = item.get("task_name")
    
    # Skip items without a valid task_name
    if not task_name or task_name not in ["task-main"]:
        continue
    
    new_item = {
        "task_name": task_name,
        "level": item.get("level", ""),
        "example_id": item.get("example_id", ""),
        "image": [item.get("image_path_begin", ""),item.get("image_path_end", "")]
    }
    
    # Configure each task according to the text file
    if task_name == "task-main":
        new_item["problem"] = "You are a robot that sorts and organizes colored blocks by adding and removing them to stacks.\nYou can move them between stacks to produce a desired end state.\n\nIn this task, you will see two photos of blocks showing the beginning and end states. Your goal is to find the shortest movement plan to transition between these states.\n\n### Game Rules\n- Each block has a unique color (blue, yellow, purple, orange, red, green).\n- Blocks are stacked vertically, and you can only move the top block of a stack.\n- Moves can:\n - Place the top block onto the top of another stack, or\n - Place the top block on the table to create a new stack.\n- Invalid moves: Attempting to move non-top blocks.\n- Order of stacks does not matter. Configurations are equivalent if stacks contain the same blocks, regardless of their arrangement.\n\n### Output Format\nList moves step-by-step using the format move(SOURCE, TARGET), where:\n- SOURCE is the color of the block being moved (always the top block of a stack).\n- TARGET is either a destination stack (specified by the color of its top block) or table (to create a new stack).\n\n### Task\nGenerate the shortest plan to transform the beginning state into the end state shown in the images.\n\n"
        #"Analyze the given grid-based maze and devise the shortest action plan for the player (@) to reach the goal (*) without stepping into any holes (#) \nRules:\nAllowed moves: 'L' (left), 'R' (right), 'U' (up), 'D' (down)\nMoving into a hole results in failure.\nMoving off the grid has no effect.\nReaching the goal means success.\nProcedure:\nIdentify the positions of the player, goal, and holes.\nGenerate the shortest  step-by-step action plan to reach the goal safely.\nVerify the plan and output it as:\nAction plan: <PLAN> (e.g., Action plan: R,D,R,U).\nSolve the provided maze and generate the correct action plan"
        # Extract from answer field after <Output> tag
        #if "answer" in item:
        new_item["solution"] = item["solution"]


    

    if "solution" in new_item:
        new_data.append(new_item)

# Write the new JSON file
with open("b_plan_grpo.json", "w", encoding="utf-8") as file:
    json.dump(new_data, file, indent=4)

print("Conversion completed successfully!")