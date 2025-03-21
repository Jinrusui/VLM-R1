import json
import re

def extract_answer(text, tag="<Output>"):
    """Extract answer after specified tag"""
    match = re.search(f'{tag}\\s*(.*?)(?:\\n|$|")', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""
def convert_to_standard_format(text,solution):
    """
    Convert a response with <Interpret> and <Output> sections to the standard format
    with <think> and <answer> sections with JSON formatting.

    Args:
        text (str): The original text response
        
    Returns:
        str: The converted text in standard format
    """
    # Extract the think part (everything before <Output>)
    if "<Output>" in text:
        think_part = text.split("<Output>")[0]
    else:
        think_part = text
    
    # Replace <Interpret> with <think> if needed
    if think_part.startswith("<Analysis>"):
        think_part = think_part.replace("<Analysis>", "<think>", 1)
    elif not think_part.startswith("<think>"):
        think_part = "<think>" + think_part
    
    # Make sure the think part ends with </think>
    if not think_part.strip().endswith("</think>"):
        think_part = think_part.strip() + "</think>"
    
    # Extract the action plan from the output section
    action_plan = []
    if "<Output>" in text:
        output_section = text.split("<Output>")[1]
        # Look for "Action plan:" followed by a list of actions
        if "Action plan:" in output_section:
            actions_text = output_section.split("Action plan:")[1].strip()
            # Actions might be comma-separated or space-separated
            if "," in actions_text:
                action_plan = [a.strip() for a in actions_text.split(",")]
            else:
                action_plan = [a.strip() for a in actions_text.split()]
    
    # If no action plan found, try to extract it directly from the think part
    if not action_plan:
        # Look for actions in the Action Plan section
        if "<Action Plan>" in think_part:
            plan_section = think_part.split("<Action Plan>")[1].split("<")[0]
            moves = []
            for line in plan_section.split("\n"):
                if "Moving" in line and "(" in line and ")" in line:
                    # Extract action within parentheses
                    action = line.split("(")[1].split(")")[0]
                    moves.append(action)
            action_plan = moves
    
    # Format the answer part as JSON

    solution_dict = {"path": str(solution)}
    solution_str = str(solution_dict).replace("'", '"')  # 转换单引号为双引号
    answer_part = f"<answer>{solution_str}</answer>"

 
    # Combine the parts
    result = think_part + "\n" + answer_part
    
    return result

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
        if "analysis" in item:
            new_item["solution"] = convert_to_standard_format(item["analysis"],item["solution"])
    
    if "solution" in new_item:
        new_data.append(new_item)

# Write the new JSON file
with open("b_plan_sft.json", "w", encoding="utf-8") as file:
    json.dump(new_data, file, indent=4)

print("Conversion completed successfully!")





