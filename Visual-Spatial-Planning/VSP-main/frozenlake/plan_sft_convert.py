import json
import re

def extract_answer(text, tag="<Output>"):
    """Extract answer after specified tag"""
    match = re.search(f'{tag}\\s*(.*?)(?:\\n|$|")', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""
def convert_to_standard_format(text):
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
    if think_part.startswith("<Interpret>"):
        think_part = think_part.replace("<Interpret>", "<think>", 1)
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
    answer_part = f"""<answer>\n{{"path":{str(action_plan).replace("'", '"')}}}</answer>"""
    
    # Combine the parts
    result = think_part + "\n" + answer_part
    
    return result

# Load the original JSON file
with open("VSP-main/frozenlake/frozenlake_examples_dataset.json", "r") as file:
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
        "image": item.get("image_path", "")
    }
    
    # Configure each task according to the text file
    if task_name == "task-main":
        new_item["problem"] = "As a professional maze solver, your task is to analyze a grid-based map and devise an action plan that enables a player to reach the goal from the starting point without falling into any holes, using the fewest possible moves. Since coding is not within your skill set, your approach relies on logical reasoning of the map.\n\n## Game Setup\n- The game presents a fully observable grid-based map.\n- The player starts at a specified grid square, with the goal located elsewhere on the map.\n- Each grid square is either safe or contains a hole.\n- Your goal is to guide the player to the goal while avoiding holes.\n\n## Moving Rules\n- The action plan involves a series of moves: 'L' (left), 'R' (right), 'U' (up), or 'D' (down).\n- Each move transfers the player to the adjacent square in that direction, provided it is a safe square. The player cannot move more than one square at a time.\n- Moving off the edge of the map has no effect. The player will remain at the same square.\n- DO NOT MOVE INTO A HOLE! Falling into a hole results in defeat.\n- Locating at the grid containing the goal results in victory.\n\nPlease generate action plan for the given maze:"

        #"Analyze the given grid-based maze and devise the shortest action plan for the player (@) to reach the goal (*) without stepping into any holes (#) \nRules:\nAllowed moves: 'L' (left), 'R' (right), 'U' (up), 'D' (down)\nMoving into a hole results in failure.\nMoving off the grid has no effect.\nReaching the goal means success.\nProcedure:\nIdentify the positions of the player, goal, and holes.\nGenerate the shortest  step-by-step action plan to reach the goal safely.\nVerify the plan and output it as:\nAction plan: <PLAN> (e.g., Action plan: R,D,R,U).\nSolve the provided maze and generate the correct action plan"
        # Extract from answer field after <Output> tag
        if "answer" in item:
            new_item["solution"] = convert_to_standard_format(item["answer"])
    
    if "solution" in new_item:
        new_data.append(new_item)

# Write the new JSON file
with open("plan_sft.json", "w", encoding="utf-8") as file:
    json.dump(new_data, file, indent=4)

print("Conversion completed successfully!")





