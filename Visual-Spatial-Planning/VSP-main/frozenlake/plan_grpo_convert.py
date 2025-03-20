import json
import re

def extract_answer(text, tag="<Output>"):
    """Extract answer after specified tag"""
    match = re.search(f'{tag}\\s*(.*?)(?:\\n|$|")', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

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
            new_item["solution"] = extract_answer(item["answer"])
    
    # elif task_name == "task2":
    #     new_item["problem"] = "Determine the relative position of the player with respect to the goal using the directional indicators: {\"Above\", \"Below\", \"Left\", \"Right\"}.\n-Above / Below for row differences\n-Left / Right for column differences"
    #     # Extract from answer field after <Output> tag
    #     if "answer" in item:
    #         new_item["solution"] = extract_answer(item["answer"])
    
    # elif task_name == "task3":
    #     new_item["problem"] = "From {A, B, C, D}, select the textual maze representation that exactly matches the given image.\n\nSymbols:\n- @ : Player\n- # : Hole\n- _ : Safe land\n- * : Goal" +f"""\n{item.get('question', '')}"""
    #     # Extract from analysis field after <Answer> tag
    #     if "analysis" in item:
    #         new_item["solution"] = extract_answer(item["analysis"], "<Answer>")
    
    # elif task_name == "task4":
    #     new_item["problem"] = f"Determine if the action sequence {item.get('question', '')} is safe (avoids holes) in the maze shown in <TEST-IMAGE>, using the rules:\n - Player (@) moves via L/R/U/D but cannot enter holes (#).\n - Moving off-grid or into a hole fails.\n - Success requires reaching the goal (*).\n Output from {{Yes,No}}"
    #     # Extract from analysis field after <Output> tag
    #     if "analysis" in item:
    #         new_item["solution"] = extract_answer(item["analysis"])
    
    # Add the new item to the list if it has an answer
    if "solution" in new_item:
        new_data.append(new_item)

# Write the new JSON file
with open("plan_grpo.json", "w", encoding="utf-8") as file:
    json.dump(new_data, file, indent=4)

print("Conversion completed successfully!")