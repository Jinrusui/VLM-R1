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
    if not task_name or task_name not in ["task1", "task2", "task3", "task4"]:
        continue
    
    new_item = {
        "task_name": task_name,
        "level": item.get("level", ""),
        "example_id": item.get("example_id", ""),
        "image": item.get("image_path", "")
    }
    
    # Configure each task according to the text file
    if task_name == "task1":
        new_item["problem"] = "In this task, you will analyze a maze to determine if there is a hole in a specific position.\nThe following figure illustrates the appearances of the player, holes, lands, and the goal within the maze. You will need to focus on the appearance of the hole.\n\nYour output should be: \"No\" or \"Yes\", depending on whether there is a hole at the specified position.\n\nNow you will analyze the given maze and answer the question:\n\n "+ item.get("question", "")
        #item.get("question", "") + " \noutput from {Yes,No}"
        # Extract from answer field after <Output> tag
        if "answer" in item:
            new_item["solution"] = extract_answer(item["answer"])
    
    elif task_name == "task2":
        new_item["problem"] = "In this task, you will analyze a maze to determine the relative positions of the player and the goal.\n\nTo describe their relative positions, use the directional indicators from {\"Above\", \"Below\", \"Left\", \"Right\"}. Remember that we should answer the player's position with respect to the goal, not the opposite.\n\nYour output should be two parts:\n1. Analyze the rows and columns of the player and the goal. \n2. Following your analysis, output answer as \"<Position>\". For example,\"Above,Left\" means the player is above and to the left of the goal, and \"Below\" means the player is below the goal. \nNote that you should not output \"Left\" or \"Right\" if the plyaer and the goal are at the same column, and similarly, you should not output \"Above\" or \"Below\" if the player and the goal are at the same row.\n\nNow you will analyze the following maze and determine the relative position of the player in relation to the goal.\n\n"
        #"Determine the relative position of the player with respect to the goal using the directional indicators: {\"Above\", \"Below\", \"Left\", \"Right\"}.\n-Above / Below for row differences\n-Left / Right for column differences"
        # Extract from answer field after <Output> tag
        if "answer" in item:
            new_item["solution"] = extract_answer(item["answer"])
    
    elif task_name == "task3":
        new_item["problem"] = "In this task, you will analyze a maze presented in an image. Later, you will be presented with four choices, each offering a textual representation of a candidate maze. You will need to choose the representation that exactly reflects the contents of the given image.\n\nThis is how the player, the holes (non-safe grid), the lands (safe grids), and the goals look like in a map:\n- The player is represented as \"@\"\n- The hole is represented as \"#\"\n- The safe grid is represented as \"_\"\n- The goal is represented as \"*\"\n- If the player is at the goal (at this case the game is solved), that grid is represented as \"%\"\n\nYour output should consist of two parts:\n1. First, analysis the input image and candidates.\n2. Following the reasoning process, output answer as \"<Choice>\", where \"<Choice>\" is one of {A,B,C,D}.\nImportant: Note that there will be only one correct answer. If you find no answer or multiple answers, you must go back and recheck your reasoning process. You are not allowed to provide 0 or more than 1 answer.\n\nNow answer the question below. Here are the image input and the textual candidates:\n\n" +f"""{item.get('question', '')}"""
        #"From {A, B, C, D}, select the textual maze representation that exactly matches the given image.\n\nSymbols:\n- @ : Player\n- # : Hole\n- _ : Safe land\n- * : Goal" +f"""\n{item.get('question', '')}"""
        # Extract from analysis field after <Answer> tag
        if "analysis" in item:
            new_item["solution"] = extract_answer(item["analysis"], "<Answer>")
    
    elif task_name == "task4":
        new_item["problem"] = "You are a maze-solving agent playing a pixelated maze videogame.\nMazes are presented on grid maps, where each tile can be empty land, or contain a player, hole, or goal.\nEach of the above tile types are represented as square pixel art images.\n\nIn this task, you will analyze a grid-based map and determine if a provided action plan is safe. A safe action plan avoids stepping into holes in the map.\n\n## Moving Rules\n- The action plan involves a series of moves: 'L' (left), 'R' (right), 'U' (up), or 'D' (down).\n- Each move transfers the player to the adjacent square in that direction, provided it is a safe square. The player cannot move more than one square at a time.\n- Moving off the edge of the map has no effect. The player will remain at the same square.\n- DO NOT MOVE INTO A HOLE! Falling into a hole results in defeat.\n- Locating at the grid containing the goal results in victory.\n\n## Procedure and Output\nYour output should include the following parts:\n1. First, interpret map. List where the player is at now, where is the goal, and where are the holes.\n2. Then, reasoning by following the given action plan. At each step, you should check:\n    (a) Where the current move leads the player to (the row and column);\n    (b) What is in that grid. Is it a hole? Is it the goal? Is it an empty space?\n    (c) Determine if that is a safe action.\n3. Output if the action sequence is safe using \"Yes\" or \"No\". A safe action sequence should not include any unsafe actions.\n\nNow please determine if the action sequence is safe for the given maze and action sequence:\n\n"+ f""" {item.get('question', '')}"""
        #f"Determine if the action sequence {item.get('question', '')} is safe (avoids holes) in the maze shown in <TEST-IMAGE>, using the rules:\n - Player (@) moves via L/R/U/D but cannot enter holes (#).\n - Moving off-grid or into a hole fails.\n - Success requires reaching the goal (*).\n Output from {{Yes,No}}"
        # Extract from analysis field after <Output> tag
        if "analysis" in item:
            new_item["solution"] = extract_answer(item["analysis"])
    
    # Add the new item to the list if it has an answer
    if "solution" in new_item:
        new_data.append(new_item)

# Write the new JSON file
with open("percive_grpo.json", "w", encoding="utf-8") as file:
    json.dump(new_data, file, indent=4)

print("Conversion completed successfully!")