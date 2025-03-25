import json
import re

def extract_answer(text, tag="<Output>"):
    """Extract answer after specified tag"""
    match = re.search(f'{tag}\\s*(.*?)(?:\\n|$|")', text, re.IGNORECASE)
    if match:
        return "<answer>" + json.dumps({"Answer": match.group(1).strip()}) + "</answer>"
    return ""

def convert_to_standard_format(text, task_type=None):
    """
    Convert different task response formats to a standardized format with <think> and <answer> sections.
    
    Args:
        text (str): The original text response
        task_type (int, optional): Task type (1, 2, 3, or 4) for specific task handling
        
    Returns:
        str: The converted text in standard format (<think>...</think>\n<answer>{"Answer": ...}</answer>)
    """
    # Extract the think part (everything before the output/answer section)
    think_part = text
    
    # Handle different prefixes for the thinking section
    for prefix in ["<Interpret>", "<Analysis>"]:
        if text.startswith(prefix):
            think_part = text.replace(prefix, "<think>", 1)
            break
    
    # If no prefix was found, add <think> if needed
    if not think_part.startswith("<think>"):
        think_part = "<think>" + think_part
    
    # Extract output/answer part
    output_text = ""
    if "<Output>" in think_part:
        parts = think_part.split("<Output>", 1)
        think_part = parts[0]
        output_text = parts[1].strip()
    
    # Make sure think part ends properly
    if not think_part.endswith("</think>"):
        think_part = think_part.strip() + "</think>"
    
    # Prepare answer JSON based on task type
    answer_json = {}
    
    if task_type == 1 or (output_text and any(color in output_text.lower() for color in ["red", "blue", "green", "yellow", "purple", "orange"])):
        # Task 1: Color identification
        color = output_text.strip()
        # Extract just the color name if needed
        for c in ["red", "blue", "green", "yellow", "purple", "orange"]:
            if c in color.lower():
                answer_json = {"Answer": c}
                break
        if not answer_json:
            answer_json = {"Answer": color}
    
    elif task_type == 2 or (output_text and any(letter in output_text for letter in "ABCD")):
        # Task 2: Spatial relation (A, B, C, D choices)
        choice = output_text.strip()
        # Extract just the letter
        for c in "ABCD":
            if c in choice:
                answer_json = {"Answer": c}
                break
        if not answer_json:
            answer_json = {"Answer": choice}
    
    elif task_type == 3 or (output_text and any(letter in output_text for letter in "ABCD")):
        # Task 3: Multiple choice (A, B, C, D)
        choice = output_text.strip()
        # Extract just the letter
        for c in "ABCD":
            if c in choice:
                answer_json = {"Answer": c}
                break
        if not answer_json:
            answer_json = {"Answer": choice}
    
    elif task_type == 4 or (output_text and ("Yes" in output_text or "No" in output_text)):
        # Task 4: Yes/No validation
        yes_no = "Yes" if "Yes" in output_text else "No"
        answer_json = {"Answer": yes_no}
    
    else:
        # Default handling if no specific pattern is recognized
        answer_json = {"Answer": output_text.strip() if output_text else ""}
    
    # Format the answer part as JSON
    
    answer_part = "<answer>"+json.dumps(answer_json)+"</answer>"    
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
        new_item["problem"] = "In this task, you will see a photo of blocks. You will analyze the block configuration and then answer a question regarding the color of blocks in a specific place. Since coding is not within your skill set, your approach relies on logical reasoning.\n\n## Game Setup\n- Each block has a unique color (blue, yellow, purple, orange, red, green).\n- Blocks are stacked vertically in a stack, forming multiple stacks.\n- In the questions, the position of the blocks is represented as \"Stack s, Level l\". The stack number is counted from left to right, and the level number is counted from bottom to top.\n\n## Procedure and Output\nYour output should follow this format: \n1. First, analyze the block configuration;\n2. Then, answer the question with the format <Color>, where <Color> is one of (blue, yellow, purple, orange, red, green). For example, \"red\".\n\nNow please answer the following question based on the given image:\n\n" + item.get("text")
        #item.get("question", "") + " \noutput from {Yes,No}"
        # Extract from answer field after <Output> tag
        if "answer" in item:
            new_item["solution"] = convert_to_standard_format(item["answer"],1)
    
    elif task_name == "task2":
        new_item["problem"] = "In this task, you will see a photo of blocks. You will analyze the block configuration and then answer a question regarding the spatial relation of two specified blocks. Since coding is not within your skill set, your approach relies on logical reasoning.\n\n## Game Setup\n- Each block has a unique color (blue, yellow, purple, orange, red, green).\n- Blocks are stacked vertically in a stack, forming multiple stacks.\n- The possible relations of two blocks include: (A) One block is directly above another block, and they are at the same stack; (B) One block is directly below another block, and they are at the same stack; (C) Two blocks are at different blocks; (D) At least one of the asked blocks does not exist in the image.\n\n## Procedure and Output\nYour output should follow this format: \n1. First, analyze the block configuration;\n2. Then, answer the question with the format <Choice>, where <Choice> is one of {A,B,C,D}. For example, \"A\".\n\nNow please answer the following question based on the given image:\n\n" + item.get("text")        #"Determine the relative position of the player with respect to the goal using the directional indicators: {\"Above\", \"Below\", \"Left\", \"Right\"}.\n-Above / Below for row differences\n-Left / Right for column differences"
        #"Determine the relative position of the player with respect to the goal using the directional indicators: {\"Above\", \"Below\", \"Left\", \"Right\"}.\n-Above / Below for row differences\n-Left / Right for column differences"
        # Extract from answer field after <Output> tag
        if "answer" in item:
            new_item["solution"] = convert_to_standard_format(item["answer"],2)
    
    elif task_name == "task3":
        new_item["problem"] = "You are a robot that sorts and organizes colored blocks by adding and removing them to stacks.\nYou can move them between stacks to produce a desired end state.\n\nIn this task, you will see a photo of blocks. This photo shows the beginning state of the blocks. You will see a photo of blocks. This photo shows the beginning state of the blocks. Meanwhile, you will be provided an action sequence about moving blocks. Your task is to determine if the provided action plan can be successfully executed. Since coding is not within your skill set, your approach relies on logical reasoning of the map.\n\n## Game Setup\n- The block configuration is presented in the image. You must view and interpret the image in order to determine which blocks are in which stack and determine the consequence of moving.\n- Each block has a unique color (blue, yellow, purple, orange, red, green).\n- Blocks are stacked vertically in a stack, forming multiple stacks.\n- A valid action can only move the top block of any stacks. Attempting to move lower blocks is considered an invalid move.\n- For the destination, a valid move can either (a) move the top block to the top of another stack, or (b) place the top block on the table, creating a new stack with just one block.\n\n## Procedure and Output\nYour output should follow this format:\n1. First, briefly analyze the block configuration, and check each action step by step to see if the provided step is valid.\n2. Then, answer the question with the format \"Yes\" or \"No\" to indicate if the action sequence is valid.\n\nHere is an example for the output:\n<Analysis> In the image, there are three stacks:\n- Stack 1: Purple block (alone)\n- Stack 2: Blue block (alone)\n- Stack 3: From bottom to top: Orange block, Red block\nThe first action \"move(red,table)\" is valid, because the red block is on top of a stack (stack 3 in this case), and the target is \"table\". After the first action, the state will become:\n- Stack 1: Purple block (alone)\n- Stack 2: Blue block (alone)\n- Stack 3: Orange block (alone)\n- Stack 4: Red block (alone)\nThe second action \"move(green,table)\" is invalid, because there is no green block.\nTherefore, the provided action sequence is invalid.\nNo\n\nNow please determine if the provided action sequence is valid given the input state.\n\nThe action sequence is:\n\n"+item.get("text")       #f"Determine if the action sequence {item.get('question', '')} is safe (avoids holes) in the maze shown in <TEST-IMAGE>, using the rules:\n - Player (@) moves via L/R/U/D but cannot enter holes (#).\n - Moving off-grid or into a hole fails.\n - Success requires reaching the goal (*).\n Output from {{Yes,No}}"
        #"From {A, B, C, D}, select the textual maze representation that exactly matches the given image.\n\nSymbols:\n- @ : Player\n- # : Hole\n- _ : Safe land\n- * : Goal" +f"""\n{item.get('question', '')}"""
        # Extract from analysis field after <Answer> tag
        if "answer" in item:
            new_item["solution"] = convert_to_standard_format(item["answer"],3)
    
    elif task_name == "task4":
        new_item["problem"] = "You are a maze-solving agent playing a pixelated maze videogame.\nMazes are presented on grid maps, where each tile can be empty land, or contain a player, hole, or goal.\nEach of the above tile types are represented as square pixel art images.\n\nIn this task, you will analyze a grid-based map and determine if a provided action plan is safe. A safe action plan avoids stepping into holes in the map.\n\n## Moving Rules\n- The action plan involves a series of moves: 'L' (left), 'R' (right), 'U' (up), or 'D' (down).\n- Each move transfers the player to the adjacent square in that direction, provided it is a safe square. The player cannot move more than one square at a time.\n- Moving off the edge of the map has no effect. The player will remain at the same square.\n- DO NOT MOVE INTO A HOLE! Falling into a hole results in defeat.\n- Locating at the grid containing the goal results in victory.\n\n## Procedure and Output\nYour output should include the following parts:\n1. First, interpret map. List where the player is at now, where is the goal, and where are the holes.\n2. Then, reasoning by following the given action plan. At each step, you should check:\n    (a) Where the current move leads the player to (the row and column);\n    (b) What is in that grid. Is it a hole? Is it the goal? Is it an empty space?\n    (c) Determine if that is a safe action.\n3. Output if the action sequence is safe using \"Yes\" or \"No\". A safe action sequence should not include any unsafe actions.\n\nNow please determine if the action sequence is safe for the given maze and action sequence:\n\n"+ f""" {item.get('question', '')}"""
        #f"Determine if the action sequence {item.get('question', '')} is safe (avoids holes) in the maze shown in <TEST-IMAGE>, using the rules:\n - Player (@) moves via L/R/U/D but cannot enter holes (#).\n - Moving off-grid or into a hole fails.\n - Success requires reaching the goal (*).\n Output from {{Yes,No}}"
        # Extract from analysis field after <Output> tag
        if "answer" in item:
            new_item["solution"] = convert_to_standard_format(item["answer"],4)
    
    # Add the new item to the list if it has an answer
    if "solution" in new_item:
        new_data.append(new_item)

# Write the new JSON file
with open("b_percive_sft.json", "w", encoding="utf-8") as file:
    json.dump(new_data, file, indent=4)

print("Conversion completed successfully!")