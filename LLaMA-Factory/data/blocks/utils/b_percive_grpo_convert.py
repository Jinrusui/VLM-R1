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
    if not task_name or task_name not in ["task1", "task2", "task3", "task4"]:
        continue
    
    new_item = {
        "task_name": task_name,
        "level": item.get("level", ""),
        "example_id": item.get("example_id", ""),
        "image": item.get("image_path", "").replace("VSP-main/","")
    }
    
    # Configure each task according to the text file
    if task_name == "task1":
        new_item["problem"] = "In this task, you will see a photo of blocks. You will analyze the block configuration and then answer a question regarding the color of blocks in a specific place. Since coding is not within your skill set, your approach relies on logical reasoning.\n\n## Game Setup\n- Each block has a unique color (blue, yellow, purple, orange, red, green).\n- Blocks are stacked vertically in a stack, forming multiple stacks.\n- In the questions, the position of the blocks is represented as \"Stack s, Level l\". The stack number is counted from left to right, and the level number is counted from bottom to top.\n\n## Procedure and Output\nYour output should follow this format: \n1. First, analyze the block configuration;\n2. Then, answer the question with the format <Color>, where <Color> is one of (blue, yellow, purple, orange, red, green). For example, \"red\".\n\nNow please answer the following question based on the given image:\n\n" + item.get("text")
        # Extract from answer field after <Output> tag
        if "answer" in item:
            new_item["solution"] = extract_answer(item["answer"])
    
    elif task_name == "task2":
        new_item["problem"] = "In this task, you will see a photo of blocks. You will analyze the block configuration and then answer a question regarding the spatial relation of two specified blocks. Since coding is not within your skill set, your approach relies on logical reasoning.\n\n## Game Setup\n- Each block has a unique color (blue, yellow, purple, orange, red, green).\n- Blocks are stacked vertically in a stack, forming multiple stacks.\n- The possible relations of two blocks include: (A) One block is directly above another block, and they are at the same stack; (B) One block is directly below another block, and they are at the same stack; (C) Two blocks are at different blocks; (D) At least one of the asked blocks does not exist in the image.\n\n## Procedure and Output\nYour output should follow this format: \n1. First, analyze the block configuration;\n2. Then, answer the question with the format <Choice>, where <Choice> is one of {A,B,C,D}. For example, \"A\".\n\nNow please answer the following question based on the given image:\n\n" + item.get("text")        #"Determine the relative position of the player with respect to the goal using the directional indicators: {\"Above\", \"Below\", \"Left\", \"Right\"}.\n-Above / Below for row differences\n-Left / Right for column differences"
        # Extract from answer field after <Output> tag
        if "answer" in item:
            new_item["solution"] = extract_answer(item["answer"])
    
    elif task_name == "task3":
        new_item["problem"] = "In this task, you will analyze an image containing several stacks of blocks. Later, you will be presented with four choices, each offering a textual representation of a block configuration. You will need to choose the configuration that exactly reflects the contents of the given image.\n\n## Game Setup\n- Each block has a unique color (blue, yellow, purple, orange, red, green).\n- Blocks are stacked vertically in a stack, forming multiple stacks.\n\n## Procedure and Output\nYour output should follow this format:\n1. First, analyze the block configuration in the image and candidates;\n2. Then, answer the question with the format <Choice>, where <Choice> is one of {A,B,C,D}. For example, \"A\".\n\nNow please choose the correct textual representation based on the given image and here are the textual candidates:\n\n" + item.get("text")    #"From {A, B, C, D}, select the textual maze representation that exactly matches the given image.\n\nSymbols:\n- @ : Player\n- # : Hole\n- _ : Safe land\n- * : Goal" +f"""\n{item.get('question', '')}"""
        # Extract from analysis field after <Answer> tag
        if "answer" in item:
            new_item["solution"] = extract_answer(item["answer"])
    
    elif task_name == "task4":
        new_item["problem"] = "You are a robot that sorts and organizes colored blocks by adding and removing them to stacks.\nYou can move them between stacks to produce a desired end state.\n\nIn this task, you will see a photo of blocks. This photo shows the beginning state of the blocks. You will see a photo of blocks. This photo shows the beginning state of the blocks. Meanwhile, you will be provided an action sequence about moving blocks. Your task is to determine if the provided action plan can be successfully executed. Since coding is not within your skill set, your approach relies on logical reasoning of the map.\n\n## Game Setup\n- The block configuration is presented in the image. You must view and interpret the image in order to determine which blocks are in which stack and determine the consequence of moving.\n- Each block has a unique color (blue, yellow, purple, orange, red, green).\n- Blocks are stacked vertically in a stack, forming multiple stacks.\n- A valid action can only move the top block of any stacks. Attempting to move lower blocks is considered an invalid move.\n- For the destination, a valid move can either (a) move the top block to the top of another stack, or (b) place the top block on the table, creating a new stack with just one block.\n\n## Procedure and Output\nYour output should follow this format:\n1. First, briefly analyze the block configuration, and check each action step by step to see if the provided step is valid.\n2. Then, answer the question with the format \"Yes\" or \"No\" to indicate if the action sequence is valid.\n\nHere is an example for the output:\n<Analysis> In the image, there are three stacks:\n- Stack 1: Purple block (alone)\n- Stack 2: Blue block (alone)\n- Stack 3: From bottom to top: Orange block, Red block\nThe first action \"move(red,table)\" is valid, because the red block is on top of a stack (stack 3 in this case), and the target is \"table\". After the first action, the state will become:\n- Stack 1: Purple block (alone)\n- Stack 2: Blue block (alone)\n- Stack 3: Orange block (alone)\n- Stack 4: Red block (alone)\nThe second action \"move(green,table)\" is invalid, because there is no green block.\nTherefore, the provided action sequence is invalid.\nNo\n\nNow please determine if the provided action sequence is valid given the input state.\n\nThe action sequence is:\n\n"+item.get("text")       #f"Determine if the action sequence {item.get('question', '')} is safe (avoids holes) in the maze shown in <TEST-IMAGE>, using the rules:\n - Player (@) moves via L/R/U/D but cannot enter holes (#).\n - Moving off-grid or into a hole fails.\n - Success requires reaching the goal (*).\n Output from {{Yes,No}}"
        # Extract from analysis field after <Output> tag
        if "answer" in item:
            new_item["solution"] = extract_answer(item["answer"])
    
    # Add the new item to the list if it has an answer
    if "solution" in new_item:
        new_data.append(new_item)

# Write the new JSON file
with open("b_percive_grpo.json", "w", encoding="utf-8") as file:
    json.dump(new_data, file, indent=4)

print("Conversion completed successfully!")