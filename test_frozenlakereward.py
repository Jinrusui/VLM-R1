import os
import json
import re
from datetime import datetime

def frozenlake_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion's answer matches the solution for FrozenLake tasks."""
    contents = [completion[0]["content"] for completion in completions]
    
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    # Extract answer from <answer> tags
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    
    for content, sol in zip(contents, solution):
        
        reward = 0.0

        try:
            # Extract text from <answer> tags
            content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
            
            if content_answer_match:
                answer_text = content_answer_match.group(1).strip()
                
                try:
                    # Try to parse as JSON
                    answer_json = json.loads(answer_text)
                    model_answer = answer_json.get("answer", "")
                except json.JSONDecodeError:
                    # If not JSON, use the raw text
                    model_answer = answer_text
                
                # Get the correct answer from solution
                correct_answer = sol
                #print("model_answer",model_answer)
                #print("correct_answer",correct_answer)
                # Compare answers (case-insensitive for text)
                if isinstance(model_answer, str) and isinstance(correct_answer, str):
                    if "," in sol:
                        model_answer_list = sorted([item.strip() for item in model_answer.split(",")])
                        correct_answer_list = sorted([item.strip() for item in correct_answer.split(",")])
                
                        if model_answer_list == correct_answer_list:
                            reward = 1.0

                    elif model_answer.strip().lower() == correct_answer.strip().lower():
                        reward = 1.0

                elif model_answer == correct_answer:
                    reward = 1.0
        except Exception as e:
            # Leave reward as 0.0 in case of errors
            #print("pass")
            pass
            
        rewards.append(reward)
        
        # Log for debugging if needed
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} FrozenLake reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {json.dumps(sol)}\n")
    # print("model_answer:",model_answer)
    # print("correct_answer:",correct_answer)
    # print(model_answer == correct_answer)
    # print(rewards)
    return rewards



def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    pattern = r"<think>[\s\S]*?</think>\s*<answer>\s*\{\s*\"answer\"\s*:\s*(?:\[[\s\S]*?\]|\"[\s\S]*?\")\s*\}[\s\S]*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

completions = [[{"content": """<think>\nThe maze is represented as follows:\n```\n1st Move: [R]\n2nd Move: [R]\n3rd Move: [R]\n```\nThe player starts at the top-left corner and needs to reach the goal. The player needs to move right three times to reach the goal without entering a hole.\n\nSince there are no holes along the path to the goal, the action sequence L (left) is safe.\n\n</think>\n<answer>{"answer": "Yes"}</answer>"""}],
[{"content": """<think>\nTo determine if the action sequence R is safe, I need to see the pixelated image and assess its content.\n\nI count 4 holes (#) and 2 goals (*). The player starts at position (3, 3). The goal is at position (1, 4).\n\nThe sequence R likely means the player might move right. Since the action is on the grid and moving from a position to the right means entering a hole in both cases, it's unsafe.\n</think>\n<answer>\n{"answer":"No"}\n</answer>"""}],
]



# solution = [ {
#         "task_name": "task1",
#         "level": "level8",
#         "example_id": 3,
#         "image": "frozenlake/task1/example/level8/img/3.png",
#         "problem": "Is there a hole in row 5, column 4? \noutput from {Yes,No}",
#         "solution": "Yes"
#     },
#     {
#         "task_name": "task2",
#         "level": "level3",
#         "example_id": 0,
#         "image": "frozenlake/task2/example/level3/img/0.png",
#         "problem": "Determine the relative position of the player with respect to the goal using the directional indicators: {\"Above\", \"Below\", \"Left\", \"Right\"}.\n-Above / Below for row differences\n-Left / Right for column differences",
#         "solution": "Left,Above"
#     },
#     {
#         "task_name": "task3",
#         "level": "level3",
#         "example_id": 0,
#         "image": "frozenlake/task3/example/level3/img/0.png",
#         "problem": "From {A, B, C, D}, select the textual maze representation that exactly matches the given image.\n\nSymbols:\n- @ : Player\n- # : Hole\n- _ : Safe land\n- * : Goal\n(A)\n| | Col 1 | Col 2 | Col 3 | Col 4 | Col 5 | Col 6 | Col 7 |\n| Row 1 | _ | _ | _ | _ | # | _ | # |\n| Row 2 | _ | _ | # | _ | _ | # | # |\n| Row 3 | * | # | _ | # | _ | # | _ |\n| Row 4 | _ | _ | _ | _ | # | # | _ |\n| Row 5 | _ | _ | _ | _ | # | _ | _ |\n| Row 6 | _ | _ | _ | _ | _ | _ | # |\n| Row 7 | _ | _ | @ | # | _ | # | _ |\n\n(B)\n| | Col 1 | Col 2 | Col 3 |\n| Row 1 | _ | @ | _ |\n| Row 2 | _ | _ | _ |\n| Row 3 | _ | * | _ |\n\n(C)\n| | Col 1 | Col 2 | Col 3 | Col 4 | Col 5 | Col 6 | Col 7 | Col 8 |\n| Row 1 | _ | # | _ | _ | # | _ | _ | _ |\n| Row 2 | _ | _ | # | _ | _ | _ | # | # |\n| Row 3 | # | _ | _ | _ | @ | _ | _ | _ |\n| Row 4 | _ | _ | _ | _ | _ | _ | _ | _ |\n| Row 5 | _ | _ | # | _ | _ | _ | _ | _ |\n| Row 6 | _ | # | _ | # | _ | _ | # | _ |\n| Row 7 | * | _ | _ | _ | _ | _ | _ | _ |\n| Row 8 | _ | _ | _ | _ | _ | # | _ | _ |\n\n(D)\n| | Col 1 | Col 2 | Col 3 |\n| Row 1 | @ | _ | _ |\n| Row 2 | _ | * | _ |\n| Row 3 | _ | # | # |",
#         "solution": "D"
#     },
#     {
#         "task_name": "task4",
#         "level": "step1",
#         "example_id": 0,
#         "image": "frozenlake/task4/example/level_step1/img/0.png",
#         "problem": "Determine if the action sequence R is safe (avoids holes) in the maze shown in <TEST-IMAGE>, using the rules:\n - Player (@) moves via L/R/U/D but cannot enter holes (#).\n - Moving off-grid or into a hole fails.\n - Success requires reaching the goal (*).\n Output from {Yes,No}",
#         "solution": "Yes"
#     },]
solution = ["Yes","No"]
print(frozenlake_reward(completions, solution),format_reward(completions))