import os
from datetime import datetime
import json
import re
import ast
compeletions =[[{'role': 'assistant', 'content': '<think>\nThe holes are located on the outer walls of the maze and on the main path leading to the goal. In a 6×8 maze, rows 0 and 7 have no holes, and rows 1 and 5 each have one hole, and row 8 has no holes. In addition, in row 8, the hole is at column 7, which is at the goal entrance. Therefore, there is a hole at row 8, column 7.\n</think>\n<answer> "Yes" </answer>'}], [{'role': 'assistant', 'content': '<think>\nWe need to find a hole in maze row 8, column 7.\n\n- We need to check all grid squares in this location.\n- The red square (8,7) is in cell [45, 35] in the array.\n- We examine the squares in positions [45, 34], [45, 35], [45, 36] and [45, 37].\n- The first cell is not the hole, the second cell is the goal, the third cell does not contain a hole, and the fourth cell is not the hole.\n\nThe hole is not found.\n\n</think>\n<answer>\n{"answer": "No"}\n</answer>'}]]#print(compeletions[1][0]["content"])
solution = ["Yes","Yes"]
def percive_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion's answer matches the solution for FrozenLake tasks."""
    contents = [completion[0]["content"] for completion in completions]
    #print(contents[0])
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    # Extract answer from <answer> tags

    answer_tag_pattern = r'<answer>\s*(.*?)\s*</answer>'

    
    for content, sol in zip(contents, solution):
         
      reward = 0.0

      try:
          # Extract text from <answer> tags
          #print(content)
          content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)

          if content_answer_match:
              answer_text = content_answer_match.group(1).strip()
              #print(answer_text)
              try:
                  # Try to parse as JSON
                  answer_json = json.loads(answer_text)
                  if type(answer_json) == dict:
                    model_answer = answer_json.get("answer", "")
                  else:
                      model_answer = answer_json
                  if model_answer =="":
                      model_answer = list(answer_json.values())[0]
              except json.JSONDecodeError:
                  # If not JSON, use the raw text
                  model_answer = answer_text
                  
                  if model_answer.startswith('"') and model_answer.endswith('"'):
                      model_answer = model_answer[1:-1]

              
              # Get the correct answer from solution
              correct_answer = sol
              #print(model_answer)
              # Compare answers (case-insensitive for text)
              if isinstance(model_answer, str) and isinstance(correct_answer, str):
                  if "," in sol:
                      model_answer_list = sorted([item.strip() for item in model_answer.split(",")])
                      correct_answer_list = sorted([item.strip() for item in correct_answer.split(",")])
              
                      if model_answer_list == correct_answer_list:
                          reward = 1.0

                  elif model_answer.strip().lower() == correct_answer.strip().lower():
                      reward = 1.0
                  elif correct_answer.strip().lower() in model_answer.strip().lower():
                      reward = 0.2

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

    #print(rewards)
    return rewards



print(percive_reward(compeletions, solution))
