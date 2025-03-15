
import os
import json
import re
from datetime import datetime
####reward function####

def path_reward(completions, solution, **kwargs):
    def path_verifier(env_config, path):
        # 保持原有验证逻辑不变

        map_size = env_config["map_size"]
        start_position = env_config["start_position"]
        lake_positions = set(map(tuple, env_config["lake_positions"]))
        gift_position = tuple(env_config["gift_position"])
        
        position = tuple(start_position)
        
        moves = {
            "Left": (0, -1),
            "Right": (0, 1),
            "Up": (-1, 0),
            "Down": (1, 0)
        }
        
        for move in path:
            if move not in moves:
                return "invalid_move"
            
            dx, dy = moves[move]
            new_position = (position[0] + dx, position[1] + dy)
            
            if not (0 <= new_position[0] < map_size and 0 <= new_position[1] < map_size):
                return "out_of_bounds"
            
            position = new_position
            
            if position in lake_positions:
                return "failed"
            
            if position == gift_position:
                return "success"
        
        return "incomplete"

    # 保持与iou_reward一致的输入处理
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    # 解析<answer>标签中的路径
    answer_tag_pattern = r'<answer>(.*?)</answer>'
   
    for content, sol in zip(contents, solution):
        reward = 0.0

        try:

            content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)

            if content_answer_match:
                answer_json = json.loads(content_answer_match.group(1).strip())
                path = answer_json.get("path", [])
                result = path_verifier(sol, path)
                if result == "success":
                    reward = 1.0
                elif result == "failed":
                    reward = -0.1
                elif result == "out_of_bounds":
                    reward = -0.1
                elif result == "incomplete":
                    reward = -0.1
                elif result == "invalid_move":
                    reward = -0.1
        except Exception as e:
            # 调试时可打印错误信息
            #print(f"Error parsing path: {e}")
            pass

        rewards.append(reward)
        
        # 保持与iou_reward一致的日志格式
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Path reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {json.dumps(sol)}\n")
    
    return rewards


# def format_reward(completions, **kwargs):
#     """Reward function that checks if the completion has a specific format."""
#     # pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
#     pattern = r"<think>.*?</think>\s*<answer>\s*\{\s*\"path\"\s*:\s*\[.*\]\s*\}.*?</answer>"
#     completion_contents = [completion[0]["content"] for completion in completions]
#     matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
#     return [1.0 if match else 0.0 for match in matches]
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
                
                # Compare answers (case-insensitive for text)
                if isinstance(model_answer, str) and isinstance(correct_answer, str):
                    if model_answer.lower() == correct_answer.lower():
                        reward = 1.0
                elif model_answer == correct_answer:
                    reward = 1.0
        except Exception as e:
            # Leave reward as 0.0 in case of errors
            pass
            
        rewards.append(reward)
        
        # Log for debugging if needed
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} FrozenLake reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {json.dumps(sol)}\n")
    
    return rewards


# def format_reward_frozenlake(completions, **kwargs):
#     """Reward function that checks if the completion has the proper format for FrozenLake tasks."""
#     pattern = r"<think>.*?</think>\s*<answer>\s*(\{.*?\}|[A-Za-z,]+)\s*</answer>"
#     completion_contents = [completion[0]["content"] for completion in completions]
#     matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
#     return [1.0 if match else 0.0 for match in matches]

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    pattern = r"<think>.*?<presentation>.*?</presentation>.*?</think>\s*<answer>\s*\{\s*\"path\"\s*:\s*\[.*\]\s*\}.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]