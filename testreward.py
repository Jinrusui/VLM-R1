import os
from datetime import datetime
import json
import re
import ast
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
completions = [[{'content':"""<think>
To find the optimal path from the agent to the gift, while avoiding lakes, we can use a breadth-first search (BFS) algorithm. This algorithm explores all the nodes level by level, which ensures we find the shortest path to the gift.

1. Start at the agent's position.
2. Add the agent's position to the queue and mark it as visited.
3. Explore all possible moves (Up, Down, Left, Right) from the current position, avoiding lakes (holes).
4. Repeat the process until the gift is reached or the queue is empty.

Following this process, let's track the moves:

- Start at the agent's position (top-left corner).
- Possible moves: Up, Left.
- Move Up to the safe tile above the agent.
- Continue moving up until reaching the gift.

The path would be: Up, Up, Right, Right, Down, Down, Right, Right, Down.

</think>
<answer>
{"path": ["Right", "Right", "Right", "Down", "Down", "Down"]}
</answer>
"""}]]




solution ={
        "problem": "Given an image of a FrozenLake grid, where the agent's position, safe tiles, lakes (holes), and the goal (gift) are visible, generate an optimal path from the agent to the gift, avoiding lakes. Your answer should be a list of moves in ['Up', 'Down', 'Left', 'Right'] format.",
        "solution": {
            "map_size": 4,
            "start_position": [
                0,
                0
            ],
            "lake_positions": [
                [
                    1,
                    1
                ],
                [
                    1,
                    2
                ],
                [
                    2,
                    0
                ],
                [
                    2,
                    1
                ],
                [
                    3,
                    0
                ],
                [
                    3,
                    2
                ]
            ],
            "gift_position": [
                3,
                3
            ]
        },
        "image": "frozenlake_4x4_sample0.png"
    }
solution = [solution["solution"]]
#print(solution["map_size"])

print(path_reward(completions,solution))