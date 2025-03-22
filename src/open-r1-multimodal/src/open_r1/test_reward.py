import os
import json
import re
from datetime import datetime

# Import reward functions from grpo_pre
#from rewards import path_reward, format_reward, frozenlake_reward

# def test_path_reward():
#     """Test the path_reward function with different scenarios."""
#     print("\n=== Testing path_reward ===")
    
#     # FrozenLake environment configuration
#     frozenlake_config = {
#         "map_size": 4,
#         "start_position": [0, 0],
#         "lake_positions": [[1, 1], [1, 2], [2, 0], [2, 1], [3, 0], [3, 2]],
#         "gift_position": [3, 3]
#     }
    
#     # Successful path completion
#     successful_completion = [{
#         "content": """<think>
# To navigate from the start position to the goal while avoiding lakes, I'll analyze the grid.
# <presentation>
# The optimal path from start to goal while avoiding lakes is: Right, Right, Right, Down, Down, Down.
# </presentation>
# </think>
# <answer>
# {"path": ["Right", "Right", "Right", "Down", "Down", "Down"]}
# </answer>"""
#     }]
    
#     # Failed path completion (falls into lake)
#     failed_completion = [{
#         "content": """<think>
# Analyzing the grid to find a path.
# <presentation>
# The proposed path is: Down, Down, Right, Right, Right, Down.
# </presentation>
# </think>
# <answer>
# {"path": ["Down", "Down", "Right", "Right", "Right", "Down"]}
# </answer>"""
#     }]
    
#     # Incomplete path
#     incomplete_completion = [{
#         "content": """<think>
# Let me find a safe path through the maze.
# <presentation>
# A partial path would be: Right, Right.
# </presentation>
# </think>
# <answer>
# {"path": ["Right", "Right"]}
# </answer>"""
#     }]
    
#     # Invalid format
#     invalid_format_completion = [{
#         "content": """<think>
# I think the solution is as follows.
# <presentation>
# The path is: Right, Right, Right, Down, Down, Down.
# </presentation>
# </think>
# <answer>
# Right, Right, Right, Down, Down, Down
# </answer>"""
#     }]
    
#     # Test with different completions
#     print("Successful path reward:", path_reward(successful_completion, [frozenlake_config]))
#     print("Failed path reward:", path_reward(failed_completion, [frozenlake_config]))
#     print("Incomplete path reward:", path_reward(incomplete_completion, [frozenlake_config]))
#     print("Invalid format reward:", path_reward(invalid_format_completion, [frozenlake_config]))

def test_format_reward():
    """Test the format_reward function with different format compliance."""
    print("\n=== Testing format_reward ===")
    
    # Correct format
    correct_format = [{
        "content": """<think>
To solve this maze problem, I need to find a safe path.
<presentation>
The optimal path is: Right, Down, Right, Down.
</presentation>
</think>
<answer>
{"path": ["Right", "Down", "Right", "Down"]}
</answer>"""
    }]
    
    # Missing presentation tag
    missing_presentation = [{
        "content": """<think>
To solve this maze problem, I need to find a safe path.
</think>
<answer>
{"path": ["Right", "Down", "Right", "Down"]}
</answer>"""
    }]
    
    # Invalid JSON format in answer
    invalid_json = [{
        "content": """<think>
To solve this maze problem, I need to find a safe path.
<presentation>
The optimal path is: Right, Down, Right, Down.
</presentation>
</think>
<answer>
Right, Down, Right, Down
</answer>"""
    }]
    
    # Test with different formats
    print("Correct format reward:", format_reward(correct_format))
    print("Missing presentation reward:", format_reward(missing_presentation))
    print("Invalid JSON format reward:", format_reward(invalid_json))

def test_frozenlake_reward():
    """Test the frozenlake_reward function with examples from different tasks."""
    print("\n=== Testing frozenlake_reward ===")
    
    # Task examples from different FrozenLake tasks
    examples = [
        {
            "task_name": "task1",
            "level": "level3",
            "example_id": 0,
            "image_path": "frozenlake/task1/example/level3/img/0.png",
            "question": "Is there a hole in row 3, column 1?",
            "answer": "Yes"
        },
        {
            "task_name": "task2",
            "level": "level3",
            "example_id": 0,
            "image_path": "frozenlake/task2/example/level3/img/0.png",
            "question": "Determine the relative position of the player with respect to the goal",
            "answer": "Left,Above"
        },
        {
            "task_name": "task3",
            "level": "level3",
            "example_id": 0,
            "image_path": "frozenlake/task3/example/level3/img/0.png",
            "question": "Select the textual maze representation that matches the image",
            "answer": "D"
        },
        {
            "task_name": "task4",
            "level": "step1",
            "example_id": 0,
            "image_path": "frozenlake/task4/example/level_step1/img/0.png",
            "question": "Determine if the action sequence R is safe",
            "answer": "Yes"
        }
    ]
    
    # Task completion templates
    completions = [
        # Task1 - Yes/No question about hole
        {
            "content": """<think>
In this maze, I need to determine if there's a hole at row 3, column 1.
Looking at the image, I can see that position indeed contains a hole.
</think>
<answer>
{"answer": "Yes"}
</answer>"""
        },
        
        # Task2 - Relative position
        {
            "content": """<think>
To determine the relative position of the player to the goal, I need to analyze their coordinates.
The player is in a row above the goal and in a column to the left of the goal.
</think>
<answer>
{"answer": "Left,Above"}
</answer>"""
        },
        
        # Task3 - Multiple choice
        {
            "content": """<think>
After analyzing all options, I can see that option D matches the maze layout in the image.
</think>
<answer>
{"answer": "D"}
</answer>"""
        },
        
        # Task4 - Safety of action sequence
        {
            "content": """<think>
The given action sequence R moves the player right, which is a safe move in this maze.
</think>
<answer>
{"answer": "Yes"}
</answer>"""
        }
    ]
    
    # Test each task type
    for i, (example, completion) in enumerate(zip(examples, completions)):
        print(f"\nTask: {example['task_name']}")
        print(f"Question: {example['question']}")
        print(f"Ground Truth: {example['answer']}")
        reward = frozenlake_reward(completion, [example['answer']])
        print(f"Reward: {reward}")

if __name__ == "__main__":
    # Run all tests
    #test_path_reward()
    #test_format_reward()
    j = "{\"a\":\"A\"}"
    j = json.loads(j)
    print(j.get("a"))
    #test_frozenlake_reward()

# completions = [
#         # Task1 - Yes/No question about hole
#         {
#             "content": """<think>
# In this maze, I need to determine if there's a hole at row 3, column 1.
# Looking at the image, I can see that position indeed contains a hole.
# </think>
# <answer>
# {"answer": "Yes"}
# </answer>"""
#         },
        
#         # Task2 - Relative position
#         {
#             "content": """<think>
# To determine the relative position of the player to the goal, I need to analyze their coordinates.
# The player is in a row above the goal and in a column to the left of the goal.
# </think>
# <answer>
# {"answer": "Left,Above"}
# </answer>"""
#         },
        
#         # Task3 - Multiple choice
#         {
#             "content": """<think>
# After analyzing all options, I can see that option D matches the maze layout in the image.
# </think>
# <answer>
# {"answer": "D"}
# </answer>"""
#         },
        
#         # Task4 - Safety of action sequence
#         {
#             "content": """<think>
# The given action sequence R moves the player right, which is a safe move in this maze.
# </think>
# <answer>
# {"answer": "Yes"}
# </answer>"""
#         }
#     ]
# print(completions[0]["content"])