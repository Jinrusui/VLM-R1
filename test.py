import re

# def format_reward(completions, **kwargs):
#     """Reward function that checks if the completion has a specific format."""
#     # pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
#     pattern = r"<think>.*?</think>\s*<answer>\s*\{\s*\"path\"\s*:\s*\[.*\]\s*\}.*?</answer>"

#     #completion_contents = [completion[0]["content"] for completion in completions]
#     matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completions]
#     return [1.0 if match else 0.0 for match in matches]

completion = ["""<think>
<presentation>
</presentation> 
To find the optimal path from the agent to the gift, we need to navigate around the lakes and move strategically. Here's the step-by-step decision process:
1. The agent is at the top-left corner.
2. Move right to avoid the first lake.
3. Move right to avoid the second lake.
4. Move right to avoid the third lake.
5. Move down to move further away from the lakes.
6. Move down to reach the bottom row.
7. Move right to avoid the lake in the bottom-right corner.
8. Finally, move right to reach the goal (gift).

The moves needed are: Right, Right, Right, Down, Down, Right, Right.
</think>
<answer>
{"path": ["Right", "Right", "Right", "Down", "Down", "Down"]}
</answer>"""]


def format_reward(completions, **kwargs):
    """
    Reward function that checks if the completion follows the required structure:
    <think>
      ... (including <presentation>...</presentation>)
    </think>
    <answer>
      ...
    </answer>
    """
    # Regex pattern that enforces:
    # - A <think> block
    # - Which must contain a <presentation> block
    # - Then the </think> tag
    # - Followed by <answer> and </answer> tags
    pattern = (
        r"^<think>.*?<presentation>.*?</presentation>.*?</think>\s*"
        r"<answer>.*?</answer>$"
    )
    
    #completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completions]
    return [1.0 if match else 0.0 for match in matches]

print(format_reward(completion))  # Output: [1.0]



# import wandb
# wandb_table = wandb.Table(columns=["step", "prompt"])
# wandb.init()
# for i in range(100):
#     prompt = "prompt"   
#     wandb_table.add_data(i, "prompt")
# wandb.log({
#     "table": wandb_table})