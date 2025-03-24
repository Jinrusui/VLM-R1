# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import pathlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from utils.math import compute_score
from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from open_r1.trainer import VLMGRPOTrainer, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
import PIL
import json

from open_r1.vlm_modules import *

# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple
from transformers.utils import logging


logger = logging.get_logger(__name__)


def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        # print(111, 222, 333, 444, 555, 666, 777, 888, 999)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.
    """
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    image_folders: str = field(
        default=None,
        metadata={"help": "Paths to image folders, separated by ':'"},
    )
    arrow_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to arrow cache directory"},
    )
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    reward_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "Choose reward method: 'default', 'mcp', ..."
        },
    )

def plan_reward(completions, solution, **kwargs):
    """
    Reward function for planning tasks such as block movement and maze navigation.
    Evaluates if the completion contains a valid plan that achieves the goal.
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    def parse_maze_description(description):
        """Parse a textual maze description into an environment configuration."""
        env_config = {
            "map_size": 0,
            "start_position": [0, 0],
            "lake_positions": [],
            "gift_position": [0, 0]
        }
        
        # Extract map size
        size_match = re.search(r'(\d+)x(\d+)', description)
        if size_match:
            env_config["map_size"] = int(size_match.group(1))
        
        # Extract player position (start position)
        player_match = re.search(r'player is at: row (\d+), column (\d+)', description, re.IGNORECASE)
        if player_match:
            # Note: Converting to 0-indexed for internal processing
            env_config["start_position"] = [int(player_match.group(1))-1, int(player_match.group(2))-1]
        
        # Extract hole positions (lake positions)
        hole_matches = re.findall(r'Row (\d+), Column (\d+)', description)
        for match in hole_matches:
            # Convert to 0-indexed
            env_config["lake_positions"].append([int(match[0])-1, int(match[1])-1])
        
        # Extract goal position (gift position)
        goal_match = re.search(r'goal is at: Row (\d+), Column (\d+)', description, re.IGNORECASE)
        if goal_match:
            # Convert to 0-indexed
            env_config["gift_position"] = [int(goal_match.group(1))-1, int(goal_match.group(2))-1]
        
        return env_config
    
    def path_verifier(env_config, path):
        """Verify if a path is valid in the given environment."""
        map_size = env_config["map_size"]
        start_position = env_config["start_position"]
        lake_positions = set(map(tuple, env_config["lake_positions"]))
        gift_position = tuple(env_config["gift_position"])
        
        position = tuple(start_position)
        
        moves = {
            "L": (0, -1),
            "R": (0, 1),
            "U": (-1, 0),
            "D": (1, 0)
        }
        
        for move in path:
            if move not in moves:
                return "invalid_move"
            
            dx, dy = moves[move]
            new_position = (position[0] + dx, position[1] + dy)
            
            # Check if we're staying on the map (if not, position doesn't change)
            if not (0 <= new_position[0] < map_size and 0 <= new_position[1] < map_size):
                continue  # Player stays in same position when trying to move off the grid
            
            position = new_position
            
            # Check if we fell into a hole
            if position in lake_positions:
                return "failed"
            
            # Check if we reached the goal
            if position == gift_position:
                return "success"
        
        # If we completed the path but didn't reach the goal
        return "incomplete"
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        
        # Extract answer from content if it has think/answer tags
        content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        answer = content_match.group(1).strip() if content_match else content.strip()
        
        # Determine if this is a block movement task or maze path task
        if "move" in answer and "(" in answer and ")" in answer:
            # This is likely a block movement task
            # Parse the step-by-step plan from the answer
            steps = re.findall(r'move\([^)]+\)', answer)
            
            # Compare with solution (which could be the expected number of steps or exact plan)
            if isinstance(sol, int):
                # If solution specifies the expected number of steps
                reward = 1.0 if len(steps) == sol else 0.0
            else:
                # Compare the exact plans (ignoring whitespace)
                sol_steps = re.findall(r'move\([^)]+\)', sol)
                if len(steps) == len(sol_steps):
                    # Check if all steps match
                    all_match = True
                    for step, expected_step in zip(steps, sol_steps):
                        if step.replace(" ", "").lower() != expected_step.replace(" ", "").lower():
                            all_match = False
                            break
                    reward = 1.0 if all_match else 0.0
                    
        elif any(direction in answer.upper() for direction in ["U", "D", "L", "R"]):
            # This is likely a maze path task
            # Extract movement directions from the answer
            directions = []
            for c in answer.upper():
                if c in "UDLR":
                    directions.append(c)
            
            # Process the solution based on its format
            if isinstance(sol, dict) and "map_size" in sol and "start_position" in sol:
                # If the solution is a maze configuration with start/end positions
                result = path_verifier(sol, directions)
                if result == "success":
                    reward = 1.0
            
            elif isinstance(sol, str):
                # Check if the solution is a maze description that needs parsing
                if "map" in sol.lower() and "player" in sol.lower() and "goal" in sol.lower():
                    env_config = parse_maze_description(sol)
                    result = path_verifier(env_config, directions)
                    if result == "success":
                        reward = 1.0
                else:
                    # If solution is provided as a string of directions
                    sol_directions = []
                    for c in sol.upper():
                        if c in "UDLR":
                            sol_directions.append(c)
                    
                    reward = 1.0 if directions == sol_directions else 0.0
            
            elif isinstance(sol, int):
                # If solution specifies the expected number of steps
                reward = 1.0 if len(directions) == sol else 0.0

        rewards.append(reward)
        
        # Log for debugging if enabled
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Plan reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Answer: {answer}\n")
                f.write(f"Solution: {sol}\n")
                f.write(f"Directions: {directions if 'directions' in locals() else ''}\n")
                
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
            f.write(f"------------- {current_time} Format reward -------------\n")
            for content, match in zip(completion_contents, matches):
                f.write(f"Content: {content}\n")
                f.write(f"Has format: {bool(match)}\n")

    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": plan_reward,
    "format": format_reward,
}


@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False
    adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pre-trained adapter to resume training from"},
    )

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)
def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")

def main(script_args, training_args, model_args):
    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    print("using vlm module:", vlm_module_cls.__name__)
    question_prompt = vlm_module_cls.get_question_template(task_type="default")

    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    # Load the JSONL datasets
    import json
    from datasets import Dataset
    
    data_files = script_args.data_file_paths.split(":")
    image_folders = script_args.image_folders.split(":")
    
    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")
    
    if script_args.reward_method is None:
        accu_reward_methods = ["default"] * len(data_files)
    else:
        accu_reward_methods = script_args.reward_method.split(":")
        assert len(accu_reward_methods) == len(data_files), f"Number of reward methods must match number of data files: {len(accu_reward_methods)} != {len(data_files)}"

    
    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")
    
    all_data = []
    for data_file, image_folder, accu_reward_method in zip(data_files, image_folders, accu_reward_methods):
        with open(data_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                if 'image' in item:
                    if isinstance(item['image'], str):
                        # Store image path instead of loading the image
                        item['image_path'] = [os.path.join(image_folder, item['image'])]
                        del item['image'] # remove the image column so that it can be loaded later
                    elif isinstance(item['image'], list):
                        # if the image is a list, then it is a list of images (for multi-image input)
                        item['image_path'] = [os.path.join(image_folder, image) for image in item['image']]
                        del item['image'] # remove the image column so that it can be loaded later
                    else:
                        raise ValueError(f"Unsupported image type: {type(item['image'])}")
                # Remove immediate image loading
                item['problem'] = item['conversations'][0]['value'].replace('<image>', '')
                
                # Handle solution that could be a float or string
                solution_value = item['conversations'][1]['value']
                if isinstance(solution_value, str):
                    item['solution'] = solution_value.replace('<answer>', '').replace('</answer>', '').strip()
                else:
                    # If it's a float or other non-string type, keep it as is
                    item['solution'] = str(solution_value)
                
                del item['conversations']
                item['accu_reward_method'] = item.get('accu_reward_method', accu_reward_method) # if accu_reward_method is in the data jsonl, use the value in the data jsonl, otherwise use the defined value
                all_data.append(item)

    dataset = Dataset.from_list(all_data)

    def make_conversation_from_jsonl(example):
        if 'image_path' in example and example['image_path'] is not None:
            # Don't load image here, just store the path
            return {
                'image_path': [p for p in example['image_path']],  # Store path instead of loaded image
                'problem': example['problem'],
                'solution': f"<answer> {example['solution']} </answer>",
                'accu_reward_method': example['accu_reward_method'],
                'prompt': [{
                    'role': 'user',
                    'content': [
                        *({'type': 'image', 'text': None} for _ in range(len(example['image_path']))),
                        {'type': 'text', 'text': question_prompt.format(Question=example['problem'])}
                    ]
                }]
            }
        else:
            return {
                'problem': example['problem'],
                'solution': f"<answer> {example['solution']} </answer>",
                'accu_reward_method': example['accu_reward_method'],
                'prompt': [{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': question_prompt.format(Question=example['problem'])}
                    ]
                }]
            }

    # Map the conversations
    dataset = dataset.map(make_conversation_from_jsonl, num_proc=8)
   
    peft_config = None
    if model_args.use_peft:
        peft_config = get_peft_config(model_args)
        if model_args.adapter_path:
            print(f"Will load adapter from: {model_args.adapter_path}")
            peft_config.adapter_path = model_args.adapter_path

    
    # Split dataset for validation if requested
    splits = {'train': dataset}
    if script_args.val_split_ratio > 0:
        train_val_split = dataset.train_test_split(
            test_size=script_args.val_split_ratio
        )
        splits['train'] = train_val_split['train']
        splits['validation'] = train_val_split['test']

    # Select trainer class based on vlm_trainer argument
    trainer_cls = VLMGRPOTrainer
    print("using trainer:", trainer_cls.__name__)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=splits['train'],
        eval_dataset=splits.get('validation') if training_args.eval_strategy != "no" else None,
        peft_config=peft_config,
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
