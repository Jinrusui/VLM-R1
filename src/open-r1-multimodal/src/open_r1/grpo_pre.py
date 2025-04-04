import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math

# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple
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


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )
    

# @dataclass
# class GRPOModelConfig(ModelConfig):
#     freeze_vision_modules: bool = False
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

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []

        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999

                for data in datasets:
                    json_path = data.get("json_path")
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        # Format into conversation
        def make_conversation(example):
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["problem"]},
                ],
            }
        # This is only for Grounding task
        QUESTION_TEMPLATE = "{Question} \n\nFirst output the thinking or analyzing process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
        

        def make_conversation_image(example):
            return {
                "prompt": [
                    # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                        ],
                    },
                ],
            }
        


        def make_conversation_multi_image(example):
            return {
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image1"},
                            {"type": "image2"},
                            {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                        ],
                    },
                ], }

        example = self.list_data_dict[i]
        image_root = self.script_args.image_root
        # Check if example['image'] is a list first
        if 'image' in example and isinstance(example['image'], list):
            # Handle multi-image case
            try:
                return {
                    'image1': Image.open(os.path.join(image_root, example['image'][0])).convert("RGB"),
                    'image2': Image.open(os.path.join(image_root, example['image'][1])).convert("RGB"),
                    'task_name': example["task_name"],
                    'problem': example['problem'],
                    'solution': example['solution'],
                    'prompt': make_conversation_multi_image(example)['prompt'],
                }
            except Exception as e:
                print(f"Error loading multi-image: {e}")
                # Fallback to randomly selecting another example
                return self.__getitem__(random.randint(0, len(self.list_data_dict)-1))
        elif 'image' in example:
            # Handle single image case
            image_path = os.path.join(image_root, example['image'])
            # In case the image is not found
            while not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, randomly selecting another image")
                new_index = random.randint(0, len(self.list_data_dict)-1)
                example = self.list_data_dict[new_index]
                # Check again if the new example has a list of images
                if isinstance(example['image'], list):
                    return self.__getitem__(new_index)
                image_path = os.path.join(image_root, example['image'])
            image = Image.open(image_path).convert("RGB")
            return {
                'image': image,
                'task_name': example["task_name"],
                'problem': example['problem'],
                'solution': example['solution'],
                'prompt': make_conversation_image(example)['prompt'],
            }
        else:
            # Handle text-only case
            return {
                'task_name': example["task_name"],
                'problem': example['problem'],
                'solution': example['solution'],
                'prompt': make_conversation(example)['prompt'],
            }



# def percive_reward(completions, solution, **kwargs):
#     """Reward function that checks if the completion's answer matches the solution for FrozenLake tasks."""
#     contents = [completion[0]["content"] for completion in completions]
    
#     rewards = []
#     current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
#     # Extract answer from <answer> tags
#     #answer_tag_pattern = r'<answer>(.*?)</answer>'
#     answer_tag_pattern = r'<answer>\s*(.*?)\s*</answer>'
    
#     for content, sol in zip(contents, solution):
        
#         reward = 0.0

#         try:
#             # Extract text from <answer> tags
#             content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
            
#             if content_answer_match:
#                 answer_text = content_answer_match.group(1).strip()
                
#                 try:
#                     # Try to parse as JSON
#                     answer_json = json.loads(answer_text)
#                     model_answer = answer_json.get("answer", "")
#                     if model_answer =="":
#                         model_answer = list(answer_json.values())[0]
#                 except json.JSONDecodeError:
#                     # If not JSON, use the raw text
#                     model_answer = answer_text
                
#                 # Get the correct answer from solution
#                 correct_answer = sol
#                 # Compare answers (case-insensitive for text)
#                 if isinstance(model_answer, str) and isinstance(correct_answer, str):
#                     if "," in sol:
#                         model_answer_list = sorted([item.strip() for item in model_answer.split(",")])
#                         correct_answer_list = sorted([item.strip() for item in correct_answer.split(",")])
                
#                         if model_answer_list == correct_answer_list:
#                             reward = 1.0

#                     elif model_answer.strip().lower() == correct_answer.strip().lower():
#                         reward = 1.0
#                     elif correct_answer.strip().lower() in model_answer.strip().lower():
#                         reward = 0.2

#                 elif model_answer == correct_answer:
#                     reward = 1.0
#         except Exception as e:
#             # Leave reward as 0.0 in case of errors
#             #print("pass")
#             pass
            
#         rewards.append(reward)
        
#         # Log for debugging if needed
#         if os.getenv("DEBUG_MODE") == "true":
#             log_path = os.getenv("LOG_PATH")
#             with open(log_path, "a", encoding='utf-8') as f:
#                 f.write(f"------------- {current_time} FrozenLake reward: {reward} -------------\n")
#                 f.write(f"Content: {content}\n")
#                 f.write(f"Solution: {json.dumps(sol)}\n")
#     # print("model_answer:",model_answer)
#     # print("correct_answer:",correct_answer)
#     # print(model_answer == correct_answer)
#     print(rewards)
#     return rewards
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



# def format_reward(completions, **kwargs):
#     """Reward function that checks if the completion has a specific format."""
#     # pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
#     pattern = r"<think>[\s\S]*?</think>\s*<answer>\s*\{\s*\"answer\"\s*:\s*(?:\[[\s\S]*?\]|\"[\s\S]*?\")\s*\}[\s\S]*?</answer>"
#     completion_contents = [completion[0]["content"] for completion in completions]
#     matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
#     return [1.0 if match else 0.0 for match in matches]


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format.
    It is compatible with both escaped and unescaped answer strings.
    """
    # Regular expression matches <think>...</think> followed by <answer>{ "answer": ... }</answer>
    pattern = r"<think>[\s\S]*?</think>\s*<answer>\s*\{\s*\"answer\"\s*:\s*(?:\[[\s\S]*?\]|\"[\s\S]*?\")\s*\}[\s\S]*?</answer>"
    
    # Extract content and replace escaped quotes (replace \" with ")
    completion_contents = [completion[0]["content"] for completion in completions]
    processed_contents = [content.replace('\\"', '"') for content in completion_contents]
    
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in processed_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": percive_reward,
    "format": format_reward,
}


def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    # Load the dataset
    dataset = LazySupervisedDataset(script_args.dataset_name, script_args)
    
    # Create PEFT config once and reuse it
    peft_config = None
    if model_args.use_peft:
        peft_config = get_peft_config(model_args)
        if model_args.adapter_path:
            print(f"Will load adapter from: {model_args.adapter_path}")
            peft_config.adapter_path = model_args.adapter_path

    trainer_cls = Qwen2VLGRPOTrainer
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=peft_config,  # Use the previously created peft_config
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        torch_dtype=model_args.torch_dtype,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":

    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)