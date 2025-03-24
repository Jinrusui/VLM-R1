import os
import re
import json
import torch
from tqdm import tqdm
from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER_PATH = "/jinru/output_plan_direct/Qwen2.5-VL-3B-GRPO-PRE-lora-continued/checkpoint-400"#"/jinru/MLP-CW3/checkpoint-160"
DATA_PATH = "/jinru/VLM-R1/Visual-Spatial-Planning/plan_grpo_val.jsonl"
IMAGE_DIR = "/jinru/VLM-R1/Visual-Spatial-Planning/VSP-main"
OUTPUT_DIR = "evaluation_results"
BATCH_SIZE = 8
MAX_NEW_TOKENS = 512
STEPS = 400  # Use the step number from the checkpoint

def extract_plan_answer(content):
    """Extract the answer from the model's output"""
    # First try to find content within answer tags
    answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    # If no tags, return the whole content (stripped)
    return content.strip()

def evaluate_plan(content, solution_type="blocks"):
    """
    Evaluate if the generated plan is correct
    solution_type: "blocks" for block movement or "maze" for maze navigation
    """
    content = extract_plan_answer(content)
    
    if solution_type == "blocks":
        # For block movement tasks, check if the content has valid move() statements
        moves = re.findall(r'move\([^)]+\)', content)
        # Return the number of valid moves found (we'll compare with ground truth later)
        return len(moves) > 0, moves
    
    elif solution_type == "maze":
        # For maze navigation tasks, extract directions (U, D, L, R)
        directions = []
        for c in content.upper():
            if c in "UDLR":
                directions.append(c)
        # Return whether we found any valid directions and the directions themselves
        return len(directions) > 0, directions
    
    return False, None

def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load base model
    print(f"Loading base model: {MODEL_NAME}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
    )
    
    # Load LoRA adapter
    print(f"Loading adapter from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    
    # Load processor
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    processor.padding_side = 'left'
    # Load data
    print(f"Loading data from: {DATA_PATH}")
    with open(DATA_PATH, 'r') as f:
        data = [json.loads(line) for line in f]
    
    all_results = []
    
    for i in tqdm(range(0, len(data), BATCH_SIZE)):
        batch_data = data[i:i + BATCH_SIZE]
        batch_messages = []
        
        # Prepare batch inputs
        for item in batch_data:
            images = item["image"]
            conversation = item["conversations"][0]["value"]
            
            # Convert image paths to full paths
            image_paths = []
            for img in images if isinstance(images, list) else [images]:
                image_paths.append(f"file://{os.path.join(IMAGE_DIR, img)}")
            
            # Create message with images
            image_content = [{"type": "image", "image": path} for path in image_paths]
            text_content = {"type": "text", "text": conversation}
            
            message = [
                {
                    "role": "user",
                    "content": image_content + [text_content]
                }
            ]
            
            batch_messages.append(message)
        
        # Process batch
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            padding_side = 'left'
        )
        inputs = inputs.to("cuda:0")
        
        # Generate outputs
        generated_ids = model.generate(
            **inputs, 
            use_cache=True, 
            max_new_tokens=MAX_NEW_TOKENS, 
            do_sample=False
        )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_outputs = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Process results
        for item, output in zip(batch_data, batch_outputs):
            # Determine task type based on the content
            prompt = item["conversations"][0]["value"]
            task_type = "blocks" if "blocks" in prompt.lower() else "maze"
            
            # Evaluate the plan
            valid_plan, extracted_plan = evaluate_plan(output, task_type)
            
            # Get ground truth plan
            gt_plan = item["conversations"][1]["value"] if len(item["conversations"]) > 1 else ""
            
            # Store result
            result = {
                "id": item.get("id", 0),
                "task_type": task_type,
                "prompt": prompt,
                "model_output": output,
                "extracted_plan": extracted_plan,
                "ground_truth": gt_plan,
                "valid_plan": valid_plan
            }
            all_results.append(result)
    
    # Calculate metrics
    total = len(all_results)
    valid_plans = sum(1 for res in all_results if res["valid_plan"])
    
    # Count exact matches with ground truth
    exact_matches = 0
    for res in all_results:
        if not res["valid_plan"]:
            continue
            
        gt = res["ground_truth"]
        if res["task_type"] == "blocks":
            # For block movement, compare the move statements
            gt_moves = re.findall(r'move\([^)]+\)', gt)
            if res["extracted_plan"] == gt_moves:
                exact_matches += 1
        else:
            # For maze navigation, compare the directions
            gt_directions = []
            for c in gt.upper():
                if c in "UDLR":
                    gt_directions.append(c)
            if res["extracted_plan"] == gt_directions:
                exact_matches += 1
    
    # Print metrics
    print(f"Total examples: {total}")
    print(f"Valid plans: {valid_plans} ({valid_plans/total*100:.2f}%)")
    print(f"Exact matches: {exact_matches} ({exact_matches/total*100:.2f}%)")
    
    # Save results
    output_file = f"{OUTPUT_DIR}/planning_results_step{STEPS}.json"
    with open(output_file, "w") as f:
        json.dump({
            "metrics": {
                "total": total,
                "valid_plans": valid_plans,
                "valid_plan_rate": valid_plans/total,
                "exact_matches": exact_matches,
                "exact_match_rate": exact_matches/total,
            },
            "results": all_results
        }, f, indent=2)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()