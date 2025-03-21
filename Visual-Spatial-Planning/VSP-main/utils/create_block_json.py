import os
import json
from pathlib import Path

def create_block_dataset():
    """
    Create a hierarchical JSON structure containing all blocks examples data.
    Scans through blocks directory, all task folders, example directories,
    and collects all the relevant information.
    """
    base_path = Path("VSP-main/blocks")
    all_examples = []
    
    # Find all task directories
    task_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    for task_dir in sorted(task_dirs):
        task_name = task_dir.name
        prompt_text_file = task_dir/"prompt-text"/"prompt-text.txt"
        example_dir = task_dir / "example"
        
        if not example_dir.exists():
            continue
        
        # Read the prompt text once
        prompt_text_content = None
        if prompt_text_file.exists():
            try:
                prompt_text_content = prompt_text_file.read_text().strip()
            except Exception as e:
                print(f"Error reading prompt text for {task_name}: {e}")
                
        # Find all level directories
        level_dirs = [d for d in example_dir.iterdir() if d.is_dir()]
        
        for level_dir in sorted(level_dirs):
            level_name = level_dir.name
            
            # Handle different level naming conventions
            if level_name.startswith("level_"):
                level_name = level_name.replace("level_", "")
            elif level_name.startswith("level"):
                level_name = level_name.replace("level", "")
            
            # For task-main, create a dictionary to store begin/end image pairs
            if task_name == "task-main":
                # First, collect all images from "begin" and "end" directories
                begin_files = {}
                end_files = {}
                begin_dir = level_dir / "begin"
                end_dir = level_dir / "end"
                
                if begin_dir.exists():
                    for img_file in begin_dir.glob("*.jpg"):
                        try:
                            example_id = int(img_file.stem)
                        except ValueError:
                            example_id = img_file.stem
                        begin_files[example_id] = str(img_file).replace("\\", "/")
                
                if end_dir.exists():
                    for img_file in end_dir.glob("*.jpg"):
                        try:
                            example_id = int(img_file.stem)
                        except ValueError:
                            example_id = img_file.stem
                        end_files[example_id] = str(img_file).replace("\\", "/")
                
                for example_id in begin_files.keys():
                    if example_id in end_files:
                        # Construct paths to additional file content
                        analysis_file = level_dir / f"""analysis{example_id}.txt"""
                        question_file = level_dir / f"""question{example_id}.txt"""
                        solution_file = level_dir / f"""sol_{example_id}.txt"""
                        
                        example = {
                            "task_name": task_name,
                            "level": level_name,
                            "example_id": example_id,
                            "image_path_begin": begin_files[example_id],
                            "image_path_end": end_files[example_id],
                            "prompt_text": prompt_text_content,
                        }
                        
                        # Read analysis content if file exists
                        if analysis_file.exists():
                            try:
                                with open(analysis_file, "r", encoding="utf-8") as f:
                                    example["analysis"] = f.read().strip()
                            except Exception as e:
                                print(f"Error reading analysis for {task_name}/{level_name}/{example_id}: {e}")
                        
                        # Read question content if file exists
                        if question_file.exists():
                            try:
                                with open(question_file, "r", encoding="utf-8") as f:
                                    example["question"] = f.read().strip()
                            except Exception as e:
                                print(f"Error reading question for {task_name}/{level_name}/{example_id}: {e}")
                        
                        # Read solution content if file exists
                        if solution_file.exists():
                            try:
                                with open(solution_file, "r", encoding="utf-8") as f:
                                    example["solution"] = f.read().strip()
                            except Exception as e:
                                print(f"Error reading solution for {task_name}/{level_name}/{example_id}: {e}")
                        
                        all_examples.append(example)
            
            else:
                # For other tasks, handle image directories
                possible_img_dirs = ["image_input"]
                
                for img_dir_name in possible_img_dirs:
                    img_dir = level_dir / img_dir_name
                    if not img_dir.exists():
                        continue
                    
                    # Find all image files in the directory
                    img_files = [f for f in img_dir.glob("*.jpg") if f.is_file()]
                    if not img_files:
                        img_files = [f for f in img_dir.glob("*.png") if f.is_file()]
                    
                    for img_file in sorted(img_files):
                        # Try to convert filename to int, but handle non-numeric filenames
                        try:
                            example_id = int(img_file.stem)
                        except ValueError:
                            example_id = img_file.stem
                        
                        # Create the base example object
                        example = {
                            "task_name": task_name,
                            "level": level_name,
                            "example_id": example_id,
                            "image_path": str(img_file).replace("\\", "/"),
                            "prompt_text": prompt_text_content,
                            #"annotation": level_dir/"annotation.txt",
                        }
                        
                        # Check for answer content
                        answer_dir = level_dir / "answer"
                        if answer_dir.exists():
                            answer_file = answer_dir / f"{example_id}.txt"
                            if answer_file.exists():
                                try:
                                    with open(answer_file, "r", encoding="utf-8") as f:
                                        example["answer"] = f.read().strip()
                                except Exception as e:
                                    print(f"Error reading answer for {task_name}/{level_name}/{example_id}: {e}")
                        
                        # Check for text_input content
                        text_input_dir = level_dir / "text_input"
                        if text_input_dir.exists():
                            text_file = text_input_dir / f"{example_id}.txt"
                            if text_file.exists():
                                try:
                                    with open(text_file, "r", encoding="utf-8") as f:
                                        example["text"] = f.read().strip()
                                except Exception as e:
                                    print(f"Error reading text for {task_name}/{level_name}/{example_id}: {e}")
                        
                        # Check for pure_text_rep_input content
                        pure_text_dir = level_dir / "pure_text_rep_input"
                        if pure_text_dir.exists():
                            pure_text_file = pure_text_dir / f"{example_id}.txt"
                            if pure_text_file.exists():
                                try:
                                    with open(pure_text_file, "r", encoding="utf-8") as f:
                                        example["pure_text"] = f.read().strip()
                                except Exception as e:
                                    print(f"Error reading pure_text for {task_name}/{level_name}/{example_id}: {e}")
                        
                        # Check for table_rep_input content
                        table_dir = level_dir / "table_rep_input"
                        if table_dir.exists():
                            table_file = table_dir / f"{example_id}.txt"
                            if table_file.exists():
                                try:
                                    with open(table_file, "r", encoding="utf-8") as f:
                                        example["table"] = f.read().strip()
                                except Exception as e:
                                    print(f"Error reading table for {task_name}/{level_name}/{example_id}: {e}")
                        
                        all_examples.append(example)

    # Write the dataset to a JSON file
    output_file = Path("blocks_examples_dataset.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)
    
    print(f"Created dataset with {len(all_examples)} examples at {output_file}")

if __name__ == "__main__":
    create_block_dataset()