import os
import json
from pathlib import Path

def create_maze_dataset():
    """
    Create a hierarchical JSON structure containing all maze examples data.
    Scans through maze directory, all task folders, example directories,
    and collects all the relevant information.
    """
    base_path = Path("VSP-main/frozenlake")
    all_examples = []
    
    # Find all task directories
    task_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    for task_dir in sorted(task_dirs):
        task_name = task_dir.name
        prompt_text = task_dir/"prompt-text"/"prompt-text.txt"
        example_dir = task_dir / "example"
        
        if not example_dir.exists():
            continue
        
        # Find all level directories
        level_dirs = [d for d in example_dir.iterdir() if d.is_dir()]
        
        for level_dir in sorted(level_dirs):
            level_name = level_dir.name
            
            # Handle different level naming conventions
            if level_name.startswith("level"):
                level_name = level_name.replace("level_", "")
            elif level_name.startswith("level"):
                level_name = level_name.replace("level", "")
            
            # Find all examples in the level
            img_dir = level_dir / "img"
            if not img_dir.exists():
                continue
            
            img_files = [f for f in img_dir.glob("*.png") if f.is_file()]
            
            for img_file in sorted(img_files):
                example_id = int(img_file.stem)  # Assuming filenames are 0.png, 1.png, etc.
                
                # Create the base example object
                example = {
                    "task_name": task_name,
                    "level": level_name,
                    "example_id": example_id,
                    "image_path": str(img_file).replace("\\", "/"),
                    "prompt_text": prompt_text.read_text().strip() if prompt_text.exists() else None,
                }
                
                # Look for question content
                question_dir = level_dir / "question"
                if question_dir.exists():
                    question_file = question_dir / f"{example_id}.txt"
                    if question_file.exists():
                        try:
                            with open(question_file, "r", encoding="utf-8") as f:
                                example["question"] = f.read().strip()
                        except:
                            pass
                
                # Look for answer content
                answer_dir = level_dir / "answer"
                if answer_dir.exists():
                    answer_file = answer_dir / f"{example_id}.txt"
                    if answer_file.exists():
                        try:
                            with open(answer_file, "r", encoding="utf-8") as f:
                                example["answer"] = f.read().strip()
                        except:
                            pass
                
                # Look for analysis content (found in task4)
                analysis_dir = level_dir / "analysis"
                if analysis_dir.exists():
                    analysis_file = analysis_dir / f"{example_id}.txt"
                    if analysis_file.exists():
                        try:
                            with open(analysis_file, "r", encoding="utf-8") as f:
                                example["analysis"] = f.read().strip()
                        except:
                            pass
                
                # Look for table content
                table_dir = level_dir / "table"
                if table_dir.exists():
                    table_file = table_dir / f"{example_id}.txt"
                    if table_file.exists():
                        try:
                            with open(table_file, "r", encoding="utf-8") as f:
                                example["table"] = f.read().strip()
                        except:
                            pass
                # Look for pure_text content
                pure_text_dir = level_dir / "pure_text"
                if pure_text_dir.exists():
                    pure_text_file = pure_text_dir / f"{example_id}.txt"
                    if pure_text_file.exists():
                        try:
                            with open(pure_text_file, "r", encoding="utf-8") as f:
                                example["pure_text"] = f.read().strip()
                        except:
                            pass
                #lok for special_text content
                special_text_dir = level_dir / "special_text"
                if special_text_dir.exists():
                    special_text_file = special_text_dir / f"{example_id}.txt"
                    if special_text_file.exists():
                        try:
                            with open(special_text_file, "r", encoding="utf-8") as f:
                                example["special_text"] = f.read().strip()
                        except:
                            pass
                # Look for text content
                text_dir = level_dir / "text"
                if text_dir.exists():
                    text_file = text_dir / f"{example_id}.txt"
                    if text_file.exists():
                        try:
                            with open(text_file, "r", encoding="utf-8") as f:
                                example["text"] = f.read().strip()
                        except:
                            pass

                all_examples.append(example)
    
    # Write the dataset to a JSON file
    output_file = Path("frozenlake_examples_dataset.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)
    
    print(f"Created dataset with {len(all_examples)} examples at {output_file}")

if __name__ == "__main__":
    create_maze_dataset()