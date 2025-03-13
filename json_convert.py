import json

# 读取 JSON 文件
input_path = r"E:\VLM-R1\frozenlake_samples2\frozenlake_coordinates.json"  # 你的 JSON 文件路径
output_path = r"E:\VLM-R1\frozenlake_samples2\frozenlake_format_coordinates.json"  # 输出 JSON 路径

with open(input_path, "r") as f:
    data = json.load(f)

formatted_data = []
problem_text = ("""Given an image of a FrozenLake grid, where the agent's position, 
                safe tiles, lakes (holes), and the goal (gift) are visible, generate an optimal path from the agent
                to the gift, avoiding lakes. Your answer should be a list of moves in ["Up", "Down", "Left", Right"] format.""")

# 遍历数据并重新格式化
for entry in data:
    solution = {
        "map_size": entry["map_size"],
        "start_position": entry["start_position"],
        "lake_positions": entry["lake_positions"],
        "gift_position": entry["gift_position"]
    }
    formatted_data.append({
        "problem": problem_text,
        "solution": solution,
        'image': entry['image_path']
    })

# 写入新 JSON 文件
with open(output_path, "w") as f:
    json.dump(formatted_data, f, indent=4)

print(f"Formatted data saved to: {output_path}")
