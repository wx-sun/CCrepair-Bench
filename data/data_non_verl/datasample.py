import json
import random

# 1. 读取原始数据集
input_file = 'data/data_non_verl/filtered_llm_examples_data_only_train2_deduplicated.json'  # 替换为你的文件路径
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 随机抽样 100 个样本
sampled_data = random.sample(data, min(100, len(data)))  # 防止总数不足100时报错

# 3. 保存抽样结果到新文件
output_file = 'data/data_non_verl/sampled_dataset.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(sampled_data, f, ensure_ascii=False, indent=2)

print(f"已成功从 {input_file} 中随机抽取 100 个样本并保存至 {output_file}")