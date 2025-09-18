import re
import os
import datasets
import json

#from verl.utils.hdfs_io import copy, makedirs
import argparse

# To extract the solution for each prompts in the dataset
# def extract_solution(solution_str):
# ...
def remove_cpp_comments(code: str) -> str:
    """删除C++代码中的注释
    
    删除以下类型的注释：
    1. 单行注释 //
    2. 多行注释 /* */
    """
    # 处理多行注释
    code = re.sub(r'/\*[\s\S]*?\*/', '', code)
    
    # 处理单行注释
    code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
    
    # 删除空行
    code = '\n'.join(line for line in code.splitlines() if line.strip())
    
    return code
    
def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str) # extract the solution after ####
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return final_solution

instruction_following = "Let's think step by step and output the final answer after \"####\"."

# add a row to each data item that represents a unique id
def make_map_fn(split):

    def process_fn(example, idx):
        # 提取编译错误代码和编译器报错信息
        error_code = example.get('error_example_llm_code', '')
        error_detail = example.get('error_example_llm_detail', '')
        
        # 去除代码中的注释
        cleaned_code = remove_cpp_comments(error_code)
        
        # 构造 prompt 内容
        prompt_content = f"""以下是一段包含编译错误的C++代码：

```cpp
{cleaned_code}
```

编译器报错信息：
```
{error_detail}
```

请修复这段代码中的编译错误，并提供修复后的完整代码。"""

        data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": prompt_content
            }],
            "ability": "code_repair",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                'index': idx,
                'uuid': example.get('uuid', ''),
                'error_type': example.get('error_type', ''),
                'error_type_detail': example.get('error_type_detail', ''),
                'original_error_code': error_code,
                'cleaned_error_code': cleaned_code,
                'error_detail': error_detail
                },
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'uuid': example.get('uuid', ''),
                'error_type': example.get('error_type', ''),
                'error_type_detail': example.get('error_type_detail', ''),
                'original_error_code': error_code,
                'cleaned_error_code': cleaned_code,
                'error_detail': error_detail
            }
        }
        return data

    return process_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/home/10354352/Desktop/结果/补充材料/verl-main/data/compile_error_verl_gt')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = '/home/10354352/Desktop/结果/补充材料/Dataset/filtered_llm_examples_data_only_train2_deduplicated.json'

    # 加载 JSON 数据
    with open(data_source, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 按 9:1 分割数据集
    total_size = len(data)
    train_size = int(total_size * 0.9)
    
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # 创建数据集
    train_dataset = datasets.Dataset.from_list(train_data)
    test_dataset = datasets.Dataset.from_list(test_data)

    # 处理数据集
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # 确保输出目录存在
    os.makedirs(local_dir, exist_ok=True)

    # 保存处理后的数据集
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"数据已保存到: {local_dir}")

    # # 如果需要复制到 HDFS
    # if hdfs_dir:
    #     makedirs(hdfs_dir)
    #     copy(src=local_dir, dst=hdfs_dir)