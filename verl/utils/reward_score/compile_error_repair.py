'''
用于设计强化学习的奖励，首先提取模型输出的代码块，然后进行编译，如果编译成功，则奖励1，否则奖励0
'''

import re
import os
import tempfile
import subprocess
import json
import requests
import time


class LLMJudge:
    """大模型评判器，用于判断代码修复是否真正解决了编译错误"""
    
    def __init__(self, base_url=None, headers=None):
        """初始化LLM评判器
        
        Args:
            base_url: API端点URL
            headers: 请求头信息
        """
        self.base_url = base_url or 'http://localhost:7804/v1/chat/completions' #'http://10.55.26.91:31098/v1/chat/completions'
        self.headers = headers or {
            "Authorization": "TEST-46542881-54d4-4096-b93d-6d5a3db326ac",
            "Content-Type": "application/json"
        }
        
        self.judge_prompt_template = """你是一个C++编程专家，需要评判代码修复的质量。

原始错误代码：
```cpp
{original_error_code}
```

错误类型：{error_type}
错误描述：{error_type_detail}

修复后的代码：
```cpp
{fixed_code}
```

请评判这个修复是否真正解决了编译错误，而不是简单地删除了有问题的代码。

评判标准：
1. 修复后的代码是否保留了原代码的核心功能和逻辑
2. 修复是否针对性地解决了指定的错误类型
3. 修复是否是通过删除问题代码来"解决"问题（这种情况应该判定为不合格）
4. 修复后的代码是否在逻辑上完整和合理

请只回答 "合格" 或 "不合格"，不需要其他解释。"""

    def get_llm_response(self, prompt, temperature=0.1):
        """调用LLM API获取响应"""
        payload = {
            "model": "test",
            "max_tokens": 1024,  # 只需要简短回答
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": "你是一个C++编程专家，专门评判代码修复的质量。请只回答'合格'或'不合格'。"},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }

        try:
            response = requests.request("POST", self.base_url, json=payload, headers=self.headers, verify=False)
            response.raise_for_status()
            content = json.loads(response.text)['choices'][0]['message']['content']
            return content.strip()
        except Exception as e:
            print(f"LLM API调用错误: {e}")
            return None

    def judge_fix_quality(self, original_error_code, error_type, error_type_detail, fixed_code):
        """评判修复质量
        
        Args:
            original_error_code: 原始错误代码
            error_type: 错误类型
            error_type_detail: 错误详细描述
            fixed_code: 修复后的代码
            
        Returns:
            bool: True表示修复合格，False表示不合格
        """
        try:
            prompt = self.judge_prompt_template.format(
                original_error_code=original_error_code,
                error_type=error_type,
                error_type_detail=error_type_detail,
                fixed_code=fixed_code
            )
            
            response = self.get_llm_response(prompt)
            
            if response is None:
                print("LLM响应为空，默认判定为不合格")
                return False
                
            # 判断回答是否包含"合格"
            return "合格" in response and "不合格" not in response
            
        except Exception as e:
            print(f"LLM评判过程出错: {str(e)}")
            return False
        finally:
            # 避免触发API速率限制
            time.sleep(0.1)


def extract_cpp_code(generated_code):
    """从生成的文本中提取C++代码"""
    # 尝试提取```cpp代码块
    pattern = r"```cpp\n(.*?)```"
    match = re.search(pattern, generated_code, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 尝试提取```c++代码块
    pattern = r"```c\+\+\n(.*?)```"
    match = re.search(pattern, generated_code, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 尝试提取```代码块
    pattern = r"```\n(.*?)```"
    match = re.search(pattern, generated_code, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 如果没有代码块标记，返回原文本
    return generated_code.strip()

def compile_cpp_code(code):
    """使用GCC编译C++代码并获取编译结果"""
    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix='.cpp', mode='w', delete=False) as temp_file:
        temp_file.write(code)
        temp_file_path = temp_file.name

    try:
        # 尝试编译代码
        result = subprocess.run(
            ['g++', '-c', temp_file_path, '-std=c++11'],
            capture_output=True,
            text=True
        )
        
        # 返回编译结果
        return {
            'success': result.returncode == 0,
            'error_output': result.stderr if result.stderr else "",
            'stdout': result.stdout if result.stdout else ""
        }
    
    except Exception as e:
        return {
            'success': False,
            'error_output': f"编译过程出错: {str(e)}",
            'stdout': ""
        }
    
    finally:
        # 清理临时文件
        try:
            os.unlink(temp_file_path)
            remove_o_files(temp_file_path)
        except:
            pass

def remove_o_files(file_path):
    try:
        # 获取临时文件所在目录
        dir_path = os.path.dirname(file_path)
        # 遍历目录中所有文件
        dir_path = "/mnt/tenant-home_speed/zhaijucai/VERL/verl-main"
        for filename in os.listdir(dir_path):
            if filename.endswith('.o'):
                file_to_remove = os.path.join(dir_path, filename)
                os.unlink(file_to_remove)
        return True
    except Exception as e:
        print(f"删除.o文件时出错: {str(e)}")
        return False 

def remove_object_file(file_path):
    """删除编译产生的.o文件
    
    Args:
        file_path: 源文件路径（.cpp文件路径）
        
    Returns:
        bool: 删除是否成功
    """
    try:
        # 获取.o文件路径（将.cpp替换为.o）
        object_file = file_path[:-4] + '.o'
        if os.path.exists(object_file):
            os.unlink(object_file)
            return True
        return False
    except Exception as e:
        print(f"删除.o文件时出错: {str(e)}")
        return False

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


def compute_score(generated_text,ground_truth= None, success_score=1.0, failure_score=0.0, format_score=0.0):
    """计算C++代码编译成功的奖励分数
    
    参考 GSM8K 的评分函数结构，对C++代码编译结果进行评分。
    
    Args:
        generated_text: 模型生成的文本，可能包含C++代码
        success_score: 编译成功时的分数，默认为1.0
        failure_score: 编译失败时的分数，默认为0.0
        format_score: 能提取出代码但编译失败时的分数，默认为0.0
        
    Returns:
        float: 奖励分数
    """
    # 提取C++代码
    cpp_code = extract_cpp_code(generated_text)
    
    # 如果无法提取到有效代码，返回失败分数
    if not cpp_code or cpp_code.strip() == "":
        return failure_score
    
    # 编译代码
    compile_result = compile_cpp_code(cpp_code)
    
    # 根据编译结果返回相应分数
    if compile_result['success']:
        return success_score
    else:
        # 能提取到代码格式但编译失败，给予格式分数
        return format_score


def compute_score_with_details(generated_text, success_score=1.0, failure_score=0.0, format_score=0.0):
    """计算C++代码编译成功的奖励分数（带详细信息）
    
    扩展版本的评分函数，返回详细的编译信息。
    
    Args:
        generated_text: 模型生成的文本，可能包含C++代码
        success_score: 编译成功时的分数，默认为1.0
        failure_score: 编译失败时的分数，默认为0.0
        format_score: 能提取出代码但编译失败时的分数，默认为0.0
        
    Returns:
        dict: 包含分数和详细信息的字典
    """
    # 提取C++代码
    cpp_code = extract_cpp_code(generated_text)
    
    result = {
        'score': failure_score,
        'extracted_code': cpp_code,
        'compile_success': False,
        'compile_errors': "",
        'reason': ""
    }
    
    # 如果无法提取到有效代码，返回失败分数
    if not cpp_code or cpp_code.strip() == "":
        result['reason'] = "无法提取有效的C++代码"
        return result
    
    # 编译代码
    compile_result = compile_cpp_code(cpp_code)
    result['compile_success'] = compile_result['success']
    result['compile_errors'] = compile_result['error_output']
    
    # 根据编译结果返回相应分数
    if compile_result['success']:
        result['score'] = success_score
        result['reason'] = "编译成功"
    else:
        result['score'] = format_score
        result['reason'] = f"编译失败: {compile_result['error_output']}"
    
    return result


def compute_score_with_llm_judge(generated_text, ground_truth, llm_judge=None, 
                                llm_pass_score=0.9, compile_pass_score=0.1, 
                                failure_score=0.0):
    """基于LLM评判和编译验证的门槛式奖励计算
    
    实现门槛式奖励机制：
    1. 首先通过LLM评判代码修复质量（0.5分）
    2. 然后进行编译验证（额外0.5分）
    3. 总分最高1.0分
    
    Args:
        generated_text: 模型生成的文本，包含修复后的代码
        ground_truth: 包含原始错误信息的字典，需要包含：
                     - error_type: 错误类型
                     - error_type_detail: 错误详细描述  
                     - original_error_code: 原始错误代码
        llm_judge: LLM评判器实例，如果为None则创建默认实例
        llm_pass_score: LLM评判通过的分数，默认0.5
        compile_pass_score: 编译通过的额外分数，默认0.5
        failure_score: 失败时的分数，默认0.0
        
    Returns:
        dict: 包含详细评分信息的字典
    """
    result = {
        'total_score': failure_score,
        'llm_judge_score': 0.0,
        'compile_score': 0.0,
        'llm_judge_pass': False,
        'compile_pass': False,
        'extracted_code': '',
        'reason': '',
        'compile_errors': ''
    }
    
    # 检查ground_truth参数
    if not ground_truth or not isinstance(ground_truth, dict):
        result['reason'] = "ground_truth参数无效"
        return result
    
    required_keys = ['error_type', 'error_type_detail', 'original_error_code']
    for key in required_keys:
        if key not in ground_truth:
            result['reason'] = f"ground_truth缺少必要字段: {key}"
            return result
    
    # 提取修复后的代码
    fixed_code = extract_cpp_code(generated_text)
    result['extracted_code'] = fixed_code
    
    if not fixed_code or fixed_code.strip() == "":
        result['reason'] = "无法提取有效的C++代码"
        return result
    
    # 初始化LLM评判器
    if llm_judge is None:
        llm_judge = LLMJudge()
    
    # 第一阶段：LLM评判
    try:
        llm_judgment = llm_judge.judge_fix_quality(
            original_error_code=ground_truth['original_error_code'],
            error_type=ground_truth['error_type'],
            error_type_detail=ground_truth['error_type_detail'],
            fixed_code=fixed_code
        )
        
        result['llm_judge_pass'] = llm_judgment
        
        if llm_judgment:
            result['llm_judge_score'] = llm_pass_score
            result['total_score'] = llm_pass_score
        else:
            result['reason'] = "LLM评判未通过：修复质量不合格"
            return result
            
    except Exception as e:
        result['reason'] = f"LLM评判过程出错: {str(e)}"
        return result
    
    # 第二阶段：编译验证（只有LLM评判通过才进行）
    try:
        compile_result = compile_cpp_code(fixed_code)
        result['compile_pass'] = compile_result['success']
        result['compile_errors'] = compile_result['error_output']
        
        if compile_result['success']:
            result['compile_score'] = compile_pass_score
            result['total_score'] = llm_pass_score + compile_pass_score
            result['reason'] = "LLM评判和编译验证均通过"
        else:
            result['reason'] = f"LLM评判通过但编译失败: {compile_result['error_output']}"
            
    except Exception as e:
        result['reason'] = f"编译验证过程出错: {str(e)}"
    
    return result


def compute_score_with_llm_judge_simple(generated_text, ground_truth, llm_judge=None):
    """简化版本的LLM评判奖励函数，只返回总分
    
    Args:
        generated_text: 模型生成的文本
        ground_truth: 包含错误信息的字典
        llm_judge: LLM评判器实例
        
    Returns:
        float: 奖励分数（0.0-1.0）
    """
    result = compute_score_with_llm_judge(generated_text, ground_truth, llm_judge)
    return result['total_score']


# 使用示例和说明
"""
使用示例：

# 1. 基本使用（自动创建LLM评判器）
ground_truth = {
    'error_type': 'syntax_error',
    'error_type_detail': '缺少分号',
    'original_error_code': 'int main() { return 0 }'
}

generated_text = '''
修复后的代码：
```cpp
int main() { 
    return 0; 
}
```
'''

# 获取详细评分结果
result = compute_score_with_llm_judge(generated_text, ground_truth)
print(f"总分: {result['total_score']}")
print(f"LLM评判: {'通过' if result['llm_judge_pass'] else '不通过'}")
print(f"编译验证: {'通过' if result['compile_pass'] else '不通过'}")

# 或者只获取总分
score = compute_score_with_llm_judge_simple(generated_text, ground_truth)
print(f"奖励分数: {score}")

# 2. 自定义LLM评判器
custom_judge = LLMJudge(
    base_url='your_api_endpoint',
    headers={'Authorization': 'your_token'}
)

result = compute_score_with_llm_judge(
    generated_text, 
    ground_truth, 
    llm_judge=custom_judge
)

# 3. 自定义奖励分数
result = compute_score_with_llm_judge(
    generated_text, 
    ground_truth,
    llm_pass_score=0.6,      # LLM评判通过得0.6分
    compile_pass_score=0.4,  # 编译通过额外得0.4分
    failure_score=0.0
)

奖励机制说明：
- 门槛式奖励：必须先通过LLM评判才能进行编译验证
- LLM评判（0.5分）：判断修复是否真正解决问题，而非删除代码
- 编译验证（额外0.5分）：确保修复后代码能够编译通过
- 总分范围：0.0-1.0

ground_truth字段说明：
- error_type: 错误类型（必需）
- error_type_detail: 错误详细描述（必需）
- original_error_code: 原始错误代码（必需）
"""