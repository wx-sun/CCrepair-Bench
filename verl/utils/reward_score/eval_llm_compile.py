#这个文件的功能是用来调用llm模型，评价模型能否生成解决编译错误的代码片段，评判要求是llm生成的代码能够通过gcc编译，并且能够解决编译错误。请用并发的方式调用模型。

import json
import requests
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import subprocess
import tempfile
import os
from typing import Dict, List, Tuple
import statistics

'''
错误类型: {error_type}
错误描述: {error_type_detail}
'''

class LLMEvaluator:
    def __init__(self, base_url, headers):
        """初始化LLM评估器"""
        self.base_url = base_url
        self.headers = headers
        self.fix_prompt_template = """你是一个C++编程专家。
给定以下包含编译错误的C++代码：


有问题的代码：
```cpp
{error_code}
```

编译错误信息：
{compilation_error}

请修复这段代码中的编译错误，要求：
1. 保持代码的原始意图和功能
2. 只修复编译错误，不要添加不必要的功能
3. 确保修复后的代码能够成功编译
4. 如果需要添加头文件，请包含必要的头文件
5. 保持代码简洁明了

请直接返回修复后的完整代码，不需要其他解释。"""

    def get_llm_response(self, prompt, base_url, reason=True, temperature=0.1):
        """调用LLM API获取响应"""
        payload = {
            "model": "Qwen3-235B-A22B", #"test",
            "max_tokens": 4096,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": "你是一个C++编程专家，专门修复编译错误。"},
                {"role": "user", "content": prompt + "/no_think" if reason else prompt}
            ],
            "stream": False
        }

        try:
            response = requests.request("POST", base_url, json=payload, headers=self.headers, verify=False)
            response.raise_for_status()
            content = json.loads(response.text)['choices'][0]['message']['content']
            return content
        except Exception as e:
            print(f"API调用错误: {e}")
            return None

    def generate_fix(self, error_type, error_type_detail, error_code, compilation_error):
        """生成修复代码"""
        try:
            prompt = self.fix_prompt_template.format(
                error_type=error_type,
                error_type_detail=error_type_detail,
                error_code=error_code,
                compilation_error=compilation_error
            )
            
            # 调用API生成修复代码
            generated_fix = self.get_llm_response(
                prompt, 
                base_url=self.base_url, #'http://10.55.56.14:31135/v1/chat/completions', 
                reason=True
            )
            if generated_fix:
                return generated_fix.strip()
            return ""
            
        except Exception as e:
            print(f"生成修复代码时出错 ({error_type}): {str(e)}")
            return ""
        
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
            if os.path.exists(temp_file_path[:-4] + '.o'):
                os.unlink(temp_file_path[:-4] + '.o')
        except:
            pass

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

def evaluate_single_item(evaluator, item):
    """评估单个错误项目"""
    try:
        error_type = item['error_type']
        error_type_detail = item['error_type_detail']
        error_code = item['error_example_code']  #item['error_example_llm_code'] #  #item['original_code'] 
        compilation_error = item['error_example_llm_detail'] #item['reason']
        uuid = item['error_type'] #item['uuid'] #
        error_code_without_comments = remove_cpp_comments(error_code)
        # 生成修复代码
        fix_response = evaluator.generate_fix(
            error_type, 
            error_type_detail, 
            error_code_without_comments, 
            compilation_error
        )
        
        if not fix_response:
            return {
                'uuid': uuid,
                'error_type': error_type,
                'success': False,
                'reason': 'LLM未生成修复代码',
                'original_code': error_code,
                'fix_response': '',
                'fixed_code': '',
                'fixed_code_without_comments': '',
                'compilation_result': None
            }
        
        # 提取修复后的代码
        fixed_code = extract_cpp_code(fix_response)
        
        if not fixed_code:
            return {
                'uuid': uuid,
                'error_type': error_type,
                'success': False,
                'reason': '无法从LLM响应中提取代码',
                'original_code': error_code,
                'fix_response': fix_response,
                'fixed_code': '',
                'fixed_code_without_comments': '',
                'compilation_result': None
            }
        
        # 去掉注释后的代码
        fixed_code_without_comments = remove_cpp_comments(fixed_code)
        
        # 编译修复后的代码
        compilation_result = compile_cpp_code(fixed_code)
        
        # 判断是否修复成功
        success = compilation_result['success']
        reason = '修复成功' if success else f"编译失败: {compilation_result['error_output']}"
        
        return {
            'uuid': uuid,
            'error_type': error_type,
            'success': success,
            'reason': reason,
            'original_code': error_code,
            'fix_response': fix_response,
            'fixed_code': fixed_code,
            'fixed_code_without_comments': fixed_code_without_comments,
            'compilation_result': compilation_result
        }
        
    except Exception as e:
        return {
            'uuid': item.get('uuid', 'unknown'),
            'error_type': item.get('error_type', 'unknown'),
            'success': False,
            'reason': f'评估过程出错: {str(e)}',
            'original_code': item.get('error_example_llm_code', ''),
            'fix_response': '',
            'fixed_code': '',
            'fixed_code_without_comments': '',
            'compilation_result': None
        }

def calculate_statistics(results):
    """计算评估统计信息"""
    total = len(results)
    successful = sum(1 for r in results if r['success'])
    success_rate = successful / total if total > 0 else 0
    
    # 按错误类型统计
    error_type_stats = {}
    for result in results:
        error_type = result['error_type']
        if error_type not in error_type_stats:
            error_type_stats[error_type] = {'total': 0, 'success': 0}
        
        error_type_stats[error_type]['total'] += 1
        if result['success']:
            error_type_stats[error_type]['success'] += 1
    
    # 计算每个错误类型的成功率
    for error_type in error_type_stats:
        stats = error_type_stats[error_type]
        stats['success_rate'] = stats['success'] / stats['total'] if stats['total'] > 0 else 0
    
    return {
        'total_items': total,
        'successful_fixes': successful,
        'overall_success_rate': success_rate,
        'error_type_stats': error_type_stats
    }

def evaluate_llm_fixes(input_file, output_file, base_url, headers, max_workers=12):
    """评估LLM修复编译错误的能力"""
    # 读取输入数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # data = data["failed_items"]
    print(f"加载了 {len(data)} 个错误示例")
    
    # 初始化评估器
    evaluator = LLMEvaluator(base_url, headers)
    
    # 存储评估结果
    results = []
    
    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = []
        for item in data:
            future = executor.submit(evaluate_single_item, evaluator, item)
            futures.append(future)
        
        # 收集结果
        for future in tqdm(as_completed(futures), total=len(futures), desc="评估修复效果"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"处理任务时出错: {str(e)}")
    
    # 计算统计信息
    statistics_info = calculate_statistics(results)
    
    # 准备输出数据
    output_data = {
        'evaluation_results': results,
        'statistics': statistics_info,
        'metadata': {
            'input_file': input_file,
            'total_evaluated': len(results),
            'evaluation_time': time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    print(f"\n评估完成！")
    print(f"总共评估: {statistics_info['total_items']} 个项目")
    print(f"修复成功: {statistics_info['successful_fixes']} 个")
    print(f"总体成功率: {statistics_info['overall_success_rate']:.2%}")
    print(f"\n各错误类型成功率:")
    for error_type, stats in statistics_info['error_type_stats'].items():
        print(f"  {error_type}: {stats['success']}/{stats['total']} ({stats['success_rate']:.2%})")
    
    print(f"\n结果已保存到: {output_file}")

def main():
    """主函数"""
    # 配置参数
    input_file = '/home/10350334@zte.intra/Desktop/研发提效三阶段/medical_cpt/error_data/origin_error_summary_verify.json'#'error_data_with_llm_examples_qwen3_32b_3.json'
    # input_file = '/home/10350334@zte.intra/Desktop/研发提效三阶段/medical_cpt/error_data/origin_error_summary_verify.json'
    output_file = 'llm_fix_evaluation_results_origin_data_qwen235.json'
    
    # API配置
    base_url = 'http://10.55.26.91:31098/v1/chat/completions' #'http://10.55.42.83:31063/v1/chat/completions'
    headers = {
        "Authorization": "TEST-46542881-54d4-4096-b93d-6d5a3db326ac",
        "Content-Type": "application/json"
    }
    
    # 运行评估
    evaluate_llm_fixes(
        input_file=input_file,
        output_file=output_file,
        base_url=base_url,
        headers=headers,
        max_workers=12  # 并发数
    )

if __name__ == "__main__":
    main()