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
llm_examples_data中的错误源码:{error_example_llm_code}
error_base_data中的错误源码:{error_example_code}

'''

class LLMJudge:
    def __init__(self, base_url, headers):
        self.base_url = base_url
        self.headers = headers
        self.judge_prompt_template = """你是一个C++编程专家和代码质量评审员。请评估一个代码修复是否真正解决了编译错误，而不是简单地删除了相关代码。

原始错误信息：
错误类型: {error_type}
错误描述: {error_type_detail}

原始有问题的代码：
```cpp
{original_code}
```

修复后的代码：
```cpp
{fixed_code}
```

请分析修复后的代码是否真正解决了编译错误，评估标准如下：

1. **真正修复**: 代码修正了语法错误、添加了缺失的头文件/声明、修复了类型不匹配等问题，保持了原有功能逻辑
2. **简单删除**: 通过删除出错的代码行、函数或功能来避免编译错误，但丢失了原有功能
3. **过度修改**: 大幅改变了原有逻辑或添加了不必要的代码
4. **无效修复**: 修复不正确或可能引入新的问题

请给出你的判断，格式如下：
判断结果: [真正修复/简单删除/过度修改/无效修复]
置信度: [0-100]
理由: [详细说明你的判断依据，包括具体的修复点分析]"""

    def get_llm_response(self, prompt, temperature=0.1):
        """调用LLM API获取响应"""
        payload = {
            "model": "test",
            "max_tokens": 4096,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": "你是一个C++编程专家和代码质量评审员。"},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }

        try:
            response = requests.request("POST", self.base_url, json=payload, headers=self.headers, verify=False)
            response.raise_for_status()
            content = json.loads(response.text)['choices'][0]['message']['content']
            return content
        except Exception as e:
            print(f"裁判API调用错误: {e}")
            return None

    def judge_fix_quality(self, error_type, error_type_detail, original_code, fixed_code):
        """判断修复质量"""
        try:
            prompt = self.judge_prompt_template.format(
                error_type=error_type,
                error_type_detail=error_type_detail,
                original_code=original_code,
                fixed_code=fixed_code
            )
            
            judge_response = self.get_llm_response(prompt)
            if not judge_response:
                return {
                    'judge_result': '无效判断',
                    'confidence': 0,
                    'reason': 'LLM裁判调用失败',
                    'raw_response': ''
                }
            
            # 解析裁判响应
            result = self.parse_judge_response(judge_response)
            result['raw_response'] = judge_response
            return result
            
        except Exception as e:
            return {
                'judge_result': '无效判断',
                'confidence': 0,
                'reason': f'判断过程出错: {str(e)}',
                'raw_response': ''
            }

    def parse_judge_response(self, response):
        """解析裁判响应"""
        try:
            # 提取判断结果
            judge_pattern = r"判断结果:\s*\[?([^\]\n]+)\]?"
            judge_match = re.search(judge_pattern, response)
            judge_result = judge_match.group(1).strip() if judge_match else "未知"
            
            # 提取置信度
            confidence_pattern = r"置信度:\s*\[?(\d+)\]?"
            confidence_match = re.search(confidence_pattern, response)
            confidence = int(confidence_match.group(1)) if confidence_match else 0
            
            # 提取理由
            reason_pattern = r"理由:\s*([^\n]+(?:\n[^\n]*)*)"
            reason_match = re.search(reason_pattern, response)
            reason = reason_match.group(1).strip() if reason_match else "未提供理由"
            
            return {
                'judge_result': judge_result,
                'confidence': confidence,
                'reason': reason
            }
            
        except Exception as e:
            return {
                'judge_result': '解析失败',
                'confidence': 0,
                'reason': f'响应解析出错: {str(e)}'
            }

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
            "model": "test", #"Qwen3-235B-A22B", #"test",
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
                reason=False
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
            remove_o_files(temp_file_path)
            # os.unlink(temp_file_path)
            # if os.path.exists(temp_file_path[:-4] + '.o'):
            #     os.unlink(temp_file_path[:-4] + '.o')
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
        
        # 根据数据来源获取错误代码和编译错误信息
        if 'error_example_wo_comments' in item:
            # 使用去掉注释后的代码
            error_code = item['error_example_wo_comments']
            error_code_without_comments = error_code
        elif 'error_example_llm_code' in item:
            # LLM示例数据
            error_code = item['error_example_llm_code']
            error_code_without_comments = remove_cpp_comments(error_code)
        elif 'error_example_code' in item:
            # 错误基础数据
            error_code = item['error_example_code']
            error_code_without_comments = remove_cpp_comments(error_code)
        else:
            error_code = item.get('original_code', '')
            error_code_without_comments = remove_cpp_comments(error_code)
        
        compilation_error = item.get('error_example_llm_detail', '')
        uuid = item.get('uuid', item.get('error_type', 'unknown'))
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

def evaluate_llm_fixes_merged(input_file, output_file, base_url, headers, max_workers=64):
    """
    评估LLM修复编译错误的能力 - 处理merged数据格式
    
    该函数是整个评估流程的第一步，负责：
    1. 加载包含编译错误的C++代码数据
    2. 使用LLM生成修复代码
    3. 通过GCC编译器验证修复是否成功
    4. 生成详细的评估报告
    
    Args:
        input_file (str): 输入的merged格式数据文件路径
                         文件应包含 'llm_examples_data' 和 'error_base_data' 两个字段
        output_file (str): 输出评估结果的JSON文件路径
        base_url (str): LLM API的基础URL地址
        headers (dict): API请求头，通常包含Authorization等认证信息
        max_workers (int): 并发处理的最大线程数，默认12
    
    Returns:
        dict: 包含以下结构的评估结果字典：
            {
                'evaluation_results': [
                    {
                        'uuid': '唯一标识符',
                        'error_type': '错误类型',
                        'success': True/False,  # 编译是否成功
                        'reason': '成功/失败原因',
                        'original_code': '原始错误代码',
                        'fix_response': 'LLM完整响应',
                        'fixed_code': '提取的修复代码',
                        'fixed_code_without_comments': '去除注释的修复代码',
                        'compilation_result': '编译详细结果'
                    },
                    ...
                ],
                'statistics': {
                    'overall': '总体统计信息',
                    'llm_examples': 'LLM示例数据统计',
                    'error_base': '错误基础数据统计'
                },
                'metadata': {
                    'input_file': '输入文件路径',
                    'total_evaluated': '评估总数',
                    'evaluation_time': '评估时间'
                }
            }
    
    功能流程：
        1. 数据加载：从merged格式文件中加载llm_examples_data和error_base_data
        2. 并发处理：使用线程池并发调用LLM生成修复代码
        3. 代码提取：从LLM响应中提取C++代码块
        4. 编译验证：使用GCC编译器验证修复代码是否能成功编译
        5. 统计分析：计算成功率、按错误类型分组统计等
        6. 结果保存：将完整结果保存为JSON文件
    
    注意事项：
        - 支持两种数据源：llm_examples_data（LLM生成的示例）和error_base_data（基础错误数据）
        - 会自动去除代码中的注释避免编译干扰
        - 编译失败的情况会记录详细错误信息
        - 函数执行完成后会打印详细的统计信息
    """
    # 读取输入数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取LLM示例数据和错误基础数据
    llm_examples_data = data.get('llm_examples_data', [])
    error_base_data = data.get('error_base_data', [])
    
    print(f"找到 {len(llm_examples_data)} 个LLM示例")
    print(f"找到 {len(error_base_data)} 个错误基础数据")
    
    # 合并所有数据
    all_data = []
    
    # 处理LLM示例数据
    for item in llm_examples_data:
        item['data_source'] = 'llm_examples'
        all_data.append(item)
    
    # 处理错误基础数据
    for item in error_base_data:
        item['data_source'] = 'error_base'
        all_data.append(item)
    
    print(f"总共加载了 {len(all_data)} 个错误示例")
    
    # 初始化评估器
    evaluator = LLMEvaluator(base_url, headers)
    
    # 存储评估结果
    results = []
    
    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = []
        for item in all_data:
            future = executor.submit(evaluate_single_item, evaluator, item)
            futures.append(future)
        
        # 收集结果
        for future in tqdm(as_completed(futures), total=len(futures), desc="评估修复效果"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"处理任务时出错: {str(e)}")
    
    # 创建数据源映射
    data_source_map = {}
    for item in all_data:
        item_uuid = item.get('uuid', item.get('error_type', 'unknown'))
        data_source_map[item_uuid] = item.get('data_source', 'unknown')
    
    # 按数据源分组统计
    llm_results = [r for r in results if data_source_map.get(r['uuid']) == 'llm_examples']
    error_results = [r for r in results if data_source_map.get(r['uuid']) == 'error_base']
    
    # 计算统计信息
    overall_statistics = calculate_statistics(results)
    llm_statistics = calculate_statistics(llm_results) if llm_results else None
    error_statistics = calculate_statistics(error_results) if error_results else None
    
    # 准备输出数据
    output_data = {
        'evaluation_results': results,
        'statistics': {
            'overall': overall_statistics,
            'llm_examples': llm_statistics,
            'error_base': error_statistics
        },
        'metadata': {
            'input_file': input_file,
            'total_evaluated': len(results),
            'llm_examples_count': len(llm_examples_data),
            'error_base_count': len(error_base_data),
            'evaluation_time': time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    print(f"\n评估完成！")
    print(f"总共评估: {overall_statistics['total_items']} 个项目")
    print(f"修复成功: {overall_statistics['successful_fixes']} 个")
    print(f"总体成功率: {overall_statistics['overall_success_rate']:.2%}")
    
    if llm_statistics:
        print(f"\nLLM示例数据统计:")
        print(f"  评估数量: {llm_statistics['total_items']}")
        print(f"  修复成功: {llm_statistics['successful_fixes']}")
        print(f"  成功率: {llm_statistics['overall_success_rate']:.2%}")
    
    if error_statistics:
        print(f"\n错误基础数据统计:")
        print(f"  评估数量: {error_statistics['total_items']}")
        print(f"  修复成功: {error_statistics['successful_fixes']}")
        print(f"  成功率: {error_statistics['overall_success_rate']:.2%}")
    
    print(f"\n结果已保存到: {output_file}")
    
    # 返回数据以供后续处理
    return output_data

def judge_single_item(judge, evaluation_result, original_data_map):
    """使用LLM-as-Judge判断单个修复项目的质量"""
    try:
        uuid = evaluation_result['uuid']
        
        # 如果编译失败，直接标记为无效修复
        if not evaluation_result['success']:
            return {
                'uuid': uuid,
                'error_type': evaluation_result['error_type'],
                'compile_success': False,
                'judge_result': '无效修复',
                'confidence': 100,
                'reason': f"代码编译失败: {evaluation_result['reason']}",
                'original_code': evaluation_result['original_code'],
                'fixed_code': evaluation_result['fixed_code'],
                'raw_judge_response': '',
                'data_source': original_data_map.get(uuid, {}).get('data_source', 'unknown')
            }
        
        # 获取错误类型和描述
        original_item = original_data_map.get(uuid, {})
        error_type = evaluation_result['error_type']
        error_type_detail = original_item.get('error_type_detail', '')
        
        # 使用LLM裁判评估修复质量
        judge_result = judge.judge_fix_quality(
            error_type=error_type,
            error_type_detail=error_type_detail,
            original_code=evaluation_result['original_code'],
            fixed_code=evaluation_result['fixed_code']
        )
        
        return {
            'uuid': uuid,
            'error_type': error_type,
            'compile_success': True,
            'judge_result': judge_result['judge_result'],
            'confidence': judge_result['confidence'],
            'reason': judge_result['reason'],
            'original_code': evaluation_result['original_code'],
            'fixed_code': evaluation_result['fixed_code'],
            'raw_judge_response': judge_result['raw_response'],
            'data_source': original_data_map.get(uuid, {}).get('data_source', 'unknown')
        }
        
    except Exception as e:
        return {
            'uuid': evaluation_result.get('uuid', 'unknown'),
            'error_type': evaluation_result.get('error_type', 'unknown'),
            'compile_success': evaluation_result.get('success', False),
            'judge_result': '判断失败',
            'confidence': 0,
            'reason': f'LLM裁判评估过程出错: {str(e)}',
            'original_code': evaluation_result.get('original_code', ''),
            'fixed_code': evaluation_result.get('fixed_code', ''),
            'raw_judge_response': '',
            'data_source': 'unknown'
        }

def calculate_judge_statistics(judge_results):
    """计算LLM裁判统计信息"""
    total = len(judge_results)
    
    # 按判断结果分类
    judge_categories = {}
    confidence_scores = []
    
    for result in judge_results:
        judge_result = result['judge_result']
        confidence = result['confidence']
        
        if judge_result not in judge_categories:
            judge_categories[judge_result] = 0
        judge_categories[judge_result] += 1
        
        if confidence > 0:
            confidence_scores.append(confidence)
    
    # 计算置信度统计
    avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0
    median_confidence = statistics.median(confidence_scores) if confidence_scores else 0
    
    # 按数据源分组统计
    data_source_stats = {}
    for result in judge_results:
        source = result['data_source']
        if source not in data_source_stats:
            data_source_stats[source] = {
                'total': 0,
                'judge_categories': {}
            }
        
        data_source_stats[source]['total'] += 1
        judge_result = result['judge_result']
        if judge_result not in data_source_stats[source]['judge_categories']:
            data_source_stats[source]['judge_categories'][judge_result] = 0
        data_source_stats[source]['judge_categories'][judge_result] += 1
    
    # 按错误类型分组统计
    error_type_stats = {}
    for result in judge_results:
        error_type = result['error_type']
        if error_type not in error_type_stats:
            error_type_stats[error_type] = {
                'total': 0,
                'judge_categories': {}
            }
        
        error_type_stats[error_type]['total'] += 1
        judge_result = result['judge_result']
        if judge_result not in error_type_stats[error_type]['judge_categories']:
            error_type_stats[error_type]['judge_categories'][judge_result] = 0
        error_type_stats[error_type]['judge_categories'][judge_result] += 1
    
    return {
        'total_items': total,
        'judge_categories': judge_categories,
        'confidence_stats': {
            'average': avg_confidence,
            'median': median_confidence,
            'total_scored': len(confidence_scores)
        },
        'data_source_stats': data_source_stats,
        'error_type_stats': error_type_stats
    }

def Judge_compile(evaluation_data=None, evaluation_file=None, original_data_file=None, 
                 output_file=None, base_url=None, headers=None, max_workers=64):
    """
    使用LLM-as-Judge判断代码修复质量
    
    Args:
        evaluation_data: evaluate_llm_fixes_merged函数返回的数据，如果提供则直接使用
        evaluation_file: evaluate_llm_fixes_merged生成的json文件路径
        original_data_file: 原始数据文件路径（merged格式）
        output_file: 输出文件路径
        base_url: LLM API地址
        headers: API请求头
        max_workers: 并发数
    """
    
    # 参数验证
    if evaluation_data is None and evaluation_file is None:
        raise ValueError("必须提供evaluation_data或evaluation_file参数")
    
    if base_url is None or headers is None:
        raise ValueError("必须提供base_url和headers参数")
    
    # 加载评估数据
    if evaluation_data is not None:
        print("使用提供的评估数据")
        data = evaluation_data
    else:
        print(f"从文件加载评估数据: {evaluation_file}")
        with open(evaluation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    evaluation_results = data.get('evaluation_results', [])
    print(f"找到 {len(evaluation_results)} 个评估结果")
    
    # 加载原始数据以获取错误详细信息
    original_data_map = {}
    if original_data_file:
        print(f"加载原始数据文件: {original_data_file}")
        with open(original_data_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        
        # 建立uuid到原始数据的映射
        for item in original_data.get('llm_examples_data', []):
            uuid = item.get('uuid', item.get('error_type', 'unknown'))
            item['data_source'] = 'llm_examples'
            original_data_map[uuid] = item
        
        for item in original_data.get('error_base_data', []):
            uuid = item.get('uuid', item.get('error_type', 'unknown'))
            item['data_source'] = 'error_base'
            original_data_map[uuid] = item
    
    # 初始化LLM裁判
    judge = LLMJudge(base_url, headers)
    
    # 存储判断结果
    judge_results = []
    
    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = []
        for result in evaluation_results:
            future = executor.submit(judge_single_item, judge, result, original_data_map)
            futures.append(future)
        
        # 收集结果
        for future in tqdm(as_completed(futures), total=len(futures), desc="LLM裁判评估中"):
            try:
                result = future.result()
                judge_results.append(result)
            except Exception as e:
                print(f"处理LLM裁判任务时出错: {str(e)}")
    
    # 计算统计信息
    judge_statistics = calculate_judge_statistics(judge_results)
    
    # 准备输出数据
    output_data = {
        'judge_results': judge_results,
        'judge_statistics': judge_statistics,
        'original_evaluation_statistics': data.get('statistics', {}),
        'metadata': {
            'evaluation_file': evaluation_file,
            'original_data_file': original_data_file,
            'total_judged': len(judge_results),
            'judge_time': time.strftime("%Y-%m-%d %H:%M:%S"),
            'api_config': {
                'base_url': base_url,
                'max_workers': max_workers
            }
        }
    }
    
    # 保存结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"LLM裁判结果已保存到: {output_file}")
    
    # 打印统计信息
    print(f"\nLLM裁判评估完成！")
    print(f"总共评估: {judge_statistics['total_items']} 个项目")
    print(f"平均置信度: {judge_statistics['confidence_stats']['average']:.1f}")
    print(f"中位数置信度: {judge_statistics['confidence_stats']['median']:.1f}")
    
    print(f"\n判断结果分布:")
    for category, count in judge_statistics['judge_categories'].items():
        percentage = count / judge_statistics['total_items'] * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    if judge_statistics['data_source_stats']:
        print(f"\n按数据源统计:")
        for source, stats in judge_statistics['data_source_stats'].items():
            print(f"  {source}: {stats['total']} 个项目")
            for category, count in stats['judge_categories'].items():
                percentage = count / stats['total'] * 100
                print(f"    {category}: {count} ({percentage:.1f}%)")
    
    print(f"\n按错误类型统计:")
    for error_type, stats in judge_statistics['error_type_stats'].items():
        print(f"  {error_type}: {stats['total']} 个项目")
        # 只显示真正修复的比例
        true_fix_count = stats['judge_categories'].get('真正修复', 0)
        percentage = true_fix_count / stats['total'] * 100
        print(f"    真正修复: {true_fix_count} ({percentage:.1f}%)")
    
    return output_data

def evaluate_llm_fixes(input_file, output_file, base_url, headers, max_workers=64):
    """评估LLM修复编译错误的能力 - 原始格式"""
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
    """
    主函数 - 完整的LLM代码修复双重评估流程
    
    该函数执行完整的两阶段评估：
    1. 第一阶段：编译器评估 - 使用GCC验证LLM修复的代码是否能编译通过
    2. 第二阶段：LLM裁判评估 - 使用LLM-as-Judge判断修复质量是否真正解决问题
    
    完整流程说明：
    ┌─────────────────────────────────────────────────────────────────┐
    │                        数据输入阶段                              │
    │ 1. 加载merged格式的编译错误数据集                                 │
    │    - llm_examples_data: LLM生成的错误示例                        │
    │    - error_base_data: 基础错误数据                              │
    └─────────────────────────────────────────────────────────────────┘
                                    ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                    第一阶段：编译器评估                          │
    │ 1. 使用LLM生成修复代码                                           │
    │ 2. 提取C++代码块                                                │
    │ 3. 使用GCC编译器验证                                             │
    │ 4. 记录编译成功/失败                                             │
    │ 5. 生成编译器评估报告                                            │
    └─────────────────────────────────────────────────────────────────┘
                                    ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                   第二阶段：LLM裁判评估                          │
    │ 1. 基于编译成功的修复代码                                        │
    │ 2. 使用LLM-as-Judge分析修复质量                                 │
    │ 3. 判断是否真正修复（vs简单删除）                               │
    │ 4. 提供置信度评分                                               │
    │ 5. 生成最终质量评估报告                                          │
    └─────────────────────────────────────────────────────────────────┘
                                    ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                        结果输出                                 │
    │ - 编译器评估结果文件                                             │
    │ - LLM裁判评估结果文件                                           │
    │ - 详细统计信息打印                                              │
    └─────────────────────────────────────────────────────────────────┘
    """
    
    print("🚀 开始LLM代码修复双重评估流程")
    print("=" * 80)
    
    # ===================== 配置参数 =====================
    # 数据文件配置
    input_file = '/mnt/tenant-home_speed/zhaijucai/evaluate/merged_gcc_compatible_data.json'  # 输入的错误数据集
    output_file = 'llm_fix_evaluation_results_merged_data_test_32B.json'  # 编译器评估结果
    judge_output_file = 'llm_judge_evaluation_results_32B.json'  # LLM裁判评估结果
    
    # API配置 - 代码修复模型
    fix_model_base_url = 'http://localhost:7801/v1/chat/completions'
    # API配置 - 裁判模型  
    judge_model_base_url = 'http://localhost:7804/v1/chat/completions'
    
    headers = {
        "Authorization": "TEST-46542881-54d4-4096-b93d-6d5a3db326ac",
        "Content-Type": "application/json"
    }
    
    # ===================== 输入验证 =====================
    if not os.path.exists(input_file):
        print(f"❌ 错误: 输入文件 {input_file} 不存在")
        print("请确保已运行filter_data_to_remain_gcc_compile_error.py生成merged数据")
        return
    
    print(f"📂 输入文件: {input_file}")
    print(f"📄 编译评估输出: {output_file}")
    print(f"📄 裁判评估输出: {judge_output_file}")
    print(f"🔧 修复模型API: {fix_model_base_url}")
    print(f"⚖️  裁判模型API: {judge_model_base_url}")
    
    # ===================== 第一阶段：编译器评估 =====================
    print("\n" + "=" * 60)
    print("📝 第一阶段: LLM修复代码 + GCC编译器评估")
    print("=" * 60)
    print("功能：")
    print("  1. 使用LLM生成修复代码")
    print("  2. 提取C++代码块")
    print("  3. 使用GCC编译器验证修复效果")
    print("  4. 统计编译成功率")
    print()
    
    try:
        evaluation_data = evaluate_llm_fixes_merged(
            input_file=input_file,
            output_file=output_file,
            base_url=fix_model_base_url,
            headers=headers,
            max_workers=64  # 并发数
        )
        print("✅ 第一阶段评估完成")
    except Exception as e:
        print(f"❌ 第一阶段评估失败: {str(e)}")
        return
    
    # ===================== 第二阶段：LLM裁判评估 =====================
    print("\n" + "=" * 60)
    print("⚖️  第二阶段: LLM-as-Judge 修复质量评估")
    print("=" * 60)
    print("功能：")
    print("  1. 基于编译成功的代码进行质量判断")
    print("  2. 区分'真正修复' vs '简单删除'")
    print("  3. 提供置信度评分和详细理由")
    print("  4. 按错误类型和数据源统计")
    print()
    
    try:
        print("🔄 使用方式1: 直接使用第一阶段返回的数据")
        judge_data = Judge_compile(
            evaluation_data=evaluation_data,  # 直接使用返回的数据
            original_data_file=input_file,    # 原始数据文件用于获取错误详细信息
            output_file=judge_output_file,
            base_url=judge_model_base_url,
            headers=headers,
            max_workers=64
        )
        print("✅ 第二阶段评估完成")
    except Exception as e:
        print(f"❌ 第二阶段评估失败: {str(e)}")
        return
    
    # ===================== 总结报告 =====================
    print("\n" + "=" * 60)
    print("📊 双重评估流程完成总结")
    print("=" * 60)
    
    if evaluation_data and 'statistics' in evaluation_data:
        overall_stats = evaluation_data['statistics']['overall']
        print(f"🔧 编译器评估结果:")
        print(f"   总样本数: {overall_stats['total_items']}")
        print(f"   编译成功: {overall_stats['successful_fixes']}")
        print(f"   编译成功率: {overall_stats['overall_success_rate']:.2%}")
    
    if judge_data and 'judge_statistics' in judge_data:
        judge_stats = judge_data['judge_statistics']
        print(f"\n⚖️  LLM裁判评估结果:")
        print(f"   评估样本数: {judge_stats['total_items']}")
        print(f"   平均置信度: {judge_stats['confidence_stats']['average']:.1f}")
        if '真正修复' in judge_stats['judge_categories']:
            true_fix_count = judge_stats['judge_categories']['真正修复']
            true_fix_rate = true_fix_count / judge_stats['total_items'] * 100
            print(f"   真正修复数量: {true_fix_count} ({true_fix_rate:.1f}%)")
    
    print(f"\n📁 结果文件:")
    print(f"   编译器评估: {output_file}")
    print(f"   裁判评估: {judge_output_file}")
    print("\n🎉 双重评估流程全部完成！")

def remove_o_files(file_path):
    try:
        # 获取当前终端所在目录
        dir_path = os.getcwd()
        # print(f"删除.o文件所在目录: {dir_path}")
        # 遍历目录中所有文件
        for filename in os.listdir(dir_path):
            if filename.endswith('.o'):
                file_to_remove = os.path.join(dir_path, filename)
                os.unlink(file_to_remove)
        return True
    except Exception as e:
        print(f"删除.o文件时出错: {str(e)}")
        return False 
    
def demo_judge_from_file():
    """
    演示从文件加载数据进行LLM裁判评估
    
    该函数演示如何单独运行LLM-as-Judge评估，适用于以下场景：
    1. 已经有了编译器评估结果文件
    2. 只想重新运行LLM裁判评估部分
    3. 使用不同的裁判模型重新评估
    4. 调试和测试LLM裁判功能
    
    使用场景：
    - 当第一阶段（编译器评估）已经完成，只需要进行质量判断
    - 当需要使用不同参数重新运行裁判评估时
    - 当需要对历史评估结果进行重新分析时
    """
    print("\n" + "🔄 演示: 从文件加载数据进行LLM-as-Judge评估")
    print("=" * 70)
    print("📋 使用场景:")
    print("  1. 已有编译器评估结果，只需质量判断")
    print("  2. 重新使用不同裁判模型评估")
    print("  3. 调试和测试LLM裁判功能")
    print("=" * 70)
    
    # 配置参数
    evaluation_file = '/home/10350334@zte.intra/Desktop/研发提效三阶段/medical_cpt/error_data/llm_fix_evaluation_results_merged_data_1.5b_rl_lm_228.json'  # 之前生成的评估文件
    original_data_file = 'data_final/merged_gcc_compatible_data.json'
    judge_output_file = 'llm_judge_from_file_results.json'
    
    # API配置
    base_url = 'http://10.55.42.83:31032/v1/chat/completions'
    headers = {
        "Authorization": "TEST-46542881-54d4-4096-b93d-6d5a3db326ac",
        "Content-Type": "application/json"
    }
    
    # 检查文件是否存在
    if not os.path.exists(evaluation_file):
        print(f"错误: 评估文件 {evaluation_file} 不存在")
        print("请先运行main()生成评估结果文件")
        return
    
    if not os.path.exists(original_data_file):
        print(f"错误: 原始数据文件 {original_data_file} 不存在")
        return
    
    # 方式2: 从文件加载数据
    print("使用方式2: 从文件加载评估数据")
    Judge_compile(
        evaluation_file=evaluation_file,     # 从文件加载评估数据
        original_data_file=original_data_file,  # 原始数据文件
        output_file=judge_output_file,
        base_url=base_url,
        headers=headers,
        max_workers=64
    )
def data_analyze(judge_result_file, output_file=None, show_details=True):
    '''
    分析Judge_compile函数输出的数据，统计compile_success、judge_result、data_source的分布情况
    
    Args:
        judge_result_file (str): Judge_compile函数生成的结果文件路径
        output_file (str, optional): 统计结果输出文件路径，如果不提供则只打印
        show_details (bool): 是否显示详细统计信息
        
    Returns:
        dict: 包含所有统计信息的字典
    '''
    
    print("📊 开始数据分析和统计")
    print("=" * 60)
    
    # 加载数据
    try:
        with open(judge_result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        judge_results = data.get('judge_results', [])
        print(f"📂 加载数据文件: {judge_result_file}")
        print(f"📈 总样本数量: {len(judge_results)}")
        
    except Exception as e:
        print(f"❌ 加载数据失败: {str(e)}")
        return None
    
    if not judge_results:
        print("⚠️ 数据文件中没有找到judge_results")
        return None
    
    # 初始化统计计数器
    compile_success_stats = {"成功": 0, "失败": 0}
    judge_result_stats = {}
    data_source_stats = {}
    error_type_stats = {}
    confidence_scores = []
    
    # 交叉统计 - 按数据源分组的编译成功情况
    cross_compile_by_source = {}
    # 交叉统计 - 按数据源分组的裁判结果
    cross_judge_by_source = {}
    # 交叉统计 - 按错误类型分组的裁判结果
    cross_judge_by_error_type = {}
    
    print("\n🔍 数据分析中...")
    
    # 遍历所有结果进行统计
    for result in judge_results:
        compile_success = result.get('compile_success', False)
        judge_result = result.get('judge_result', '未知').strip('*').strip()  # 去除可能的**标记
        data_source = result.get('data_source', '未知')
        error_type = result.get('error_type', '未知')
        confidence = result.get('confidence', 0)
        
        # 编译成功统计
        if compile_success:
            compile_success_stats["成功"] += 1
        else:
            compile_success_stats["失败"] += 1
        
        # 裁判结果统计
        if judge_result not in judge_result_stats:
            judge_result_stats[judge_result] = 0
        judge_result_stats[judge_result] += 1
        
        # 数据源统计
        if data_source not in data_source_stats:
            data_source_stats[data_source] = 0
        data_source_stats[data_source] += 1
        
        # 错误类型统计
        if error_type not in error_type_stats:
            error_type_stats[error_type] = 0
        error_type_stats[error_type] += 1
        
        # 置信度统计
        if confidence > 0:
            confidence_scores.append(confidence)
        
        # 交叉统计 - 按数据源分组的编译成功情况
        if data_source not in cross_compile_by_source:
            cross_compile_by_source[data_source] = {"成功": 0, "失败": 0, "总数": 0}
        
        cross_compile_by_source[data_source]["总数"] += 1
        if compile_success:
            cross_compile_by_source[data_source]["成功"] += 1
        else:
            cross_compile_by_source[data_source]["失败"] += 1
        
        # 交叉统计 - 按数据源分组的裁判结果
        if data_source not in cross_judge_by_source:
            cross_judge_by_source[data_source] = {}
        if judge_result not in cross_judge_by_source[data_source]:
            cross_judge_by_source[data_source][judge_result] = 0
        cross_judge_by_source[data_source][judge_result] += 1
        
        # 交叉统计 - 按错误类型分组的裁判结果
        if error_type not in cross_judge_by_error_type:
            cross_judge_by_error_type[error_type] = {}
        if judge_result not in cross_judge_by_error_type[error_type]:
            cross_judge_by_error_type[error_type][judge_result] = 0
        cross_judge_by_error_type[error_type][judge_result] += 1
    
    # 计算置信度统计
    confidence_stats = {}
    if confidence_scores:
        confidence_stats = {
            "平均值": statistics.mean(confidence_scores),
            "中位数": statistics.median(confidence_scores),
            "最大值": max(confidence_scores),
            "最小值": min(confidence_scores),
            "样本数": len(confidence_scores)
        }
    
    # 汇总统计结果
    analysis_results = {
        "基本统计": {
            "总样本数": len(judge_results),
            "编译成功统计": compile_success_stats,
            "裁判结果统计": judge_result_stats,
            "数据源统计": data_source_stats,
            "错误类型统计": error_type_stats,
            "置信度统计": confidence_stats
        },
        "交叉统计": {
            "按数据源的编译成功情况": cross_compile_by_source,
            "按数据源的裁判结果": cross_judge_by_source,
            "按错误类型的裁判结果": cross_judge_by_error_type
        },
        "分析元数据": {
            "分析时间": time.strftime("%Y-%m-%d %H:%M:%S"),
            "源文件": judge_result_file
        }
    }
    
    # 打印统计结果
    total_samples = len(judge_results)
    
    print("\n" + "=" * 60)
    print("📊 基本统计结果")
    print("=" * 60)
    
    # 编译成功统计
    print(f"\n🔧 编译成功情况:")
    for status, count in compile_success_stats.items():
        percentage = count / total_samples * 100
        print(f"   {status}: {count} ({percentage:.1f}%)")
    
    # 裁判结果统计
    print(f"\n⚖️ LLM裁判结果分布:")
    for result, count in sorted(judge_result_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_samples * 100
        print(f"   {result}: {count} ({percentage:.1f}%)")
    
    # 数据源统计
    print(f"\n📂 数据源分布:")
    for source, count in sorted(data_source_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_samples * 100
        print(f"   {source}: {count} ({percentage:.1f}%)")
    
    # 置信度统计
    if confidence_stats:
        print(f"\n🎯 置信度统计:")
        print(f"   平均值: {confidence_stats['平均值']:.1f}")
        print(f"   中位数: {confidence_stats['中位数']:.1f}")
        print(f"   范围: {confidence_stats['最小值']:.0f} - {confidence_stats['最大值']:.0f}")
        print(f"   有效样本: {confidence_stats['样本数']}")
    
    if show_details:
        print("\n" + "=" * 60)
        print("📈 交叉统计分析")
        print("=" * 60)
        
        # 按数据源的编译成功情况
        print(f"\n🔧 按数据源分组的编译成功情况:")
        for source, stats in cross_compile_by_source.items():
            success_rate = stats["成功"] / stats["总数"] * 100 if stats["总数"] > 0 else 0
            print(f"   {source}:")
            print(f"     总数: {stats['总数']}")
            print(f"     成功: {stats['成功']} ({success_rate:.1f}%)")
            print(f"     失败: {stats['失败']} ({100-success_rate:.1f}%)")
        
        # 按数据源的裁判结果
        print(f"\n⚖️ 按数据源分组的裁判结果:")
        for source, results in cross_judge_by_source.items():
            total_for_source = sum(results.values())
            print(f"   {source} (总数: {total_for_source}):")
            for result, count in sorted(results.items(), key=lambda x: x[1], reverse=True):
                percentage = count / total_for_source * 100
                print(f"     {result}: {count} ({percentage:.1f}%)")
        
        # 重点错误类型分析（显示前10个）
        print(f"\n🐛 主要错误类型的裁判结果 (Top 10):")
        sorted_error_types = sorted(error_type_stats.items(), key=lambda x: x[1], reverse=True)[:10]
        for error_type, total_count in sorted_error_types:
            if error_type in cross_judge_by_error_type:
                results = cross_judge_by_error_type[error_type]
                true_fix_count = results.get('真正修复', 0)
                true_fix_rate = true_fix_count / total_count * 100
                print(f"   {error_type} (总数: {total_count}):")
                print(f"     真正修复: {true_fix_count} ({true_fix_rate:.1f}%)")
                other_results = {k: v for k, v in results.items() if k != '真正修复'}
                if other_results:
                    for result, count in sorted(other_results.items(), key=lambda x: x[1], reverse=True):
                        percentage = count / total_count * 100
                        print(f"     {result}: {count} ({percentage:.1f}%)")
    
    # 保存结果
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=2)
            print(f"\n💾 统计结果已保存到: {output_file}")
        except Exception as e:
            print(f"\n❌ 保存统计结果失败: {str(e)}")
    
    print(f"\n✅ 数据分析完成！")
    
    return analysis_results

def demo_data_analyze():
    """
    演示如何使用data_analyze函数进行数据分析
    """
    print("📊 演示数据分析功能")
    print("=" * 50)
    
    # 配置文件路径
    judge_result_file = 'llm_judge_evaluation_results_32B.json'  # Judge_compile的输出文件
    analysis_output_file = 'data_analysis_results.json'    # 分析结果保存文件
    
    # 检查输入文件是否存在
    if not os.path.exists(judge_result_file):
        print(f"❌ 输入文件不存在: {judge_result_file}")
        print("请先运行Judge_compile函数生成结果文件")
        return
    
    # 执行数据分析
    print(f"🔍 分析文件: {judge_result_file}")
    
    try:
        # 调用数据分析函数
        results = data_analyze(
            judge_result_file=judge_result_file,
            output_file=analysis_output_file,
            show_details=True  # 显示详细统计
        )
        
        if results:
            print("\n🎉 数据分析完成！")
            print(f"📄 详细结果已保存到: {analysis_output_file}")
            
            # 显示关键指标摘要
            basic_stats = results.get('基本统计', {})
            print(f"\n📋 关键指标摘要:")
            print(f"   总样本数: {basic_stats.get('总样本数', 0)}")
            
            compile_stats = basic_stats.get('编译成功统计', {})
            if compile_stats:
                total = sum(compile_stats.values())
                success_rate = compile_stats.get('成功', 0) / total * 100 if total > 0 else 0
                print(f"   编译成功率: {success_rate:.1f}%")
            
            judge_stats = basic_stats.get('裁判结果统计', {})
            if judge_stats:
                true_fix_count = judge_stats.get('真正修复', 0)
                total_judge = sum(judge_stats.values())
                true_fix_rate = true_fix_count / total_judge * 100 if total_judge > 0 else 0
                print(f"   真正修复率: {true_fix_rate:.1f}%")
                
        else:
            print("❌ 数据分析失败")
            
    except Exception as e:
        print(f"❌ 分析过程出错: {str(e)}")

if __name__ == "__main__":
    # 可以选择运行不同的函数
    
    # 选项1: 运行完整的双重评估流程
    main()
    
    # 选项2: 运行LLM裁判评估（从文件加载）
    # demo_judge_from_file()
    
    # 选项3: 运行数据分析
    demo_data_analyze()
    
    # 如果要运行原来的demo_judge_from_file，取消下面的注释
    # demo_judge_from_file()

"""
===============================================================================
                        LLM代码修复双重评估系统使用说明
===============================================================================

🎯 系统目标：
   评估大语言模型（LLM）修复C++编译错误的能力，不仅验证代码能否编译通过，
   更重要的是判断修复是否真正解决问题，而非简单删除出错代码。

📋 系统架构：

   ┌─────────────────────────────────────────────────────────────────┐
   │                     第一阶段：编译器评估                          │
   │ evaluate_llm_fixes_merged()                                   │
   │ • 使用LLM生成修复代码                                           │
   │ • 通过GCC编译器验证修复效果                                      │
   │ • 统计编译成功率                                               │
   └─────────────────────────────────────────────────────────────────┘
                                   ↓
   ┌─────────────────────────────────────────────────────────────────┐
   │                    第二阶段：LLM裁判评估                         │
   │ Judge_compile()                                               │
   │ • 使用LLM-as-Judge分析修复质量                                 │
   │ • 区分真正修复vs简单删除                                        │
   │ • 提供置信度评分和详细理由                                      │
   └─────────────────────────────────────────────────────────────────┘
                                   ↓
   ┌─────────────────────────────────────────────────────────────────┐
   │                    第三阶段：数据分析统计                        │
   │ data_analyze()                                                │
   │ • 全面统计编译成功率和修复质量                                   │
   │ • 按数据源、错误类型、置信度交叉分析                             │
   │ • 生成详细的可视化报告                                          │
   └─────────────────────────────────────────────────────────────────┘

🚀 主要函数说明：

1. evaluate_llm_fixes_merged()
   功能：第一阶段编译器评估
   输入：merged格式的错误数据集
   输出：编译评估结果（成功/失败 + 详细信息）
   
   参数说明：
   - input_file: 包含错误代码的数据文件
   - output_file: 编译评估结果输出文件
   - base_url: 修复模型API地址
   - headers: API认证信息
   - max_workers: 并发线程数

2. Judge_compile()
   功能：第二阶段LLM裁判评估
   输入：编译评估结果 + 原始错误数据
   输出：修复质量判断结果
   
   两种使用方式：
   方式A - 直接使用函数返回数据：
   ```python
   evaluation_data = evaluate_llm_fixes_merged(...)
   Judge_compile(
       evaluation_data=evaluation_data,
       original_data_file="原始数据.json",
       output_file="裁判结果.json",
       base_url="裁判模型API",
       headers=headers
   )
   ```
   
   方式B - 从文件加载数据：
   ```python
   Judge_compile(
       evaluation_file="编译评估结果.json",
       original_data_file="原始数据.json",
       output_file="裁判结果.json",
       base_url="裁判模型API",
       headers=headers
   )
   ```

3. data_analyze()
   功能：第三阶段数据分析和统计
   输入：LLM裁判评估结果文件
   输出：详细的统计分析报告
   
   参数说明：
   - judge_result_file: Judge_compile生成的结果文件路径
   - output_file: 统计结果输出文件路径（可选）
   - show_details: 是否显示详细统计信息
   
   使用方式：
   ```python
   data_analyze(
       judge_result_file="llm_judge_results.json",
       output_file="analysis_results.json",
       show_details=True
   )
   ```
   
   统计内容：
   - 编译成功率分析
   - LLM裁判结果分布
   - 数据源对比分析
   - 错误类型成功率排名
   - 置信度统计分析
   - 交叉统计分析

🏷️ 判断结果分类：
   
   ✅ 真正修复：正确修复编译错误，保持原有功能逻辑
      例：添加缺失头文件、修正语法错误、修复类型匹配等
   
   ❌ 简单删除：通过删除出错代码避免编译错误，但丢失原有功能
      例：删除整个函数、注释掉出错行、移除变量声明等
   
   ⚠️ 过度修改：大幅改变原有逻辑或添加不必要的代码
      例：完全重写函数逻辑、添加无关功能等
   
   🚫 无效修复：修复不正确或可能引入新的编译问题
      例：语法仍有错误、逻辑矛盾等

📊 输出文件格式：

1. 编译评估结果文件：
   ```json
   {
     "evaluation_results": [
       {
         "uuid": "唯一标识",
         "error_type": "错误类型",
         "success": true/false,
         "original_code": "原始错误代码",
         "fixed_code": "修复后代码",
         "compilation_result": "编译详情"
       }
     ],
     "statistics": {...}
   }
   ```

2. LLM裁判评估结果文件：
   ```json
   {
     "judge_results": [
       {
         "uuid": "唯一标识",
         "judge_result": "真正修复/简单删除/过度修改/无效修复",
         "confidence": 85,
         "reason": "详细判断理由",
         "raw_judge_response": "LLM原始响应"
       }
     ],
     "judge_statistics": {...}
   }
   ```

3. 数据分析结果文件：
   ```json
   {
     "基本统计": {
       "总样本数": 1000,
       "编译成功统计": {"成功": 800, "失败": 200},
       "裁判结果统计": {"真正修复": 600, "简单删除": 150, ...},
       "数据源统计": {"llm_examples": 500, "error_base": 500},
       "错误类型统计": {"C2065": 100, "C2009": 80, ...},
       "置信度统计": {"平均值": 75.5, "中位数": 80, ...}
     },
     "交叉统计": {
       "按数据源的编译成功情况": {...},
       "按数据源的裁判结果": {...},
       "按错误类型的裁判结果": {...}
     }
   }
   ```

🔧 运行示例：

1. 完整流程（推荐）：
   ```python
   python evaluate_llm_merged_with_LLM_as_Judge.py
   # 或者在代码中调用 main()
   ```
   
2. 仅运行LLM裁判评估：
   ```python
   # 修改__main__部分为：
   demo_judge_from_file()
   ```

3. 仅运行数据分析：
   ```python
   # 修改__main__部分为：
   demo_data_analyze()
   ```
   
4. 三阶段完整流程：
   ```python
   # 第一阶段：编译器评估
   evaluation_data = evaluate_llm_fixes_merged(...)
   
   # 第二阶段：LLM裁判评估
   Judge_compile(evaluation_data=evaluation_data, ...)
   
   # 第三阶段：数据分析
   data_analyze(judge_result_file="judge_results.json")
   ```📈 统计信息：
   系统会自动生成详细统计，包括：
   - 总体成功率和各错误类型成功率
   - LLM裁判结果分布
   - 按数据源（llm_examples vs error_base）分组统计
   - 置信度分析（平均值、中位数）
   - 交叉统计分析（数据源vs编译成功率、错误类型vs修复质量等）
   - Top错误类型的修复质量排名
   - 可视化的百分比分布图表

⚙️ 配置说明：
   - 修复模型API：用于生成修复代码的LLM服务
   - 裁判模型API：用于评估修复质量的LLM服务（可以是同一个）
   - 并发数：根据服务器性能调整，建议8-24
   - 输入数据：需要是merged格式，包含llm_examples_data和error_base_data

💡 使用建议：
   1. 确保输入数据格式正确
   2. 根据API服务器性能调整并发数
   3. 可以使用不同的模型进行修复和裁判
   4. 注意API调用限制和延迟设置
   5. 定期检查输出文件确保评估正常进行

===============================================================================
"""

