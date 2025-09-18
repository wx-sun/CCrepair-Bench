#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
编译错误相似性分析脚本
专门用于找出代码和编译错误原因相似度超过90%的重复语料，并将相似的错误保存到新文件中
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict, Counter
import difflib
import os
import sys
import re
import hashlib
from math import sqrt

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_tfidf_vectors(texts: List[str]) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """创建TF-IDF向量（离线实现）"""
    # 预处理文本
    processed_texts = []
    for text in texts:
        # 提取代码标识符和关键词
        words = extract_code_features(text)
        processed_texts.append(words)
    
    # 构建词汇表
    vocab = set()
    for words in processed_texts:
        vocab.update(words)
    vocab = list(vocab)
    
    # 计算文档频率
    doc_freq = {}
    total_docs = len(processed_texts)
    for word in vocab:
        doc_freq[word] = sum(1 for words in processed_texts if word in words)
    
    # 计算IDF
    idf = {}
    for word in vocab:
        idf[word] = max(0.1, log_safe(total_docs / doc_freq[word]))
    
    # 计算TF-IDF向量
    tfidf_vectors = []
    for words in processed_texts:
        word_count = Counter(words)
        total_words = len(words)
        
        vector = {}
        for word in word_count:
            tf = word_count[word] / max(1, total_words)
            vector[word] = tf * idf[word]
        
        tfidf_vectors.append(vector)
    
    return tfidf_vectors, idf


def extract_code_features(text: str) -> List[str]:
    """从代码文本中提取特征词"""
    if not text:
        return []
    
    # 提取C++关键词、标识符、操作符等
    features = []
    
    # C++关键词
    cpp_keywords = re.findall(r'\b(?:int|char|float|double|void|if|else|for|while|return|include|define|const|static|class|struct|namespace|using|std|cout|cin|endl)\b', text)
    features.extend(cpp_keywords)
    
    # 标识符（变量名、函数名）
    identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text)
    features.extend(identifiers)
    
    # 操作符和符号
    operators = re.findall(r'[+\-*/=<>!&|^%]+|[{}()\[\];,.]', text)
    features.extend(operators)
    
    # 字符串字面量的特征
    strings = re.findall(r'"[^"]*"', text)
    for s in strings:
        if len(s) > 2:  # 不是空字符串
            features.append('STRING_LITERAL')
    
    # 数字字面量
    numbers = re.findall(r'\b\d+\b', text)
    if numbers:
        features.append('NUMBER_LITERAL')
    
    return features


def log_safe(x):
    """安全的对数计算"""
    import math
    return math.log(max(1e-10, x))


def cosine_similarity_sparse(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """计算稀疏向量的余弦相似度"""
    if not vec1 or not vec2:
        return 0.0
    
    # 计算点积
    dot_product = 0.0
    for word in vec1:
        if word in vec2:
            dot_product += vec1[word] * vec2[word]
    
    # 计算向量模长
    norm1 = sqrt(sum(v * v for v in vec1.values()))
    norm2 = sqrt(sum(v * v for v in vec2.values()))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def calculate_offline_similarity_batch(items: List[Dict]) -> List[List[float]]:
    """使用离线算法批量计算所有item之间的相似度矩阵"""
    if not items:
        return []
    
    # 准备文本数据
    texts = []
    for item in items:
        # 组合代码和错误信息
        code = item.get('error_example_llm_code', '')
        detail = item.get('error_example_llm_detail', '')
        # 简化错误详情（去除文件路径）
        clean_detail = '\n'.join([line.split(':', 2)[-1] if ':' in line else line 
                                 for line in detail.split('\n')])
        combined_text = f"CODE: {normalize_code(code)} ERROR: {clean_detail}"
        texts.append(combined_text)
    
    # 计算TF-IDF向量
    tfidf_vectors, _ = create_tfidf_vectors(texts)
    
    # 计算相似度矩阵
    n = len(tfidf_vectors)
    similarity_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                similarity_matrix[i][j] = 1.0
            else:
                # 组合多种相似度算法
                tfidf_sim = cosine_similarity_sparse(tfidf_vectors[i], tfidf_vectors[j])
                difflib_sim = difflib.SequenceMatcher(None, texts[i], texts[j]).ratio()
                
                # 加权组合：TF-IDF权重更高
                combined_sim = 0.7 * tfidf_sim + 0.3 * difflib_sim
                
                similarity_matrix[i][j] = combined_sim
                similarity_matrix[j][i] = combined_sim
    
    return similarity_matrix


def process_error_group_offline(args):
    """使用离线算法处理单个错误类型组"""
    error_key, group_items, similarity_threshold, group_idx = args
    
    try:
        processed = set()
        similar_groups = []
        comparisons_made = 0
        
        if len(group_items) < 2:
            return similar_groups, comparisons_made, group_idx
        
        # 离线批量计算模式
        items_only = [item for _, item in group_items]
        similarity_matrix = calculate_offline_similarity_batch(items_only)
        
        if similarity_matrix:
            # 根据相似度矩阵找出相似组
            n = len(group_items)
            comparisons_made += n * (n - 1) // 2
            
            for i in range(n):
                orig_idx1, item1 = group_items[i]
                if orig_idx1 in processed:
                    continue
                    
                current_group = [item1]
                processed.add(orig_idx1)
                
                for j in range(i + 1, n):
                    orig_idx2, item2 = group_items[j]
                    if orig_idx2 in processed:
                        continue
                        
                    similarity = similarity_matrix[i][j]
                    
                    if similarity >= similarity_threshold:
                        current_group.append(item2)
                        processed.add(orig_idx2)
                
                # 只保留包含多个项目的组
                if len(current_group) > 1:
                    similar_groups.append(current_group)
        
        return similar_groups, comparisons_made, group_idx
        
    except Exception as e:
        print(f"❌ 离线算法处理错误组 {error_key} 时出错: {e}")
        return [], 0, group_idx


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
    """
    在控制台打印进度条
    @params:
        iteration   - Required  : 当前迭代 (Int)
        total       - Required  : 总迭代数 (Int)
        prefix      - Optional  : 前缀字符串 (Str)
        suffix      - Optional  : 后缀字符串 (Str)
        decimals    - Optional  : 小数点后位数 (Int)
        length      - Optional  : 进度条长度 (Int)
        fill        - Optional  : 填充字符 (Str)
        print_end   - Optional  : 行尾字符 (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # 如果完成了，打印新行
    if iteration == total: 
        print()


def load_json_file(file_path: str) -> List[Dict]:
    """加载JSON文件"""
    try:
        logger.info(f"正在加载文件: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("JSON文件应该包含一个数组")
        
        logger.info(f"成功加载 {len(data)} 条记录")
        return data
    except Exception as e:
        logger.error(f"加载JSON文件失败: {e}")
        raise


def save_json_file(data: List[Dict], file_path: str) -> None:
    """保存JSON文件"""
    try:
        logger.info(f"正在保存文件: {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"文件已保存到: {file_path}")
    except Exception as e:
        logger.error(f"保存JSON文件失败: {e}")
        raise


def normalize_code(code: str) -> str:
    """标准化代码，去除空白字符和注释"""
    if not code:
        return ""
    
    lines = code.split('\n')
    normalized_lines = []
    
    for line in lines:
        # 去除行注释
        if '//' in line:
            line = line.split('//')[0]
        
        # 去除块注释的开始部分
        if '/*' in line:
            line = line.split('/*')[0]
        
        # 去除前后空白
        line = line.strip()
        
        # 去除多余的空格
        line = ' '.join(line.split())
        
        if line:
            normalized_lines.append(line)
    
    return '\n'.join(normalized_lines)


def calculate_code_similarity(code1: str, code2: str) -> float:
    """计算两段代码的相似度"""
    if not code1 or not code2:
        return 0.0
    
    # 标准化代码：去除空白字符和注释
    normalized_code1 = normalize_code(code1)
    normalized_code2 = normalize_code(code2)
    
    if not normalized_code1 or not normalized_code2:
        return 0.0
    
    # 使用difflib计算相似度
    similarity = difflib.SequenceMatcher(None, normalized_code1, normalized_code2).ratio()
    return similarity


def calculate_error_detail_similarity(detail1: str, detail2: str) -> float:
    """计算错误详细信息的相似度"""
    if not detail1 or not detail2:
        return 0.0
    
    # 去除临时文件路径，只比较错误信息本身
    lines1 = [line.split(':', 3)[-1] if ':' in line else line for line in detail1.split('\n')]
    lines2 = [line.split(':', 3)[-1] if ':' in line else line for line in detail2.split('\n')]
    
    detail1_clean = '\n'.join(lines1).strip()
    detail2_clean = '\n'.join(lines2).strip()
    
    if not detail1_clean or not detail2_clean:
        return 0.0
    
    similarity = difflib.SequenceMatcher(None, detail1_clean, detail2_clean).ratio()
    return similarity


def calculate_overall_similarity(item1: Dict, item2: Dict) -> float:
    """计算两个错误条目的整体相似度（专注于代码和编译错误原因）- 离线版本"""
    # 权重设置 - 更专注于代码和错误原因
    code_weight = 0.6        # 代码相似度权重
    detail_weight = 0.3      # 编译错误详情权重
    error_type_weight = 0.1  # 错误类型权重
    
    # 计算代码相似度（使用增强版本）
    code1 = item1.get('error_example_llm_code', '')
    code2 = item2.get('error_example_llm_code', '')
    code_sim = calculate_enhanced_code_similarity(code1, code2)
    
    # 计算错误详情相似度
    detail_sim = calculate_error_detail_similarity(
        item1.get('error_example_llm_detail', ''),
        item2.get('error_example_llm_detail', '')
    )
    
    # 错误类型相似度（相同为1，不同为0）
    error_type_sim = 1.0 if (item1.get('error_type') == item2.get('error_type') and 
                             item1.get('error_type_detail') == item2.get('error_type_detail')) else 0.0
    
    # 加权计算总相似度
    overall_similarity = (code_sim * code_weight + 
                         detail_sim * detail_weight + 
                         error_type_sim * error_type_weight)
    
    return overall_similarity


def calculate_enhanced_code_similarity(code1: str, code2: str) -> float:
    """增强版代码相似度计算（使用TF-IDF + difflib）"""
    if not code1 or not code2:
        return 0.0
    
    # 标准化代码
    norm_code1 = normalize_code(code1)
    norm_code2 = normalize_code(code2)
    
    if not norm_code1 or not norm_code2:
        return 0.0
    
    # 使用TF-IDF计算语义相似度
    texts = [norm_code1, norm_code2]
    tfidf_vectors, _ = create_tfidf_vectors(texts)
    
    if len(tfidf_vectors) >= 2:
        tfidf_sim = cosine_similarity_sparse(tfidf_vectors[0], tfidf_vectors[1])
    else:
        tfidf_sim = 0.0
    
    # 使用difflib计算序列相似度
    difflib_sim = difflib.SequenceMatcher(None, norm_code1, norm_code2).ratio()
    
    # 计算代码结构相似度（基于行数、括号等）
    structure_sim = calculate_code_structure_similarity(norm_code1, norm_code2)
    
    # 加权组合多种相似度
    combined_sim = 0.5 * tfidf_sim + 0.3 * difflib_sim + 0.2 * structure_sim
    
    return combined_sim


def calculate_code_structure_similarity(code1: str, code2: str) -> float:
    """计算代码结构相似度"""
    if not code1 or not code2:
        return 0.0
    
    # 提取结构特征
    features1 = extract_structure_features(code1)
    features2 = extract_structure_features(code2)
    
    # 计算特征向量的余弦相似度
    all_features = set(features1.keys()) | set(features2.keys())
    if not all_features:
        return 0.0
    
    vec1 = [features1.get(f, 0) for f in all_features]
    vec2 = [features2.get(f, 0) for f in all_features]
    
    # 计算余弦相似度
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sqrt(sum(a * a for a in vec1))
    norm2 = sqrt(sum(b * b for b in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def extract_structure_features(code: str) -> Dict[str, int]:
    """提取代码结构特征"""
    features = {}
    
    # 行数
    features['line_count'] = len(code.split('\n'))
    
    # 各种括号数量
    features['curly_braces'] = code.count('{') + code.count('}')
    features['parentheses'] = code.count('(') + code.count(')')
    features['square_brackets'] = code.count('[') + code.count(']')
    
    # 分号数量（语句数）
    features['semicolons'] = code.count(';')
    
    # 关键词数量
    keywords = ['if', 'else', 'for', 'while', 'return', 'int', 'char', 'float', 'double']
    for keyword in keywords:
        features[f'keyword_{keyword}'] = len(re.findall(r'\b' + keyword + r'\b', code))
    
    # 操作符数量
    features['assignments'] = code.count('=')
    features['comparisons'] = code.count('==') + code.count('!=') + code.count('<=') + code.count('>=')
    
    return features


def find_similar_groups(data: List[Dict], similarity_threshold: float = 0.90, use_gpu: bool = False, gpu_ids: List[int] = None) -> List[List[Dict]]:
    """找出代码和编译错误原因相似度超过阈值的错误组（按error_type分组优化）"""
    logger.info(f"开始查找代码和编译错误原因相似度超过 {similarity_threshold*100:.0f}% 的语料...")
    
    # 按error_type分组以优化比较效率
    print(f"\n🔧 优化策略: 按错误类型分组比较，大幅提升效率...")
    error_type_groups = defaultdict(list)
    
    for i, item in enumerate(data):
        error_key = f"{item.get('error_type', '')}_{item.get('error_type_detail', '')}"
        error_type_groups[error_key].append((i, item))
    
    print(f"📊 分组结果: {len(data):,} 条语料分为 {len(error_type_groups)} 个错误类型组")
    
    # 计算优化后的比较次数
    estimated_comparisons = 0
    for group_items in error_type_groups.values():
        n = len(group_items)
        if n > 1:
            estimated_comparisons += n * (n - 1) // 2
    
    print(f"⚡ 优化效果: 比较次数从 {len(data)*(len(data)-1)//2:,} 减少到 {estimated_comparisons:,}")
    print(f"🚀 效率提升: {(len(data)*(len(data)-1)//2) / max(1, estimated_comparisons):.1f}x 倍")
    
    processed = set()
    similar_groups = []
    comparisons_made = 0
    
    import time
    start_time = time.time()
    last_update_time = start_time
    processed_groups = 0
    
    # 检查GPU可用性（如果需要）
    if use_gpu:
        if gpu_ids is None:
            gpu_ids = list(range(8))  # 默认使用0-7卡
        
        # 验证GPU可用性
        import torch
        if not torch.cuda.is_available():
            print("⚠️  CUDA不可用，回退到CPU模式")
            use_gpu = False
        else:
            available_gpus = list(range(torch.cuda.device_count()))
            valid_gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id in available_gpus]
            if not valid_gpu_ids:
                print(f"⚠️  指定的GPU {gpu_ids} 都不可用，回退到CPU模式")
                use_gpu = False
            else:
                gpu_ids = valid_gpu_ids
                print(f"🔍 验证GPU可用性: {gpu_ids}")
    
    print(f"\n🔍 开始按错误类型组进行相似度分析...")
    if use_gpu:
        print(f"🚀 多GPU并行模式: 使用 {len(gpu_ids)} 张GPU卡 {gpu_ids}")
    else:
        print(f"💻 CPU传统模式")
    
    # 按错误类型组进行比较
    if use_gpu and gpu_ids:
        # 多GPU并行处理模式
        import threading
        from queue import Queue
        
        # 准备任务队列
        task_queue = Queue()
        result_queue = Queue()
        
        # 只处理有多个元素的组
        valid_groups = [(k, v) for k, v in error_type_groups.items() if len(v) >= 2]
        print(f"📋 准备处理 {len(valid_groups)} 个有效错误类型组")
        
        # 为每个错误组创建任务
        for group_idx, (error_key, group_items) in enumerate(valid_groups):
            task_queue.put((error_key, group_items, similarity_threshold, group_idx))
        
        def gpu_worker(device_id, worker_id):
            """GPU工作线程"""
            while True:
                try:
                    task = task_queue.get(timeout=1)
                    if task is None:
                        break
                    
                    error_key, group_items, threshold, group_idx = task
                    
                    # 处理这个错误组
                    group_similar, group_comparisons, _ = process_error_group_on_gpu(
                        (error_key, group_items, threshold, device_id, group_idx)
                    )
                    
                    result_queue.put((group_similar, group_comparisons, error_key, group_idx))
                    task_queue.task_done()
                    
                except:
                    break
        
        # 启动多个GPU工作线程
        threads = []
        for i, device_id in enumerate(gpu_ids):
            thread = threading.Thread(target=gpu_worker, args=(device_id, i))
            thread.daemon = True
            thread.start()
            threads.append(thread)
            print(f"🔥 启动GPU工作线程 {i} (设备: cuda:{device_id})")
        
        # 等待所有任务完成并收集结果
        processed_groups = 0
        total_comparisons = 0
        
        while processed_groups < len(valid_groups):
            try:
                group_similar, group_comparisons, error_key, group_idx = result_queue.get(timeout=5)
                
                # 合并结果
                similar_groups.extend(group_similar)
                total_comparisons += group_comparisons
                processed_groups += 1
                
                if group_similar:
                    print(f"✅ GPU处理组 {error_key}: 发现 {len(group_similar)} 个相似组")
                
                # 更新进度条
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                if processed_groups > 0:
                    avg_time_per_group = elapsed_time / processed_groups
                    remaining_groups = len(valid_groups) - processed_groups
                    estimated_remaining_time = remaining_groups * avg_time_per_group
                    
                    if estimated_remaining_time > 60:
                        time_str = f"{estimated_remaining_time/60:.1f}分钟"
                    else:
                        time_str = f"{estimated_remaining_time:.0f}秒"
                else:
                    time_str = "计算中..."
                
                progress_info = f"已完成 {processed_groups}/{len(valid_groups)} | 相似组 {len(similar_groups)} | 剩余 {time_str}"
                print_progress_bar(processed_groups, len(valid_groups), 
                                 prefix='🚀 多GPU并行进度:', suffix=progress_info, length=60)
                
            except:
                break
        
        # 停止所有线程
        for _ in threads:
            task_queue.put(None)
        
        for thread in threads:
            thread.join()
        
        comparisons_made = total_comparisons
        processed_groups = len(valid_groups)
        
        print(f"\n🎯 多GPU并行处理完成!")
        print(f"   使用GPU: {gpu_ids}")
        print(f"   处理组数: {len(valid_groups)}")
        print(f"   发现相似组: {len(similar_groups)}")
        
    else:
        # CPU传统计算模式
        for error_key, group_items in error_type_groups.items():
            if len(group_items) < 2:
                continue  # 跳过只有一个元素的组
                
            print(f"\n📝 处理错误类型: {error_key} ({len(group_items)} 条语料)")
            processed_groups += 1
            
            # CPU传统计算模式
            for i in range(len(group_items)):
                orig_idx1, item1 = group_items[i]
                if orig_idx1 in processed:
                    continue
                    
                current_group = [item1]
                processed.add(orig_idx1)
                
                # 与同组内后续项目比较
                for j in range(i + 1, len(group_items)):
                    orig_idx2, item2 = group_items[j]
                    if orig_idx2 in processed:
                        continue
                        
                    similarity = calculate_overall_similarity(item1, item2)
                    comparisons_made += 1
                    
                    if similarity >= similarity_threshold:
                        current_group.append(item2)
                        processed.add(orig_idx2)
                        logger.debug(f"发现相似语料: UUID {item1.get('uuid')} 与 {item2.get('uuid')}, 相似度: {similarity:.3f}")
                
                # 只保留包含多个项目的组（即真正的重复组）
                if len(current_group) > 1:
                    similar_groups.append(current_group)
                    logger.info(f"发现相似组 {len(similar_groups)}: 包含 {len(current_group)} 个相似语料 (错误类型: {error_key})")
            
            # 更新进度条
            current_time = time.time()
            if (current_time - last_update_time) >= 1.0:
                # 计算预计剩余时间
                elapsed_time = current_time - start_time
                if processed_groups > 0:
                    avg_time_per_group = elapsed_time / processed_groups
                    remaining_groups = len(error_type_groups) - processed_groups
                    estimated_remaining_time = remaining_groups * avg_time_per_group
                    
                    # 格式化时间显示
                    if estimated_remaining_time > 3600:
                        time_str = f"{estimated_remaining_time/3600:.1f}小时"
                    elif estimated_remaining_time > 60:
                        time_str = f"{estimated_remaining_time/60:.1f}分钟"
                    else:
                        time_str = f"{estimated_remaining_time:.0f}秒"
                else:
                    time_str = "计算中..."
                
                # 更新进度条
                progress_info = f"类型组 {processed_groups}/{len([k for k, v in error_type_groups.items() if len(v) >= 2])} | 相似组 {len(similar_groups)} | 剩余 {time_str}"
                print_progress_bar(processed_groups, len([k for k, v in error_type_groups.items() if len(v) >= 2]), 
                                 prefix='💻 CPU优化分析进度:', suffix=progress_info, length=60)
                
                last_update_time = current_time
    
    # 最终进度条更新
    elapsed_time = time.time() - start_time
    print_progress_bar(len(error_type_groups), len(error_type_groups), 
                     prefix=f'{"多GPU并行" if use_gpu else "CPU优化"}分析进度:', suffix=f"完成！耗时 {elapsed_time:.1f}秒", length=60)
    
    print(f"\n✅ {'多GPU并行' if use_gpu else 'CPU优化'}分析完成！")
    print(f"📊 统计信息:")
    print(f"   - 总语料数: {len(data):,}")
    print(f"   - 错误类型组数: {len(error_type_groups)}")
    if use_gpu:
        print(f"   - 计算模式: 多GPU并行处理")
        print(f"   - 使用GPU: {gpu_ids}")
    else:
        print(f"   - 计算模式: CPU逐对计算")
        print(f"   - 实际比较次数: {comparisons_made:,}")
        print(f"   - 平均比较速度: {comparisons_made/elapsed_time:.0f} 次/秒")
        print(f"   - 效率提升: {(len(data)*(len(data)-1)//2) / max(1, comparisons_made):.1f}x 倍")
    print(f"   - 发现相似组: {len(similar_groups)}")
    print(f"   - 总耗时: {elapsed_time:.1f}秒")
    
    return similar_groups


def analyze_similarity_distribution(data: List[Dict], sample_size: int = 1000) -> Dict:
    """分析代码和编译错误原因相似度分布情况（采样分析以提高效率）"""
    logger.info(f"正在分析代码和编译错误原因相似度分布（采样 {sample_size} 对）...")
    
    import random
    
    # 如果数据量大，进行采样
    if len(data) > sample_size:
        sample_indices = random.sample(range(len(data)), min(sample_size, len(data)))
        sample_data = [data[i] for i in sample_indices]
    else:
        sample_data = data
    
    similarities = []
    high_similarity_pairs = []
    total_comparisons = 0
    
    print(f"\n正在采样分析 {len(sample_data)} 条语料的相似度分布...")
    
    import time
    start_time = time.time()
    
    for i in range(len(sample_data)):
        comparison_count = min(10, len(sample_data) - i - 1)  # 限制比较范围以提高效率
        for j in range(i + 1, i + 1 + comparison_count):
            if j >= len(sample_data):
                break
            sim = calculate_overall_similarity(sample_data[i], sample_data[j])
            similarities.append(sim)
            total_comparisons += 1
            
            if sim > 0.8:  # 记录高相似度对
                high_similarity_pairs.append({
                    'uuid1': sample_data[i].get('uuid'),
                    'uuid2': sample_data[j].get('uuid'),
                    'similarity': sim
                })
        
        # 更新进度条
        high_sim_count = len([s for s in similarities if s > 0.9])
        progress_info = f"已比较 {total_comparisons} 对 | 高相似度: {high_sim_count}"
        print_progress_bar(i + 1, len(sample_data), prefix='采样分析进度:', suffix=progress_info, length=60)
    
    # 最终统计
    elapsed_time = time.time() - start_time
    print(f"\n✅ 采样分析完成！耗时 {elapsed_time:.1f}秒")
    
    if not similarities:
        return {'error': 'No similarities calculated'}
    
    # 计算统计信息
    analysis = {
        'total_comparisons': len(similarities),
        'avg_similarity': sum(similarities) / len(similarities),
        'max_similarity': max(similarities),
        'min_similarity': min(similarities),
        'high_similarity_count': len([s for s in similarities if s > 0.9]),
        'very_high_similarity_count': len([s for s in similarities if s > 0.95]),
        'exact_duplicates': len([s for s in similarities if s > 0.99]),
        'high_similarity_examples': high_similarity_pairs[:10]  # 前10个高相似度示例
    }
    
    return analysis


def save_similar_groups(similar_groups: List[List[Dict]], output_dir: str, base_name: str) -> None:
    """将相似的语料组保存到不同的文件中"""
    logger.info(f"正在保存 {len(similar_groups)} 个相似语料组到 {output_dir}")
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 创建汇总信息
    summary = {
        'total_groups': len(similar_groups),
        'total_duplicates': sum(len(group) for group in similar_groups),
        'groups_info': []
    }
    
    for i, group in enumerate(similar_groups):
        group_file = f"{base_name}_similar_group_{i+1}.json"
        group_path = os.path.join(output_dir, group_file)
        
        # 保存相似组
        save_json_file(group, group_path)
        
        # 添加到汇总信息
        group_info = {
            'group_id': i + 1,
            'file': group_file,
            'count': len(group),
            'error_type': group[0].get('error_type', 'unknown'),
            'error_type_detail': group[0].get('error_type_detail', 'unknown'),
            'uuids': [item.get('uuid') for item in group]
        }
        summary['groups_info'].append(group_info)
        
        logger.info(f"保存相似组 {i+1}: {len(group)} 个相似语料 -> {group_file}")
    
    # 保存汇总文件
    summary_path = os.path.join(output_dir, f"{base_name}_similarity_summary.json")
    save_json_file(summary, summary_path)
    
    logger.info(f"汇总信息已保存到: {summary_path}")


def process_multiple_files(input_files: List[str], output_dir: str, similarity_threshold: float = 0.90, use_gpu: bool = False, gpu_ids: List[int] = None) -> None:
    """处理多个输入文件，查找代码和编译错误原因相似的语料"""
    all_data = []
    file_sources = {}  # 记录每个条目来自哪个文件
    
    # 加载所有文件
    for file_path in input_files:
        data = load_json_file(file_path)
        file_name = os.path.basename(file_path)
        
        for item in data:
            all_data.append(item)
            file_sources[len(all_data) - 1] = file_name
    
    print(f"\n📁 加载完成: 总共 {len(all_data):,} 条语料，来自 {len(input_files)} 个文件")
    
    # 分析相似度分布
    print(f"\n🔍 第一步: 采样分析相似度分布...")
    similarity_analysis = analyze_similarity_distribution(all_data)
    logger.info(f"相似度分析结果: 平均相似度 {similarity_analysis.get('avg_similarity', 0):.3f}, "
               f"高相似度(>90%)语料数: {similarity_analysis.get('high_similarity_count', 0)}, "
               f"极高相似度(>95%)语料数: {similarity_analysis.get('very_high_similarity_count', 0)}")
    
    # 查找相似组
    print(f"\n🔍 第二步: {'多GPU并行' if use_gpu else 'CPU优化'}分析查找相似语料组...")
    similar_groups = find_similar_groups(all_data, similarity_threshold, use_gpu, gpu_ids)
    
    if similar_groups:
        print(f"\n📊 分析结果: 发现 {len(similar_groups)} 个相似语料组，总共包含 {sum(len(group) for group in similar_groups)} 个重复语料")
        
        # 保存相似组
        base_name = "compile_errors"
        print(f"\n💾 第三步: 保存相似语料组到文件...")
        save_similar_groups(similar_groups, output_dir, base_name)
        
        # 创建统计报告
        print(f"\n📝 第四步: 生成详细分析报告...")
        create_analysis_report(all_data, similar_groups, similarity_analysis, output_dir, base_name)
        
        print(f"\n🎉 所有文件已保存到: {output_dir}")
        
    else:
        print(f"\n❌ 未发现代码和编译错误原因相似度超过 {similarity_threshold*100:.0f}% 的语料组")


def create_analysis_report(all_data: List[Dict], similar_groups: List[List[Dict]], 
                          similarity_analysis: Dict, output_dir: str, base_name: str) -> None:
    """创建详细的分析报告"""
    report = {
        'analysis_summary': {
            'total_records': len(all_data),
            'similar_groups_found': len(similar_groups),
            'total_duplicates': sum(len(group) for group in similar_groups),
            'duplicate_ratio': sum(len(group) for group in similar_groups) / len(all_data) if all_data else 0,
            'similarity_threshold_used': 0.90
        },
        'similarity_distribution': similarity_analysis,
        'error_type_analysis': {},
        'duplicate_groups_summary': []
    }
    
    # 分析错误类型分布
    error_types = {}
    for item in all_data:
        error_type = item.get('error_type', 'unknown')
        error_types[error_type] = error_types.get(error_type, 0) + 1
    
    report['error_type_analysis'] = {
        'distribution': error_types,
        'most_common': sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:10]
    }
    
    # 分析每个相似组
    for i, group in enumerate(similar_groups):
        group_summary = {
            'group_id': i + 1,
            'size': len(group),
            'error_type': group[0].get('error_type'),
            'error_type_detail': group[0].get('error_type_detail'),
            'uuids': [item.get('uuid') for item in group],
            'avg_diversity_score': sum(item.get('diversity_score', 0) for item in group) / len(group),
            'avg_retention_ratio': sum(item.get('retention_ratio', 0) for item in group) / len(group)
        }
        report['duplicate_groups_summary'].append(group_summary)
    
    # 保存报告
    report_path = os.path.join(output_dir, f"{base_name}_analysis_report.json")
    save_json_file(report, report_path)
    
    logger.info(f"详细分析报告已保存到: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='编译错误语料相似性分析工具 - 找出代码和编译错误原因90%以上相似的重复语料')
    parser.add_argument('input_files', nargs='+', help='输入JSON文件路径（可以是多个文件）')
    parser.add_argument('-o', '--output', default='./similar_errors_output', 
                       help='输出目录路径（默认: ./similar_errors_output）')
    parser.add_argument('-t', '--threshold', type=float, default=0.90,
                       help='代码和编译错误原因相似度阈值（默认: 0.90，即90%）')
    parser.add_argument('--analyze-only', action='store_true', help='仅分析相似度分布，不查找具体的相似组')
    parser.add_argument('--gpu', action='store_true', help='使用GPU加速计算（需要安装 torch 和 sentence-transformers）')
    parser.add_argument('--gpu-ids', type=str, default='0,1,2,3,4,5,6,7', 
                       help='指定使用的GPU卡号，用逗号分隔，例如: 0,1,2,3 (默认: 0,1,2,3,4,5,6,7)')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    for file_path in args.input_files:
        if not Path(file_path).exists():
            logger.error(f"输入文件不存在: {file_path}")
            return
    
    # 解析GPU IDs
    gpu_ids = None
    if args.gpu:
        try:
            gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
            print(f"🎯 指定GPU卡: {gpu_ids}")
        except:
            print("❌ GPU卡号格式错误，使用默认设置")
            gpu_ids = list(range(8))
        
        if check_gpu_availability():
            print("🚀 GPU可用，将使用多GPU并行计算")
        else:
            print("⚠️  GPU不可用，将使用CPU计算")
            print("   要使用GPU，请安装: pip install torch sentence-transformers")
            args.gpu = False
    
    try:
        if args.analyze_only:
            # 仅分析相似度分布
            all_data = []
            for file_path in args.input_files:
                data = load_json_file(file_path)
                all_data.extend(data)
            
            logger.info(f"总共加载了 {len(all_data)} 条语料")
            analysis = analyze_similarity_distribution(all_data)
            
            logger.info("\n代码和编译错误原因相似度分布分析结果:")
            for key, value in analysis.items():
                if key != 'high_similarity_examples':
                    logger.info(f"  {key}: {value}")
            
            if 'high_similarity_examples' in analysis:
                logger.info("\n高相似度语料示例:")
                for example in analysis['high_similarity_examples']:
                    logger.info(f"  {example['uuid1']} vs {example['uuid2']}: {example['similarity']:.3f}")
        else:
            # 完整分析并保存相似组
            process_multiple_files(args.input_files, args.output, args.threshold, args.gpu, gpu_ids)
            
    except Exception as e:
        logger.error(f"处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 