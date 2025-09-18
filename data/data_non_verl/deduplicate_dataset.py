#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集去重脚本
基于相似性分析结果，从原始数据中移除重复项，每个相似组只保留一个样本
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Set
import os

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def load_similarity_summary(summary_path: str) -> Dict:
    """加载相似性分析汇总文件"""
    try:
        logger.info(f"正在加载相似性分析汇总: {summary_path}")
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        logger.info(f"汇总信息: {summary['total_groups']} 个相似组，{summary['total_duplicates']} 个重复项")
        return summary
    except Exception as e:
        logger.error(f"加载相似性汇总文件失败: {e}")
        raise


def extract_duplicate_uuids(summary: Dict) -> Set[str]:
    """从汇总信息中提取需要删除的重复UUID"""
    uuids_to_remove = set()
    keep_first_uuids = set()
    
    for group_info in summary['groups_info']:
        uuids = group_info['uuids']
        if len(uuids) > 1:
            # 保留第一个UUID，删除其余的
            keep_first_uuids.add(uuids[0])
            for uuid in uuids[1:]:
                uuids_to_remove.add(uuid)
    
    logger.info(f"将保留 {len(keep_first_uuids)} 个代表性样本")
    logger.info(f"将删除 {len(uuids_to_remove)} 个重复样本")
    
    return uuids_to_remove


def deduplicate_dataset(data: List[Dict], uuids_to_remove: Set[str]) -> List[Dict]:
    """从数据集中移除重复项"""
    logger.info("开始去重处理...")
    
    original_count = len(data)
    deduplicated_data = []
    removed_count = 0
    
    for item in data:
        uuid = item.get('uuid', '')
        if uuid in uuids_to_remove:
            removed_count += 1
            logger.debug(f"删除重复项: {uuid}")
        else:
            deduplicated_data.append(item)
    
    final_count = len(deduplicated_data)
    logger.info(f"去重完成:")
    logger.info(f"  原始数据: {original_count:,} 条")
    logger.info(f"  删除重复: {removed_count:,} 条")
    logger.info(f"  保留数据: {final_count:,} 条")
    logger.info(f"  去重率: {(removed_count/original_count)*100:.2f}%")
    
    return deduplicated_data


def create_deduplication_report(original_count: int, final_count: int, removed_count: int, summary: Dict, output_path: str) -> None:
    """创建去重报告"""
    report = {
        "deduplication_summary": {
            "original_count": original_count,
            "final_count": final_count,
            "removed_count": removed_count,
            "deduplication_rate": round((removed_count/original_count)*100, 2)
        },
        "similarity_analysis": {
            "total_similar_groups": summary['total_groups'],
            "total_duplicates_found": summary['total_duplicates'],
            "analysis_method": "TF-IDF + 代码结构分析 + difflib",
            "similarity_threshold": "90%"
        },
        "group_size_distribution": {}
    }
    
    # 统计相似组大小分布
    size_distribution = {}
    for group_info in summary['groups_info']:
        group_size = group_info['count']
        size_distribution[group_size] = size_distribution.get(group_size, 0) + 1
    
    report["group_size_distribution"] = size_distribution
    
    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"去重报告已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='数据集去重工具 - 基于相似性分析结果去除重复项')
    parser.add_argument('input_file', help='输入的原始数据文件路径')
    parser.add_argument('-s', '--summary', required=True, 
                       help='相似性分析汇总文件路径 (compile_errors_similarity_summary.json)')
    parser.add_argument('-o', '--output', help='输出的去重数据文件路径（默认在输入文件同目录下创建）')
    parser.add_argument('--suffix', default='_deduplicated', 
                       help='输出文件后缀（默认: _deduplicated）')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not Path(args.input_file).exists():
        logger.error(f"输入文件不存在: {args.input_file}")
        return
    
    if not Path(args.summary).exists():
        logger.error(f"相似性汇总文件不存在: {args.summary}")
        return
    
    try:
        # 确定输出路径
        if args.output:
            output_path = args.output
        else:
            input_path = Path(args.input_file)
            output_path = input_path.parent / f"{input_path.stem}{args.suffix}{input_path.suffix}"
        
        # 加载数据
        original_data = load_json_file(args.input_file)
        summary = load_similarity_summary(args.summary)
        
        # 提取需要删除的UUID
        uuids_to_remove = extract_duplicate_uuids(summary)
        
        # 执行去重
        deduplicated_data = deduplicate_dataset(original_data, uuids_to_remove)
        
        # 保存去重后的数据
        save_json_file(deduplicated_data, str(output_path))
        
        # 创建去重报告
        report_path = str(output_path).replace('.json', '_deduplication_report.json')
        create_deduplication_report(
            len(original_data), 
            len(deduplicated_data), 
            len(original_data) - len(deduplicated_data),
            summary, 
            report_path
        )
        
        print(f"\n✅ 去重完成！")
        print(f"📊 统计信息:")
        print(f"   - 原始数据: {len(original_data):,} 条")
        print(f"   - 去重后数据: {len(deduplicated_data):,} 条")
        print(f"   - 删除重复: {len(original_data) - len(deduplicated_data):,} 条")
        print(f"   - 去重率: {((len(original_data) - len(deduplicated_data))/len(original_data))*100:.2f}%")
        print(f"💾 输出文件:")
        print(f"   - 去重数据: {output_path}")
        print(f"   - 去重报告: {report_path}")
        
    except Exception as e:
        logger.error(f"去重处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 