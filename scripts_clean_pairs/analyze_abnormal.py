#!/usr/bin/env python3
"""
分析词对/短语对文件：按 abnormal_word 分组，统计每个前词/短语的出现次数
自动检测列名（支持 prev_word / prev_phrase）
用法: python analyze_pairs.py <input.csv>
输出: abnormal_word_summary.csv, abnormal_word_details.csv
"""

import pandas as pd
from collections import Counter
import sys
from pathlib import Path

def detect_prev_column(df):
    """自动检测前置词/短语的列名"""
    possible_cols = ['prev_word', 'prev_phrase', 'prev', 'phrase']
    for col in possible_cols:
        if col in df.columns:
            return col
    # 如果都不匹配，尝试使用第一列（但需排除 abnormal_word）
    for col in df.columns:
        if col != 'abnormal_word':
            return col
    raise ValueError("无法自动检测前置词/短语列，请确保列名为 'prev_word' 或 'prev_phrase'")

def analyze_pairs(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    print(f"加载 {len(df)} 条记录")

    # 检查必需列
    if 'abnormal_word' not in df.columns:
        raise ValueError("文件缺少 'abnormal_word' 列")
    prev_col = detect_prev_column(df)
    print(f"检测到前置词列: {prev_col}")

    # 分组统计
    grouped = df.groupby('abnormal_word')[prev_col].agg(list).reset_index()
    grouped.columns = ['abnormal_word', 'prev_list']

    summary = []
    details = []
    for _, row in grouped.iterrows():
        word = row['abnormal_word']
        prev_items = row['prev_list']
        counter = Counter(prev_items)
        total = len(prev_items)
        unique = len(counter)
        top5 = counter.most_common(5)
        summary.append({
            'abnormal_word': word,
            'total_occurrences': total,
            'unique_prev': unique,
            'top_prev': str(top5)[:200]  # 限制长度
        })
        for item, cnt in counter.items():
            details.append({
                'abnormal_word': word,
                prev_col: item,
                'count': cnt
            })

    summary_df = pd.DataFrame(summary)
    details_df = pd.DataFrame(details)

    summary_path = output_dir / 'abnormal_word_summary.csv'
    details_path = output_dir / 'abnormal_word_details.csv'
    summary_df.to_csv(summary_path, index=False, encoding='utf-8')
    details_df.to_csv(details_path, index=False, encoding='utf-8')
    print(f"汇总表保存至 {summary_path}")
    print(f"详情表保存至 {details_path}")

    # 打印前10个高频异常词
    top = summary_df.nlargest(10, 'total_occurrences')[['abnormal_word', 'total_occurrences', 'unique_prev']]
    print("\nTop 10 异常词（按总出现次数）:")
    print(top.to_string(index=False))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("用法: python analyze_pairs.py <input.csv>")
        sys.exit(1)
    csv_file = Path(sys.argv[1])
    if not csv_file.exists():
        print(f"文件不存在: {csv_file}")
        sys.exit(1)
    output_dir = csv_file.parent
    analyze_pairs(csv_file, output_dir)