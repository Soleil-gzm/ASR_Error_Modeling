#!/usr/bin/env python3
"""
统计前置词（短语）后面出现的异常词类型及频次
支持单词对（prev_word, abnormal_word）和短语对（prev_phrase, abnormal_word）
用法: python analyze_prev.py <input.csv>
输出:
   - prev_summary.csv: 每个前置词的统计（总次数、不同异常词数、top异常词）
   - prev_details.csv: 每个 (前置词, 异常词) 的详细计数
"""

import pandas as pd
from collections import Counter
import sys
from pathlib import Path

def detect_prev_column(df):
    """自动检测前置词/短语列名"""
    possible = ['prev_word', 'prev_phrase', 'prev', 'phrase']
    for col in possible:
        if col in df.columns:
            return col
    # 尝试第一列（排除 abnormal_word）
    for col in df.columns:
        if col != 'abnormal_word':
            return col
    raise ValueError("无法自动检测前置词列")

def analyze_prev(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    print(f"加载 {len(df)} 条记录")

    if 'abnormal_word' not in df.columns:
        raise ValueError("文件缺少 'abnormal_word' 列")
    prev_col = detect_prev_column(df)
    print(f"检测到前置词列: {prev_col}")

    # 按前置词分组，收集异常词列表
    grouped = df.groupby(prev_col)['abnormal_word'].agg(list).reset_index()
    grouped.columns = ['prev', 'abnormal_list']

    summary = []
    details = []
    for _, row in grouped.iterrows():
        prev = row['prev']
        abnormals = row['abnormal_list']
        counter = Counter(abnormals)
        total = len(abnormals)
        unique = len(counter)
        top5 = counter.most_common(5)
        summary.append({
            'prev': prev,
            'total_occurrences': total,
            'unique_abnormal': unique,
            'top_abnormal': str(top5)[:200]
        })
        for ab, cnt in counter.items():
            details.append({
                'prev': prev,
                'abnormal_word': ab,
                'count': cnt
            })

    summary_df = pd.DataFrame(summary)
    details_df = pd.DataFrame(details)

    summary_path = output_dir / 'prev_summary.csv'
    details_path = output_dir / 'prev_details.csv'
    summary_df.to_csv(summary_path, index=False, encoding='utf-8')
    details_df.to_csv(details_path, index=False, encoding='utf-8')
    print(f"前置词汇总表保存至 {summary_path}")
    print(f"前置词-异常词明细表保存至 {details_path}")

    # 打印前10个最常触发异常的前置词
    top_prev = summary_df.nlargest(10, 'total_occurrences')[['prev', 'total_occurrences', 'unique_abnormal']]
    print("\nTop 10 前置词（按触发总次数）:")
    print(top_prev.to_string(index=False))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("用法: python analyze_prev.py <input.csv>")
        sys.exit(1)
    csv_file = Path(sys.argv[1])
    if not csv_file.exists():
        print(f"文件不存在: {csv_file}")
        sys.exit(1)
    output_dir = csv_file.parent
    analyze_prev(csv_file, output_dir)