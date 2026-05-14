#!/usr/bin/env python3
"""
过滤低频词对，生成前置词统计表（prev_word, total_occurrences, unique_abnormal, abnormal_words）
支持：
   - 根据词对频次过滤（min_count）
   - 根据前置词对应的异常词种类数过滤（min_unique_abnormal）
用法:
    python filter_pairs.py --input <噪声对文件> --output <输出目录> --min_count <阈值> --min_unique_abnormal <阈值>
默认硬编码可自行修改。
"""

import pandas as pd
import argparse
from pathlib import Path

def main():
    # ================== 可修改的硬编码默认值 ==================
    DEFAULT_INPUT = "work/test_gpt2_sample_10_pt/outputs/sample_20_analysis/prev_window_1/noise_pairs.csv"
    DEFAULT_OUTPUT = "work/test_gpt2_sample_10_pt/outputs/sample_20_analysis/prev_clean"
    DEFAULT_MIN_COUNT = 2           # 只保留出现次数≥2的词对
    DEFAULT_MIN_UNIQUE_ABNORMAL = 2 # 默认不按异常词种类过滤（1表示至少1种）
    # ========================================================

    parser = argparse.ArgumentParser(description="从noise_pairs.csv中过滤低频词对，生成前置词统计")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT,
                        help=f"输入的CSV文件路径（默认: {DEFAULT_INPUT}）")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help=f"输出目录（默认: {DEFAULT_OUTPUT}）")
    parser.add_argument("--min_count", type=int, default=DEFAULT_MIN_COUNT,
                        help=f"最小频次阈值，保留出现次数≥该值的词对（默认: {DEFAULT_MIN_COUNT}）")
    parser.add_argument("--min_unique_abnormal", type=int, default=DEFAULT_MIN_UNIQUE_ABNORMAL,
                        help=f"最小异常词种类数，保留前置词对应的不同异常词数量≥该值（默认: {DEFAULT_MIN_UNIQUE_ABNORMAL}，设为1则不过滤）")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"读取文件: {input_path}")
    df = pd.read_csv(input_path)
    print(f"原始词对数量: {len(df)}")

    # 检查列名
    if 'prev_word' not in df.columns or 'abnormal_word' not in df.columns:
        raise ValueError("CSV文件必须包含 'prev_word' 和 'abnormal_word' 列")

    # 按 (prev_word, abnormal_word) 分组统计频次
    pair_counts = df.groupby(['prev_word', 'abnormal_word']).size().reset_index(name='count')
    # 过滤低频词对
    filtered = pair_counts[pair_counts['count'] >= args.min_count]
    print(f"过滤后唯一词对数量: {len(filtered)} (频次 ≥ {args.min_count})")

    # 按前置词分组聚合
    grouped = filtered.groupby('prev_word').agg(
        total_occurrences=('count', 'sum'),
        unique_abnormal=('abnormal_word', 'nunique'),
        abnormal_words=('abnormal_word', lambda x: ' '.join(x))
    ).reset_index()

    # 按 unique_abnormal 过滤
    before_unique = len(grouped)
    grouped = grouped[grouped['unique_abnormal'] >= args.min_unique_abnormal]
    print(f"按异常词种类数过滤后剩余前置词数量: {len(grouped)} (要求 unique_abnormal ≥ {args.min_unique_abnormal})")

    # 按总出现次数降序排序
    grouped = grouped.sort_values('total_occurrences', ascending=False)

    # 输出到CSV
    output_file = output_dir / "prev_clean_summary.csv"
    grouped.to_csv(output_file, index=False, encoding='utf-8')
    print(f"结果已保存至: {output_file}")

    # 打印前10行预览
    print("\n前10个前置词统计:")
    print(grouped.head(10).to_string(index=False))

if __name__ == "__main__":
    main()