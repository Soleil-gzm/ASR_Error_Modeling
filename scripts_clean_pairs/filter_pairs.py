#!/usr/bin/env python3
"""
过滤低频词对，生成前置词统计表（prev_word, total_occurrences, unique_abnormal, abnormal_words）
支持：
   - 根据词对频次过滤（min_count）
   - 根据前置词对应的异常词种类数过滤（min_unique_abnormal）
   - 可选：过滤掉前后都是纯数字的词对
   - 异常词后自动附加出现概率（括号内小数）
用法:
    python filter_pairs.py --input <噪声对文件> --output <输出目录> [--min_count 2] [--min_unique_abnormal 2] [--drop_digit_pairs]
"""

import pandas as pd
import argparse
from pathlib import Path
from utils.pair_filters import filter_digit_pairs, filter_by_min_count, aggregate_by_prev

def main():
    # ================== 可修改的硬编码默认值 ==================
    DEFAULT_INPUT = "work/test_Qwen_pt/outputs/sample_20_analysis/prev_window_1/noise_pairs.csv"
    DEFAULT_OUTPUT = "work/test_Qwen_pt/outputs/sample_20_analysis/prev_clean"
    DEFAULT_MIN_COUNT = 2
    DEFAULT_MIN_UNIQUE_ABNORMAL = 2
    DEFAULT_DROP_DIGIT_PAIRS = True   # 默认过滤纯数字对
    # ========================================================

    parser = argparse.ArgumentParser(description="从noise_pairs.csv中过滤低频词对，生成前置词统计")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT,
                        help=f"输入的CSV文件路径（默认: {DEFAULT_INPUT}）")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help=f"输出目录（默认: {DEFAULT_OUTPUT}）")
    parser.add_argument("--min_count", type=int, default=DEFAULT_MIN_COUNT,
                        help=f"最小频次阈值，保留出现次数≥该值的词对（默认: {DEFAULT_MIN_COUNT}）")
    parser.add_argument("--min_unique_abnormal", type=int, default=DEFAULT_MIN_UNIQUE_ABNORMAL,
                        help=f"最小异常词种类数，保留前置词对应的不同异常词数量≥该值（默认: {DEFAULT_MIN_UNIQUE_ABNORMAL}）")
    parser.add_argument("--drop_digit_pairs", action='store_true', default=DEFAULT_DROP_DIGIT_PAIRS,
                        help="是否过滤掉前置词和异常词都是纯数字的词对（默认启用）")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"读取文件: {input_path}")
    df = pd.read_csv(input_path)
    print(f"原始词对数量: {len(df)}")

    # 1. 数字对过滤
    if args.drop_digit_pairs:
        before = len(df)
        df = filter_digit_pairs(df)
        print(f"数字对过滤后剩余 {len(df)} 条 (移除 {before - len(df)})")

    # 2. 频次过滤（基于 (prev_word, abnormal_word) 组合）
    df_counts = filter_by_min_count(df, args.min_count)
    print(f"频次过滤后唯一词对数量: {len(df_counts)} (要求出现≥{args.min_count})")

    # 3. 按前置词聚合（自动附加概率）
    grouped = aggregate_by_prev(df_counts, with_prob=True)

    # 4. 按异常词种类过滤
    before = len(grouped)
    grouped = grouped[grouped['unique_abnormal'] >= args.min_unique_abnormal]
    print(f"异常词种类过滤后剩余前置词数量: {len(grouped)} (要求 unique_abnormal ≥ {args.min_unique_abnormal})")

    # 输出
    output_file = output_dir / "prev_clean_summary.csv"
    grouped.to_csv(output_file, index=False, encoding='utf-8')
    print(f"结果已保存至: {output_file}")

    # 预览
    print("\n前10个前置词统计:")
    print(grouped.head(10).to_string(index=False))

if __name__ == "__main__":
    main()