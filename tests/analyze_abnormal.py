#!/usr/bin/env python3
"""
按异常词统计前置词分布，支持过滤低频和前置词种类数
输出格式: abnormal_word, total_occurrences, unique_prev, prev_words
用法:
    python analyze_abnormal_filter.py --input <noise_pairs.csv> --output <输出目录> --min_total <阈值> --min_unique_prev <阈值>
"""

import pandas as pd
import argparse
from pathlib import Path

def main():
    # ================== 可修改的硬编码默认值 ==================
    DEFAULT_INPUT = "work/test_gpt2_sample_10_pt/outputs/sample_20_analysis/prev_window_1/noise_pairs.csv"
    DEFAULT_OUTPUT = "work/test_gpt2_sample_10_pt/outputs/sample_20_analysis/abnormal_clean"
    DEFAULT_MIN_TOTAL = 2          # 只保留异常词总出现次数≥2
    DEFAULT_MIN_UNIQUE_PREV = 1    # 默认不按前置词种类过滤（1表示至少1种）
    # ========================================================

    parser = argparse.ArgumentParser(description="按异常词统计前置词分布")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT,
                        help=f"输入的CSV文件路径（默认: {DEFAULT_INPUT}）")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help=f"输出目录（默认: {DEFAULT_OUTPUT}）")
    parser.add_argument("--min_total", type=int, default=DEFAULT_MIN_TOTAL,
                        help=f"异常词最小总出现次数（默认: {DEFAULT_MIN_TOTAL}）")
    parser.add_argument("--min_unique_prev", type=int, default=DEFAULT_MIN_UNIQUE_PREV,
                        help=f"前置词最小种类数（默认: {DEFAULT_MIN_UNIQUE_PREV}）")
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

    # 按异常词分组聚合
    grouped = df.groupby('abnormal_word').agg(
        total_occurrences=('prev_word', 'count'),
        unique_prev=('prev_word', 'nunique'),
        prev_words=('prev_word', lambda x: ' '.join(set(x)))   # 去重后的前置词，用空格分隔
    ).reset_index()

    # 过滤总出现次数
    before_total = len(grouped)
    grouped = grouped[grouped['total_occurrences'] >= args.min_total]
    print(f"按总出现次数过滤后剩余异常词数量: {len(grouped)} (要求 ≥ {args.min_total})")

    # 过滤前置词种类数
    grouped = grouped[grouped['unique_prev'] >= args.min_unique_prev]
    print(f"按前置词种类数过滤后剩余异常词数量: {len(grouped)} (要求 unique_prev ≥ {args.min_unique_prev})")

    # 按总出现次数降序排序
    grouped = grouped.sort_values('total_occurrences', ascending=False)

    # 输出到CSV
    output_file = output_dir / "abnormal_clean_summary.csv"
    grouped.to_csv(output_file, index=False, encoding='utf-8')
    print(f"结果已保存至: {output_file}")

    # 打印前10行预览
    print("\n前10个异常词统计:")
    print(grouped.head(10).to_string(index=False))

if __name__ == "__main__":
    main()