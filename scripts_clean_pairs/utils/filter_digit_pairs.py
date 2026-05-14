#!/usr/bin/env python3
"""
过滤噪声词对：删除前置词和异常词都是纯数字的词对
（其他包含数字的词对保留）
支持阿拉伯数字和大写中文数字
用法：
    python filter_digit_pairs.py --input <input.csv> --output <output.csv>
"""

import re
import argparse
import pandas as pd
from pathlib import Path

# ---------- 数字检测函数 ----------
def is_digit_only(s: str) -> bool:
    """
    判断字符串是否只包含数字（阿拉伯数字或中文大写数字）
    中文大写数字：零一二三四五六七八九十百千万亿
    """
    if not s:
        return False
    # 阿拉伯数字
    if s.isdigit():
        return True
    # 中文数字字符集
    chinese_digits = set("零一二三四五六七八九十百千万亿")
    # 检查每个字符是否都在中文数字字符集中
    for ch in s:
        if ch not in chinese_digits:
            return False
    return True

def filter_digit_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    过滤词对：删除 prev_word 和 abnormal_word 都是纯数字的行
    """
    mask = df.apply(
        lambda row: not (is_digit_only(row['prev_word']) and is_digit_only(row['abnormal_word'])),
        axis=1
    )
    return df[mask]

# ---------- 主函数 ----------
def main():
    parser = argparse.ArgumentParser(description="过滤词对中前后都是纯数字的行")
    parser.add_argument("--input", type=str, required=True, help="输入CSV文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出CSV文件路径")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"错误：输入文件不存在 {input_path}")
        return

    df = pd.read_csv(input_path)
    print(f"原始词对数量: {len(df)}")

    filtered_df = filter_digit_pairs(df)
    print(f"过滤后词对数量: {len(filtered_df)} (移除 {(len(df)-len(filtered_df))} 条)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"结果已保存至: {output_path}")

if __name__ == "__main__":
    main()