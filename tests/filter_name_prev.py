#!/usr/bin/env python3
"""
筛选含有姓名或称谓的前置词对
从 prev_clean_summary.csv 中提取并保存到单独文件
"""

import re
import pandas as pd
from pathlib import Path

# ================== 硬编码路径 ==================
# INPUT_CSV = "work/test_gpt2_sample_10_pt/outputs/sample_20_analysis/prev_clean/prev_clean_summary.csv"
# OUTPUT_CSV = "work/test_gpt2_sample_10_pt/outputs/sample_20_analysis/prev_clean/prev_name_like.csv"

INPUT_CSV = "work/test_Qwen_pt/outputs/sample_20_analysis/prev_clean/prev_clean_summary.csv"
OUTPUT_CSV = "work/test_Qwen_pt/outputs/sample_20_analysis/prev_clean/prev_name_like.csv"
# ===============================================

# 常见中文姓氏（百家姓前几常用）
COMMON_SURNAMES = {
    "李", "王", "张", "刘", "陈", "杨", "赵", "黄", "周", "吴",
    "徐", "孙", "马", "朱", "胡", "林", "郭", "何", "高", "郑",
    "罗", "梁", "谢", "宋", "唐", "邓", "萧", "冯", "韩", "曹",
    "彭", "曾", "肖", "田", "董", "袁", "潘", "于", "蒋", "蔡",
    "余", "杜", "戴", "夏", "钟", "汪", "田", "任", "姜", "范"
}

def is_name_like(word: str) -> bool:
    """判断是否为姓名（至少2字符，首字为常见姓氏）"""
    if not isinstance(word, str) or len(word) < 2:
        return False
    return word[0] in COMMON_SURNAMES

def is_honorific(word: str) -> bool:
    """判断是否为称谓词（如某先生、某女士、某总等）"""
    if not isinstance(word, str):
        return False
    patterns = [
        r'.*先生$', r'.*女士$', r'.*小姐$', r'.*总$', r'.*经理$',
        r'.*老师$', r'.*医生$', r'.*老板$'
    ]
    for pat in patterns:
        if re.search(pat, word):
            return True
    return False

def main():
    input_path = Path(INPUT_CSV)
    if not input_path.exists():
        print(f"错误：输入文件不存在 {input_path}")
        return

    df = pd.read_csv(input_path)
    print(f"原始数据行数: {len(df)}")

    # 筛选条件：前置词是姓名 或 包含称谓词
    mask_name = df['prev_word'].apply(is_name_like)
    mask_honor = df['prev_word'].apply(is_honorific)
    filtered_df = df[mask_name | mask_honor]

    print(f"筛选出 {len(filtered_df)} 行 (姓名或称谓)")

    if len(filtered_df) > 0:
        output_path = Path(OUTPUT_CSV)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        filtered_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"结果已保存至: {output_path}")
        print("\n前10条示例：")
        print(filtered_df.head(10).to_string(index=False))
    else:
        print("未找到符合条件的行。")

if __name__ == "__main__":
    main()