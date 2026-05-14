# scripts_clean_pairs/pair_filters.py
"""
可复用的词对过滤函数
"""

import re
import pandas as pd

# ---------- 数字检测 ----------
def is_digit_only(s: str) -> bool:
    """判断字符串是否只包含数字（阿拉伯数字或中文大写数字）"""
    if not s:
        return False
    if s.isdigit():
        return True
    chinese_digits = set("零一二三四五六七八九十百千万亿")
    return all(ch in chinese_digits for ch in s)

def filter_digit_pairs(df: pd.DataFrame, drop_if_both_digit: bool = True) -> pd.DataFrame:
    """
    过滤掉前后都是纯数字的词对
    """
    if drop_if_both_digit:
        mask = df.apply(
            lambda row: not (is_digit_only(row['prev_word']) and is_digit_only(row['abnormal_word'])),
            axis=1
        )
        return df[mask]
    return df

# ---------- 频次过滤 ----------
def filter_by_min_count(df: pd.DataFrame, min_count: int) -> pd.DataFrame:
    """
    过滤低频词对（基于 (prev_word, abnormal_word) 组合的出现次数）
    """
    pair_counts = df.groupby(['prev_word', 'abnormal_word']).size().reset_index(name='count')
    return pair_counts[pair_counts['count'] >= min_count]

# ---------- 按前置词聚合 ----------
def aggregate_by_prev(df: pd.DataFrame) -> pd.DataFrame:
    """
    按前置词聚合，生成统计表
    """
    grouped = df.groupby('prev_word').agg(
        total_occurrences=('count', 'sum'),
        unique_abnormal=('abnormal_word', 'nunique'),
        abnormal_words=('abnormal_word', lambda x: ' '.join(x))
    ).reset_index()
    return grouped.sort_values('total_occurrences', ascending=False)

# ---------- 未来扩展（示例）----------
def filter_by_length(df: pd.DataFrame, min_len: int = 2, max_len: int = 20):
    """过滤过短或过长的词（可按需添加）"""
    mask = df['prev_word'].str.len().between(min_len, max_len) & \
           df['abnormal_word'].str.len().between(min_len, max_len)
    return df[mask]