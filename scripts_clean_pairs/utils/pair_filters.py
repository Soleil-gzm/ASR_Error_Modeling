# scripts_clean_pairs/pair_filters.py
import pandas as pd

def is_digit_only(s: str) -> bool:
    """判断字符串是否只包含数字（阿拉伯数字或中文大写数字）"""
    if not s:
        return False
    if s.isdigit():
        return True
    chinese_digits = set("零一二三四五六七八九十百千万亿")
    return all(ch in chinese_digits for ch in s)

def filter_digit_pairs(df: pd.DataFrame, drop_if_both_digit: bool = True) -> pd.DataFrame:
    """过滤掉前后都是纯数字的词对"""
    if drop_if_both_digit:
        mask = df.apply(
            lambda row: not (is_digit_only(row['prev_word']) and is_digit_only(row['abnormal_word'])),
            axis=1
        )
        return df[mask]
    return df

def filter_by_min_count(df: pd.DataFrame, min_count: int) -> pd.DataFrame:
    """过滤低频词对（基于 (prev_word, abnormal_word) 组合的出现次数）"""
    pair_counts = df.groupby(['prev_word', 'abnormal_word']).size().reset_index(name='count')
    return pair_counts[pair_counts['count'] >= min_count]

def aggregate_by_prev(df: pd.DataFrame, with_prob: bool = True) -> pd.DataFrame:
    """
    按前置词聚合，生成统计表
    df 必须包含 count 列（由 filter_by_min_count 生成）
    with_prob: 是否在 abnormal_words 列中附加概率（默认 True）
    """
    if 'count' not in df.columns:
        # 如果传入的是原始词对而非频次表，先进行统计
        df = df.groupby(['prev_word', 'abnormal_word']).size().reset_index(name='count')
    
    # 按前置词分组聚合
    grouped = df.groupby('prev_word').agg(
        total_occurrences=('count', 'sum'),
        unique_abnormal=('abnormal_word', 'nunique'),
        counts_list=('count', list),
        words_list=('abnormal_word', list)
    ).reset_index()
    
    if with_prob:
        def format_with_prob(words, counts):
            # 按概率降序排序
            pairs = sorted(zip(words, counts), key=lambda x: x[1], reverse=True)
            total = sum(counts)
            formatted = []
            for w, c in pairs:
                prob = c / total
                formatted.append(f"{w}({prob:.3f})")
            return ' '.join(formatted)
        
        grouped['abnormal_words'] = grouped.apply(
            lambda row: format_with_prob(row['words_list'], row['counts_list']), axis=1
        )
        # 删除辅助列
        grouped = grouped.drop(columns=['counts_list', 'words_list'])
    else:
        # 不计算概率时，只拼接异常词（也可以按出现次数排序，便于阅读）
        def sort_by_count(words, counts):
            pairs = sorted(zip(words, counts), key=lambda x: x[1], reverse=True)
            return ' '.join(w for w, _ in pairs)
        grouped['abnormal_words'] = grouped.apply(
            lambda row: sort_by_count(row['words_list'], row['counts_list']), axis=1
        )
        grouped = grouped.drop(columns=['counts_list', 'words_list'])
    
    return grouped.sort_values('total_occurrences', ascending=False)

# 未来扩展示例
def filter_by_length(df: pd.DataFrame, min_len: int = 2, max_len: int = 20):
    mask = df['prev_word'].str.len().between(min_len, max_len) & \
           df['abnormal_word'].str.len().between(min_len, max_len)
    return df[mask]