#!/usr/bin/env python3
"""
后处理：将 token 级 NLL 聚合为词级 NLL
输入：word_nll_details.csv（由 03_compute_word_nll.py 生成）
输出：word_level_nll.csv（每个词的平均 NLL）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import jieba
from tqdm import tqdm
from scripts.utils import setup_logger

def aggregate_to_words(df):
    """
    对每个句子，将连续 token 按 jieba 分词边界聚合，计算每个词的平均 NLL
    返回新的 DataFrame，每行对应一个词
    """
    records = []
    # 按句子分组
    for sent_id, group in tqdm(df.groupby('sentence_id'), desc="聚合词级NLL"):
        sentence = group['sentence'].iloc[0]
        # 获取该句子的所有 token 及其 NLL（按 token_index 排序）
        token_rows = group.sort_values('token_index')
        tokens = token_rows['token'].tolist()
        nlls = token_rows['nll'].tolist()
        
        # 用 jieba 对原句子进行分词，得到词及其字符区间
        words = list(jieba.cut(sentence))
        # 构建字符位置到词的映射
        word_spans = []
        cur = 0
        for w in words:
            start = cur
            end = cur + len(w)
            word_spans.append((start, end, w))
            cur = end
        
        # 将 token 聚合到词上（通过字符串匹配，注意可能边界不完全对齐）
        # 更稳健的方法：直接按顺序匹配，因为 token 顺序与字符顺序一致
        token_idx = 0
        for start, end, word in word_spans:
            # 收集属于这个词的 token NLL
            word_nlls = []
            # 积累 token 直到覆盖完这个词的字符
            char_pos = start
            while token_idx < len(tokens) and char_pos < end:
                token = tokens[token_idx]
                token_len = len(token)
                # 假设 token 连续且与词对齐（如果不对齐会有问题，但一般情况正确）
                word_nlls.append(nlls[token_idx])
                char_pos += token_len
                token_idx += 1
            if word_nlls:
                avg_nll = sum(word_nlls) / len(word_nlls)
                records.append({
                    'sentence_id': sent_id,
                    'word': word,
                    'avg_nll': avg_nll,
                    'sentence': sentence
                })
            else:
                # 没有对应 token（异常情况）
                records.append({
                    'sentence_id': sent_id,
                    'word': word,
                    'avg_nll': float('nan'),
                    'sentence': sentence
                })
    return pd.DataFrame(records)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="输入的 word_nll_details.csv 路径")
    parser.add_argument("--output_csv", type=str, required=True, help="输出的词级 CSV 路径")
    parser.add_argument("--log_dir", type=str, default="logs", help="日志目录")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    logger = setup_logger(log_dir, "aggregate_to_words")
    logger.info(f"读取文件: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    logger.info(f"共 {len(df)} 条 token 级记录")

    result_df = aggregate_to_words(df)
    logger.info(f"聚合得到 {len(result_df)} 个词")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"词级结果已保存至: {output_path}")

if __name__ == "__main__":
    main()


'''
python scripts/aggregate_to_words.py \
    --input_csv <输入的word_nll_details.csv路径> \
    --output_csv <输出的词级CSV路径> \
    --log_dir <日志目录（可选）>

python scripts/aggregate_to_words.py \
    --input_csv work/test_v2/outputs/word_nll_details.csv \
    --output_csv work/test_v2/outputs/word_level_nll.csv \
    --log_dir work/test_v2/logs

# 命令行运行案例
python scripts/aggregate_to_words.py \
    --input_csv work/test_Qwen_P80_10/outputs/word_nll_details.csv \
    --output_csv work/test_Qwen_P80_10/outputs/word_level_nll.csv \
    --log_dir work/test_Qwen_P80_10/logs

'''
