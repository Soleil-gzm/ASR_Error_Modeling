#!/usr/bin/env python3
"""
步骤05：从词级NLL异常词中提取前文词语与异常词的对应关系
输出：
  - noise_pairs.csv: 每个前驱词单独成对 (prev_word, abnormal_word)
  - noise_pairs_phrase.csv: 将前 prev_window 个词拼接成短语 (prev_phrase, abnormal_word)
支持配置前文窗口大小（prev_window: 1 或 2）
"""

import sys
import json
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utils import setup_logger
from scripts.utils import get_step_output, get_step_sample_ratio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", type=str, required=True)
    args = parser.parse_args()

    config = json.loads(args.config_json)
    task_name = config['task_name']
    base_dir = Path(config['paths']['output']['base_dir'])
    task_dir = base_dir / task_name
    project_root = base_dir.parent

    metadata_path = task_dir / "run_metadata.json"
    if not metadata_path.exists():
        print("错误：未找到 run_metadata.json")
        sys.exit(1)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # 日志
    log_dir = task_dir / "logs"
    logger = setup_logger(log_dir, "05_extract_noise_words")
    logger.info("开始提取异常词与前文词的对应关系（单词对 + 短语对）")

    # 获取步骤04的输出目录（绝对路径）
    step4_output_dir_rel = get_step_output(metadata, '04_statistics', key='output_dir')
    if step4_output_dir_rel is None:
        logger.error("无法从元数据获取步骤04的输出目录")
        sys.exit(1)
    output_dir = project_root / Path(step4_output_dir_rel)
    logger.info(f"步骤04输出目录: {output_dir}")

    # 读取词级聚合文件
    agg_csv = output_dir / "word_level_aggregated.csv"
    if not agg_csv.exists():
        sample_ratio = get_step_sample_ratio(metadata, '01_compute_sentence_nll')
        if sample_ratio < 1.0:
            agg_csv = output_dir / f"word_level_aggregated_sample_{int(sample_ratio*100)}.csv"
        if not agg_csv.exists():
            logger.error(f"未找到词级聚合文件: {output_dir}/word_level_aggregated*.csv")
            sys.exit(1)
    df = pd.read_csv(agg_csv)
    logger.info(f"加载 {len(df)} 条词级记录")

    # 确保有 word_index 列
    if 'word_index' not in df.columns:
        df['word_index'] = df.groupby('sentence_id').cumcount()

    # 从配置文件读取参数
    step_cfg = config['steps'].get('05_extract_noise_words', {})
    prev_window = step_cfg.get('prev_window', 1)   # 前文窗口大小（1 或 2）
    threshold_percentile = step_cfg.get('threshold_percentile', 95)

    # 计算异常词阈值
    threshold = df['avg_nll'].quantile(threshold_percentile / 100.0)
    logger.info(f"异常词 NLL 阈值 ({threshold_percentile}%): {threshold:.4f}")

    # 提取两种对
    single_pairs = []   # (前一个词, 异常词)
    phrase_pairs = []   # (前两个词拼接, 异常词) 仅当 prev_window >= 2 且存在足够前文时

    for sent_id, group in df.sort_values(['sentence_id', 'word_index']).groupby('sentence_id'):
        words = group['word'].tolist()
        nlls = group['avg_nll'].tolist()
        for i, (w, nll) in enumerate(zip(words, nlls)):
            if nll > threshold:
                # 提取前 prev_window 个词（如果存在）
                start = max(0, i - prev_window)
                # 生成单词对：每个前驱词单独成对
                for j in range(start, i):
                    prev_word = words[j]
                    single_pairs.append((prev_word, w))
                # 如果窗口大于1且存在足够的前文词（至少 2 个），生成短语对
                if prev_window >= 2 and i - start >= prev_window:
                    # 提取连续的 prev_window 个词作为短语（用空格连接）
                    phrase_tokens = words[start:i]
                    phrase = ' '.join(phrase_tokens)
                    phrase_pairs.append((phrase, w))

    logger.info(f"共提取 {len(single_pairs)} 条单词对，{len(phrase_pairs)} 条短语对")

    # 保存单词对
    output_single_csv = output_dir / "noise_pairs.csv"
    if len(single_pairs) > 0:
        single_df = pd.DataFrame(single_pairs, columns=['prev_word', 'abnormal_word'])
        single_df.to_csv(output_single_csv, index=False, encoding='utf-8')
    else:
        pd.DataFrame(columns=['prev_word', 'abnormal_word']).to_csv(output_single_csv, index=False)
    logger.info(f"单词对已保存至 {output_single_csv}")

    # 保存短语对
    output_phrase_csv = output_dir / "noise_pairs_phrase.csv"
    if len(phrase_pairs) > 0:
        phrase_df = pd.DataFrame(phrase_pairs, columns=['prev_phrase', 'abnormal_word'])
        phrase_df.to_csv(output_phrase_csv, index=False, encoding='utf-8')
    else:
        pd.DataFrame(columns=['prev_phrase', 'abnormal_word']).to_csv(output_phrase_csv, index=False)
    logger.info(f"短语对已保存至 {output_phrase_csv}")

    # 统计信息
    stats = {
        "total_single_pairs": len(single_pairs),
        "unique_prev_words": len(set(p[0] for p in single_pairs)),
        "unique_abnormal_words": len(set(p[1] for p in single_pairs)),
        "total_phrase_pairs": len(phrase_pairs),
        "unique_phrases": len(set(p[0] for p in phrase_pairs)),
        "threshold_percentile": threshold_percentile,
        "prev_window": prev_window
    }
    with open(output_dir / "noise_pairs_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"统计信息已保存至 {output_dir}/noise_pairs_stats.json")

if __name__ == "__main__":
    main()