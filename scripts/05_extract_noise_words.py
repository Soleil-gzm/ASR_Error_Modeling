#!/usr/bin/env python3
"""
步骤05：从词级NLL异常词中提取前文词语与异常词的对应关系
输出：prev_word, abnormal_word 对
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
    logger.info("开始提取异常词与前文词的对应关系")

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
    prev_window = step_cfg.get('prev_window', 1)
    threshold_percentile = step_cfg.get('threshold_percentile', 95)

    # 计算异常词阈值
    threshold = df['avg_nll'].quantile(threshold_percentile / 100.0)
    logger.info(f"异常词 NLL 阈值 ({threshold_percentile}%): {threshold:.4f}")

    # 提取对偶关系
    pairs = []
    for sent_id, group in df.sort_values(['sentence_id', 'word_index']).groupby('sentence_id'):
        words = group['word'].tolist()
        nlls = group['avg_nll'].tolist()
        for i, (w, nll) in enumerate(zip(words, nlls)):
            if nll > threshold:
                start = max(0, i - prev_window)
                for j in range(start, i):
                    prev_word = words[j]
                    pairs.append((prev_word, w))

    logger.info(f"共提取 {len(pairs)} 条 (前文词, 异常词) 对")

    if len(pairs) == 0:
        logger.warning("未找到任何异常词对，请检查阈值")
        output_csv = output_dir / "noise_pairs.csv"
        pd.DataFrame(columns=['prev_word', 'abnormal_word']).to_csv(output_csv, index=False)
        logger.info(f"空文件已保存至 {output_csv}")
        sys.exit(0)

    # 保存为 CSV
    pairs_df = pd.DataFrame(pairs, columns=['prev_word', 'abnormal_word'])
    output_csv = output_dir / "noise_pairs.csv"
    pairs_df.to_csv(output_csv, index=False, encoding='utf-8')
    logger.info(f"噪声对已保存至 {output_csv}，共 {len(pairs_df)} 条记录")

    # 统计信息
    stats = {
        "total_pairs": len(pairs),
        "unique_prev_words": pairs_df['prev_word'].nunique(),
        "unique_abnormal_words": pairs_df['abnormal_word'].nunique(),
        "threshold_percentile": threshold_percentile,
        "prev_window": prev_window
    }
    with open(output_dir / "noise_pairs_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"统计信息已保存至 {output_dir}/noise_pairs_stats.json")

if __name__ == "__main__":
    main()