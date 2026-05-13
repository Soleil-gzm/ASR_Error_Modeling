#!/usr/bin/env python3
"""
步骤05：从词级NLL异常词中提取前文词语与异常词的对应关系
输出：
  - noise_pairs.csv: 每个前驱词单独成对 (prev_word, abnormal_word)
  - noise_pairs_phrase.csv: 将前 prev_window 个词拼接成短语 (prev_phrase, abnormal_word)
支持配置前文窗口大小（prev_window: 1 或 2）
输出目录按 prev_window 隔离，避免覆盖。
包含计时和元数据记录
"""

import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utils import setup_logger
from scripts.utils import get_step_output, get_step_sample_ratio
from scripts.utils.timer import TimedBlock, update_metadata_timing

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

    log_dir = task_dir / "logs"
    logger = setup_logger(log_dir, "05_extract_noise_words")
    logger.info("开始提取异常词与前文词的对应关系（单词对 + 短语对）")

    # 获取步骤04的输出目录（绝对路径）
    step4_output_dir_rel = get_step_output(metadata, '04_statistics', key='output_dir')
    if step4_output_dir_rel is None:
        logger.error("无法从元数据获取步骤04的输出目录")
        sys.exit(1)
    report_dir = project_root / Path(step4_output_dir_rel)
    # 输出目录改为 report_dir 的父目录（即 outputs/sample_20_analysis/）
    base_output_dir = report_dir.parent

    # 从配置文件读取参数
    step_cfg = config['steps'].get('05_extract_noise_words', {})
    prev_window = step_cfg.get('prev_window', 1)
    # 兼容两种键名
    threshold_percentile = step_cfg.get('threshold_percentile', step_cfg.get('nll_threshold_percentile', 95))

    # 根据 prev_window 创建子目录，避免覆盖
    output_dir = base_output_dir / f"prev_window_{prev_window}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"步骤04报告目录: {report_dir}")
    logger.info(f"噪声文件输出目录: {output_dir}")

    total_start = time.perf_counter()
    timing = {}

    # 1. 读取数据
    with TimedBlock("load_data", timing):
        agg_csv = report_dir / "word_level_aggregated.csv"
        if not agg_csv.exists():
            sample_ratio = get_step_sample_ratio(metadata, '01_compute_sentence_nll')
            if sample_ratio < 1.0:
                agg_csv = report_dir / f"word_level_aggregated_sample_{int(sample_ratio*100)}.csv"
            if not agg_csv.exists():
                logger.error(f"未找到词级聚合文件: {report_dir}/word_level_aggregated*.csv")
                sys.exit(1)
        df = pd.read_csv(agg_csv)
        if 'word_index' not in df.columns:
            df['word_index'] = df.groupby('sentence_id').cumcount()
    logger.info(f"加载 {len(df)} 条词级记录")

    # 2. 提取对
    with TimedBlock("extract_pairs", timing):
        threshold = df['avg_nll'].quantile(threshold_percentile / 100.0)
        logger.info(f"异常词 NLL 阈值 ({threshold_percentile}%): {threshold:.4f}")

        single_pairs = []
        phrase_pairs = []
        for sent_id, group in df.sort_values(['sentence_id', 'word_index']).groupby('sentence_id'):
            words = group['word'].tolist()
            nlls = group['avg_nll'].tolist()
            for i, (w, nll) in enumerate(zip(words, nlls)):
                if nll > threshold:
                    start = max(0, i - prev_window)
                    for j in range(start, i):
                        single_pairs.append((words[j], w))
                    if prev_window >= 2 and i - start >= prev_window:
                        phrase_tokens = words[start:i]
                        phrase = ' '.join(phrase_tokens)
                        phrase_pairs.append((phrase, w))

    logger.info(f"共提取 {len(single_pairs)} 条单词对，{len(phrase_pairs)} 条短语对")

    # 3. 保存文件到 output_dir
    with TimedBlock("save_output", timing):
        output_single_csv = output_dir / "noise_pairs.csv"
        if single_pairs:
            single_df = pd.DataFrame(single_pairs, columns=['prev_word', 'abnormal_word'])
            single_df.to_csv(output_single_csv, index=False, encoding='utf-8')
        else:
            pd.DataFrame(columns=['prev_word', 'abnormal_word']).to_csv(output_single_csv, index=False)

        output_phrase_csv = output_dir / "noise_pairs_phrase.csv"
        if phrase_pairs:
            phrase_df = pd.DataFrame(phrase_pairs, columns=['prev_phrase', 'abnormal_word'])
            phrase_df.to_csv(output_phrase_csv, index=False, encoding='utf-8')
        else:
            pd.DataFrame(columns=['prev_phrase', 'abnormal_word']).to_csv(output_phrase_csv, index=False)

        # 统计信息也放到 output_dir
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

    timing["total_sec"] = time.perf_counter() - total_start
    logger.info(f"单词对保存至 {output_single_csv}")
    logger.info(f"短语对保存至 {output_phrase_csv}")
    logger.info(f"统计信息保存至 {output_dir}/noise_pairs_stats.json")
    logger.info(f"总耗时: {timing['total_sec']:.2f}s")

    # 构建当前运行的计时记录
    current_timing = {
        "timestamp": datetime.now().isoformat(),
        "load_data_sec": timing.get("load_data", 0),
        "extract_pairs_sec": timing.get("extract_pairs", 0),
        "save_output_sec": timing.get("save_output", 0),
        "total_sec": timing["total_sec"],
        "num_single_pairs": len(single_pairs),
        "num_phrase_pairs": len(phrase_pairs),
        "prev_window": prev_window,
        "threshold_percentile": threshold_percentile
    }

    # 最新一次运行的关键信息
    latest_info = {
        "output_dir": str(output_dir),
        "num_single_pairs": len(single_pairs),
        "num_phrase_pairs": len(phrase_pairs),
        "prev_window": prev_window,
        "threshold_percentile": threshold_percentile,
        "timestamp": datetime.now().isoformat()
    }

    # 更新元数据（历史追加）
    update_metadata_timing(metadata_path, "05_extract_noise_words", current_timing, latest_info)

    # 重新加载并存储常规字段（可选）
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    if '05_extract_noise_words' not in metadata:
        metadata['05_extract_noise_words'] = {}
    metadata['05_extract_noise_words'].update({
        "output_dir": str(output_dir),
        "num_single_pairs": len(single_pairs),
        "num_phrase_pairs": len(phrase_pairs),
        "prev_window": prev_window,
        "threshold_percentile": threshold_percentile,
        "timestamp": datetime.now().isoformat()
    })
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("步骤05完成")

if __name__ == "__main__":
    main()