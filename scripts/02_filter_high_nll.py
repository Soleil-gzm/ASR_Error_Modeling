#!/usr/bin/env python3
"""
步骤02：筛选高NLL句子
支持从元数据动态获取步骤01的输出，并输出带采样比例的文件名
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
from scripts.utils import setup_logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", type=str, required=True)
    args = parser.parse_args()

    config = json.loads(args.config_json)
    task_name = config['task_name']
    base_dir = Path(config['paths']['output']['base_dir'])
    task_dir = base_dir / task_name

    step_cfg = config['steps']['02_filter_high_nll']

    # 从元数据获取步骤01的输出文件路径（自动适配采样）
    metadata_path = task_dir / "run_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        default_input = Path(metadata['01_compute_sentence_nll']['output_csv'])
        # 获取采样比例（如果有），用于输出文件名后缀
        sample_ratio = metadata['01_compute_sentence_nll'].get('sample_ratio', 1.0)
    else:
        default_input = task_dir / "intermediate/sentence_nll.csv"
        sample_ratio = 1.0

    input_csv = Path(step_cfg.get('input_csv', default_input))
    if not input_csv.is_absolute():
        input_csv = task_dir / input_csv

    output_csv = Path(step_cfg.get('output_csv', 'intermediate/high_nll_sentences.csv'))
    if not output_csv.is_absolute():
        output_csv = task_dir / output_csv

    # 如果采样比例 < 1.0，输出文件名添加采样比例后缀
    if sample_ratio < 1.0:
        stem = output_csv.stem
        suffix = output_csv.suffix
        output_csv = output_csv.with_name(f"{stem}_sample_{int(sample_ratio*100)}{suffix}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    threshold_percentile = step_cfg.get('threshold_percentile', 95)
    min_sentence_len = step_cfg.get('min_sentence_len', 5)
    absolute_threshold = step_cfg.get('absolute_threshold', None)

    log_dir = task_dir / "logs"
    logger = setup_logger(log_dir, "02_filter_high_nll")
    logger.info(f"输入CSV: {input_csv}")
    logger.info(f"输出CSV: {output_csv}")

    if not input_csv.exists():
        logger.error(f"输入文件不存在: {input_csv}")
        sys.exit(1)

    df = pd.read_csv(input_csv)
    original_len = len(df)
    logger.info(f"加载 {original_len} 条句子")

    if min_sentence_len > 0:
        df = df[df['sentence'].str.len() >= min_sentence_len]
        logger.info(f"过滤短句后剩余 {len(df)} 条")

    if absolute_threshold is not None:
        threshold = absolute_threshold
        logger.info(f"使用绝对阈值: {threshold}")
        high_nll_df = df[df['nll'] >= threshold]
    else:
        threshold = df['nll'].quantile(threshold_percentile / 100.0)
        logger.info(f"百分位 {threshold_percentile}% → 阈值 {threshold:.4f}")
        high_nll_df = df[df['nll'] >= threshold]

    logger.info(f"筛选出 {len(high_nll_df)} 条高NLL句子 ({len(high_nll_df)/len(df)*100:.2f}%)")
    high_nll_df.to_csv(output_csv, index=False, encoding='utf-8')

    # 更新元数据（记录本步骤输出）
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    metadata['02_filter_high_nll'] = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "threshold_percentile": threshold_percentile,
        "absolute_threshold": absolute_threshold,
        "min_sentence_len": min_sentence_len,
        "num_input_sentences": original_len,
        "num_filtered": len(high_nll_df),
        "threshold_value": float(threshold),
        "sample_ratio": sample_ratio,
        "timestamp": datetime.now().isoformat()
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("步骤02完成")

if __name__ == "__main__":
    main()