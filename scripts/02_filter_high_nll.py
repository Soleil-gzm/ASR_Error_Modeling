#!/usr/bin/env python3
"""
步骤02：筛选高NLL句子（带计时和历史记录）
完全依赖元数据确定输入文件路径，自动适配采样
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
from scripts.utils.timer import TimedBlock, update_metadata_timing

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", type=str, required=True)
    args = parser.parse_args()

    config = json.loads(args.config_json)
    task_name = config['task_name']
    base_dir = Path(config['paths']['output']['base_dir'])
    task_dir = base_dir / task_name

    step_cfg = config['steps']['02_filter_high_nll']

    # 日志设置
    log_dir = task_dir / "logs"
    logger = setup_logger(log_dir, "02_filter_high_nll")

    # 从元数据获取步骤01的输出文件路径
    metadata_path = task_dir / "run_metadata.json"
    if not metadata_path.exists():
        logger.error(f"元数据文件不存在: {metadata_path}，请先运行步骤01")
        sys.exit(1)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    if '01_compute_sentence_nll' not in metadata:
        logger.error("元数据中缺少步骤01的记录，请重新运行步骤01")
        sys.exit(1)

    # 步骤01的输出路径（元数据中存的是相对于项目根目录的路径）
    input_csv_rel = Path(metadata['01_compute_sentence_nll']['output_csv'])
    project_root = base_dir.parent
    if input_csv_rel.is_absolute():
        input_csv = input_csv_rel
    else:
        input_csv = project_root / input_csv_rel

    sample_ratio = metadata['01_compute_sentence_nll'].get('sample_ratio', 1.0)

    # 输出文件路径：基于 task_dir 构建
    output_csv = Path(step_cfg.get('output_csv', 'intermediate/high_nll_sentences.csv'))
    if not output_csv.is_absolute():
        output_csv = task_dir / output_csv

    if sample_ratio < 1.0:
        stem = output_csv.stem
        output_csv = output_csv.with_name(f"{stem}_sample_{int(sample_ratio*100)}{output_csv.suffix}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # 读取参数
    threshold_percentile = step_cfg.get('threshold_percentile', 95)
    min_sentence_len = step_cfg.get('min_sentence_len', 5)
    absolute_threshold = step_cfg.get('absolute_threshold', None)

    logger.info(f"输入CSV: {input_csv}")
    logger.info(f"输出CSV: {output_csv}")
    logger.info(f"采样比例: {sample_ratio}")

    if not input_csv.exists():
        logger.error(f"输入文件不存在: {input_csv}")
        sys.exit(1)

    total_start = time.perf_counter()
    timing = {}

    # 1. 读取CSV
    with TimedBlock("read_csv", timing):
        df = pd.read_csv(input_csv)
    original_len = len(df)
    logger.info(f"加载 {original_len} 条句子")

    # 2. 过滤短句
    with TimedBlock("filter_short", timing):
        if min_sentence_len > 0:
            df = df[df['sentence'].str.len() >= min_sentence_len]
            logger.info(f"过滤短句后剩余 {len(df)} 条")

    # 3. 根据阈值筛选高NLL句子
    with TimedBlock("filter_nll_threshold", timing):
        if absolute_threshold is not None:
            threshold = absolute_threshold
            logger.info(f"使用绝对阈值: {threshold}")
            high_nll_df = df[df['nll'] >= threshold]
        else:
            threshold = df['nll'].quantile(threshold_percentile / 100.0)
            logger.info(f"百分位 {threshold_percentile}% → 阈值 {threshold:.4f}")
            high_nll_df = df[df['nll'] >= threshold]
    logger.info(f"筛选出 {len(high_nll_df)} 条高NLL句子 ({len(high_nll_df)/len(df)*100:.2f}%)")

    # 4. 写入CSV
    with TimedBlock("write_csv", timing):
        high_nll_df.to_csv(output_csv, index=False, encoding='utf-8')

    timing["total_sec"] = time.perf_counter() - total_start
    logger.info(f"总耗时: {timing['total_sec']:.2f}秒")

    # 构建当前运行的计时记录
    current_timing = {
        "timestamp": datetime.now().isoformat(),
        "read_csv_sec": timing["read_csv"],
        "filter_short_sec": timing["filter_short"],
        "filter_nll_threshold_sec": timing["filter_nll_threshold"],
        "write_csv_sec": timing["write_csv"],
        "total_sec": timing["total_sec"],
        "num_input_sentences": original_len,
        "num_after_short_filter": len(df),
        "num_filtered": len(high_nll_df),
        "threshold_percentile": threshold_percentile,
        "absolute_threshold": absolute_threshold,
        "min_sentence_len": min_sentence_len
    }

    # 最新一次运行的关键信息
    latest_info = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "threshold_percentile": threshold_percentile,
        "absolute_threshold": absolute_threshold,
        "min_sentence_len": min_sentence_len,
        "num_filtered": len(high_nll_df),
        "sample_ratio": sample_ratio,
        "timestamp": datetime.now().isoformat()
    }

    # 更新元数据（历史追加）
    update_metadata_timing(metadata_path, "02_filter_high_nll", current_timing, latest_info)

    # 同时更新原 metadata 中的其他字段（兼容下游步骤读取）
    metadata['02_filter_high_nll'].update({
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
    })
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("步骤02完成")

if __name__ == "__main__":
    main()