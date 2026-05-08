#!/usr/bin/env python3
"""
步骤02：筛选高NLL（可疑）句子
- 读取句子级NLL结果CSV
- 按分位数或绝对阈值筛选高NLL样本
- 输出筛选后的CSV，供后续词级分析
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

# 尝试导入自定义日志模块
try:
    from scripts.utils.logger import setup_logger
except ImportError:
    import logging
    def setup_logger(log_dir, name):
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{name}.log"
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        if logger.handlers:
            logger.handlers.clear()
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", type=str, help="全局配置JSON字符串")
    args = parser.parse_args()

    if not args.config_json:
        print("错误：必须提供 --config_json 参数")
        sys.exit(1)

    config = json.loads(args.config_json)
    task_name = config['task_name']
    base_dir = Path(config['paths']['output']['base_dir'])
    task_dir = base_dir / task_name

    step_cfg = config['steps']['02_filter_high_nll']
    input_csv = Path(step_cfg.get('input_csv', 'intermediate/sentence_nll.csv'))
    if not input_csv.is_absolute():
        input_csv = task_dir / input_csv
    output_csv = Path(step_cfg.get('output_csv', 'intermediate/high_nll_sentences.csv'))
    if not output_csv.is_absolute():
        output_csv = task_dir / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    threshold_percentile = step_cfg.get('threshold_percentile', 95)   # 默认取前5%高NLL
    min_sentence_len = step_cfg.get('min_sentence_len', 5)           # 额外过滤短句
    absolute_threshold = step_cfg.get('absolute_threshold', None)    # 绝对阈值（优先级更高）

    # 设置日志
    log_dir = task_dir / "logs"
    logger = setup_logger(log_dir, "02_filter_high_nll")
    logger.info(f"任务目录: {task_dir}")
    logger.info(f"输入CSV: {input_csv}")
    logger.info(f"输出CSV: {output_csv}")
    logger.info(f"阈值百分位: {threshold_percentile} (保留 NLL ≥ 该分位数的句子)")
    if absolute_threshold is not None:
        logger.info(f"绝对阈值: {absolute_threshold} (将覆盖百分位模式)")

    # 检查输入文件
    if not input_csv.exists():
        logger.error(f"输入文件不存在: {input_csv}")
        sys.exit(1)

    # 读取数据
    df = pd.read_csv(input_csv)
    original_len = len(df)
    logger.info(f"加载 {original_len} 条句子")

    # 可选：过滤过短句子（基于字符长度）
    if min_sentence_len > 0:
        df = df[df['sentence'].str.len() >= min_sentence_len]
        logger.info(f"过滤短句后剩余 {len(df)} 条")

    # 确定阈值并筛选
    if absolute_threshold is not None:
        threshold = absolute_threshold
        logger.info(f"使用绝对阈值: {threshold}")
        high_nll_df = df[df['nll'] >= threshold]
    else:
        # 计算分位数对应的阈值
        threshold = df['nll'].quantile(threshold_percentile / 100.0)
        logger.info(f"基于百分位 {threshold_percentile}% 计算阈值: {threshold:.4f}")
        high_nll_df = df[df['nll'] >= threshold]

    logger.info(f"筛选出 {len(high_nll_df)} 条高NLL句子（占比 {len(high_nll_df)/len(df)*100:.2f}%）")

    # 保存结果
    high_nll_df.to_csv(output_csv, index=False, encoding='utf-8')
    logger.info(f"输出已保存至: {output_csv}")

    # 更新元数据
    metadata_path = task_dir / "run_metadata.json"
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
        "threshold_value": float(threshold) if not isinstance(threshold, (int, float)) else threshold,
        "timestamp": datetime.now().isoformat()
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("步骤02完成")


if __name__ == "__main__":
    main()