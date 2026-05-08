#!/usr/bin/env python3
"""
步骤01：计算句子级平均NLL（GPU批处理）
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
from scripts.utils import setup_logger, load_model_and_tokenizer, compute_sentence_nll_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", type=str, required=True)
    args = parser.parse_args()

    config = json.loads(args.config_json)
    task_name = config['task_name']
    base_dir = Path(config['paths']['output']['base_dir'])
    task_dir = base_dir / task_name

    step_cfg = config['steps']['01_compute_sentence_nll']
    input_csv = Path(step_cfg.get('input_csv', 'intermediate/all_sentences.csv'))
    if not input_csv.is_absolute():
        input_csv = task_dir / input_csv
    output_csv = Path(step_cfg.get('output_csv', 'intermediate/sentence_nll.csv'))
    if not output_csv.is_absolute():
        output_csv = task_dir / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    model_name = step_cfg.get('model_name', 'uer/gpt2-chinese-cluecorpussmall')
    batch_size = step_cfg.get('batch_size', 64)
    max_seq_len = step_cfg.get('max_seq_len', 512)
    gpu_ids = step_cfg.get('gpu_ids', [6, 7])

    log_dir = task_dir / "logs"
    logger = setup_logger(log_dir, "01_compute_sentence_nll")
    logger.info(f"输入CSV: {input_csv}")
    logger.info(f"输出CSV: {output_csv}")

    # 加载句子
    if not input_csv.exists():
        logger.error(f"输入文件不存在: {input_csv}")
        sys.exit(1)
    df = pd.read_csv(input_csv)
    sentences = df['sentence'].tolist()
    logger.info(f"共加载 {len(sentences)} 条句子")

    # 加载模型
    model, tokenizer, device = load_model_and_tokenizer(model_name, device_ids=gpu_ids)
    logger.info(f"模型加载完成，设备: {device}")

    # 计算NLL
    nll_scores = compute_sentence_nll_batch(
        model, tokenizer, sentences,
        batch_size=batch_size, max_length=max_seq_len, device=device,
        desc="计算句子NLL"
    )

    df['nll'] = nll_scores
    df.to_csv(output_csv, index=False, encoding='utf-8')
    logger.info(f"结果已保存至 {output_csv}")

    # 元数据
    metadata_path = task_dir / "run_metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    metadata['01_compute_sentence_nll'] = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "model_name": model_name,
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
        "gpu_ids": gpu_ids,
        "num_sentences": len(sentences),
        "timestamp": datetime.now().isoformat()
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("步骤01完成")

if __name__ == "__main__":
    main()