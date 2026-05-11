#!/usr/bin/env python3
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
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

    # --- 输入输出路径 ---
    input_csv = Path(step_cfg.get('input_csv', 'intermediate/all_sentences.csv'))
    if not input_csv.is_absolute():
        input_csv = task_dir / input_csv
    output_csv = Path(step_cfg.get('output_csv', 'intermediate/sentence_nll.csv'))
    if not output_csv.is_absolute():
        output_csv = task_dir / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # --- 模型参数 ---
    model_name = step_cfg.get('model_name', 'uer/gpt2-chinese-cluecorpussmall')
    batch_size = step_cfg.get('batch_size', 64)
    max_seq_len = step_cfg.get('max_seq_len', 512)
    gpu_ids = step_cfg.get('gpu_ids', [6, 7])
    chunk_size = step_cfg.get('chunk_size', 50000)
    num_workers = step_cfg.get('num_workers', 4)

    # --- 采样参数 ---
    sample_ratio = step_cfg.get('sample_ratio', 1.0)
    sample_seed = step_cfg.get('sample_seed', 42)

    # --- 日志 ---
    log_dir = task_dir / "logs"
    logger = setup_logger(log_dir, "01_compute_sentence_nll")
    logger.info(f"输入CSV: {input_csv}")
    logger.info(f"输出CSV: {output_csv}")

    # --- 读取全量数据 ---
    if not input_csv.exists():
        logger.error(f"输入文件不存在: {input_csv}")
        sys.exit(1)

    df = pd.read_csv(input_csv)
    total_original = len(df)
    logger.info(f"原始句子数: {total_original}")

    # --- 采样 ---
    if sample_ratio < 1.0:
        logger.info(f"随机采样 {sample_ratio*100:.1f}% 数据 (种子={sample_seed})")
        df = df.sample(frac=sample_ratio, random_state=sample_seed).reset_index(drop=True)
        # 修改输出文件名，添加采样比例标识
        stem = output_csv.stem
        suffix = output_csv.suffix
        output_csv = output_csv.with_name(f"{stem}_sample_{int(sample_ratio*100)}{suffix}")
        logger.info(f"采样后句子数: {len(df)}")
        logger.info(f"输出文件名调整为: {output_csv}")

    sentences = df['sentence'].tolist()
    logger.info(f"实际处理句子数: {len(sentences)}")

    # --- 加载模型 ---
    model, tokenizer, device = load_model_and_tokenizer(model_name, device_ids=gpu_ids)
    logger.info(f"模型加载完成，设备: {device}")

    # --- 分块计算 NLL（支持采样后的数据）---
    nll_scores = []
    # 这里我们直接使用 compute_sentence_nll_batch 函数，它内部会处理分块
    # 但为了与分块逻辑兼容，我们将整个 sentences 传给该函数（它内部会创建 Dataset 和 DataLoader）
    nll_scores = compute_sentence_nll_batch(
        model, tokenizer, sentences,
        batch_size=batch_size, max_length=max_seq_len, device=device,
        desc="计算句子NLL",
        num_workers=num_workers
    )

    df['nll'] = nll_scores
    df.to_csv(output_csv, index=False, encoding='utf-8')
    logger.info(f"结果已保存至 {output_csv}")

    # --- 更新元数据（记录采样信息）---
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
        "chunk_size": chunk_size,
        "num_workers": num_workers,
        "sample_ratio": sample_ratio,
        "sample_seed": sample_seed,
        "num_sentences": len(sentences),
        "timestamp": datetime.now().isoformat()
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("步骤01完成")

if __name__ == "__main__":
    main()