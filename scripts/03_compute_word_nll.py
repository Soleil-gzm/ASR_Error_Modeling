#!/usr/bin/env python3
"""
步骤03：对高NLL句子进行词级NLL计算
从元数据获取步骤02的输出，输出文件名自动添加采样后缀
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utils import setup_logger, load_model_and_tokenizer, compute_word_nll

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", type=str, required=True)
    args = parser.parse_args()

    config = json.loads(args.config_json)
    task_name = config['task_name']
    base_dir = Path(config['paths']['output']['base_dir'])
    task_dir = base_dir / task_name

    step_cfg = config['steps']['03_compute_word_nll']

    log_dir = task_dir / "logs"
    logger = setup_logger(log_dir, "03_compute_word_nll")

    # 从元数据获取步骤02的输出路径
    metadata_path = task_dir / "run_metadata.json"
    if not metadata_path.exists():
        logger.error(f"元数据文件不存在: {metadata_path}，请先运行步骤02")
        sys.exit(1)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    if '02_filter_high_nll' not in metadata:
        logger.error("元数据中缺少步骤02的记录，请先运行步骤02")
        sys.exit(1)

    # 步骤02的输出路径（元数据中存的是相对于项目根目录的路径）
    input_csv_rel = Path(metadata['02_filter_high_nll']['output_csv'])
    project_root = base_dir.parent
    if input_csv_rel.is_absolute():
        input_csv = input_csv_rel
    else:
        input_csv = project_root / input_csv_rel

    # 获取采样比例（从步骤01的记录中取）
    sample_ratio = metadata.get('01_compute_sentence_nll', {}).get('sample_ratio', 1.0)

    # 输出文件路径
    output_csv = Path(step_cfg.get('output_csv', 'outputs/word_nll_details.csv'))
    if not output_csv.is_absolute():
        output_csv = task_dir / output_csv

    if sample_ratio < 1.0:
        stem = output_csv.stem
        suffix = output_csv.suffix
        output_csv = output_csv.with_name(f"{stem}_sample_{int(sample_ratio*100)}{suffix}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    max_seq_len = step_cfg.get('max_seq_len', 512)
    model_name = step_cfg.get('model_name', 'uer/gpt2-chinese-cluecorpussmall')
    gpu_ids = step_cfg.get('gpu_ids', [6, 7])

    logger.info(f"输入CSV: {input_csv}")
    logger.info(f"输出CSV: {output_csv}")
    logger.info(f"采样比例: {sample_ratio}")

    if not input_csv.exists():
        logger.error(f"输入文件不存在: {input_csv}")
        sys.exit(1)

    df = pd.read_csv(input_csv)
    sentences = df['sentence'].tolist()
    sentence_ids = df['id'].tolist()
    logger.info(f"共加载 {len(sentences)} 条高NLL句子")

    model, tokenizer, device = load_model_and_tokenizer(model_name, device_ids=gpu_ids)
    model.eval()

    all_records = []
    for sid, sent in tqdm(zip(sentence_ids, sentences), total=len(sentences), desc="逐词分析"):
        tokens, nlls = compute_word_nll(model, tokenizer, sent, device, max_length=max_seq_len)
        for idx, (token, nll) in enumerate(zip(tokens, nlls)):
            if pd.isna(nll):
                continue
            all_records.append({
                "sentence_id": sid,
                "token_index": idx,
                "token": token,
                "nll": nll,
                "sentence": sent
            })

    logger.info(f"共产生 {len(all_records)} 条 token 级记录")
    result_df = pd.DataFrame(all_records)
    result_df.to_csv(output_csv, index=False, encoding='utf-8')
    logger.info(f"已保存至 {output_csv}")

    # 更新元数据
    metadata['03_compute_word_nll'] = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "model_name": model_name,
        "gpu_ids": gpu_ids,
        "max_seq_len": max_seq_len,
        "num_sentences": len(sentences),
        "num_tokens": len(all_records),
        "sample_ratio": sample_ratio,
        "timestamp": datetime.now().isoformat()
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("步骤03完成")

if __name__ == "__main__":
    main()