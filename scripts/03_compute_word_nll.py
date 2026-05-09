#!/usr/bin/env python3
"""
步骤03：对高NLL句子进行词级NLL计算
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm
# 将项目根目录加入 sys.path
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
    input_csv = Path(step_cfg.get('input_csv', 'intermediate/high_nll_sentences.csv'))
    if not input_csv.is_absolute():
        input_csv = task_dir / input_csv
    output_csv = Path(step_cfg.get('output_csv', 'outputs/word_nll_details.csv'))
    if not output_csv.is_absolute():
        output_csv = task_dir / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # ========== 新增：读取 max_seq_len ==========
    max_seq_len = step_cfg.get('max_seq_len', 512)   # 从配置读取，默认512

    model_name = step_cfg.get('model_name', 'uer/gpt2-chinese-cluecorpussmall')
    gpu_ids = step_cfg.get('gpu_ids', [6, 7])

    log_dir = task_dir / "logs"
    logger = setup_logger(log_dir, "03_compute_word_nll")
    logger.info(f"输入CSV: {input_csv}")
    logger.info(f"输出CSV: {output_csv}")

    if not input_csv.exists():
        logger.error(f"输入文件不存在: {input_csv}")
        sys.exit(1)

    df = pd.read_csv(input_csv, encoding='utf-8')
    sentences = df['sentence'].tolist()
    sentence_ids = df['id'].tolist()
    logger.info(f"共加载 {len(sentences)} 条高NLL句子")

    model, tokenizer, device = load_model_and_tokenizer(model_name, device_ids=gpu_ids)
    model.eval()

    all_records = []
    for sid, sent in tqdm(zip(sentence_ids, sentences), total=len(sentences), desc="逐词分析"):
        # ========== 修改调用，传入 max_length ==========
        tokens, nlls = compute_word_nll(model, tokenizer, sent, device, max_length=max_seq_len)
        # tokens, nlls = compute_word_nll(model, tokenizer, sent, device)
        for idx, (token, nll) in enumerate(zip(tokens, nlls)):
            if pd.isna(nll):
                continue
            # 如果 token 是 bytes，尝试用 utf-8 解码
            if isinstance(token, bytes):
                token = token.decode('utf-8', errors='replace')
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

    # 元数据
    metadata_path = task_dir / "run_metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    metadata['03_compute_word_nll'] = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "model_name": model_name,
        "gpu_ids": gpu_ids,
        "num_sentences": len(sentences),
        "num_tokens": len(all_records),
        "timestamp": datetime.now().isoformat()
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("步骤03完成")

if __name__ == "__main__":
    main()