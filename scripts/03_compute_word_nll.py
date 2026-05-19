#!/usr/bin/env python3
"""
步骤03：对高NLL句子进行词级NLL计算（批处理版本）- 带计时和历史记录
- 支持批量推理，显著提升 GPU 利用率
- 自动 padding 和对齐，返回每个 token 的 NLL
"""

import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utils import setup_logger, load_model_and_tokenizer
from scripts.utils.timer import TimedBlock, update_metadata_timing
from scripts.utils.metadata import get_step_output, get_step_sample_ratio

# ---------- 数据集与collate函数 ----------
class WordNLLDataset(Dataset):
    ''' 包装句子列表和对应的 ID，供 DataLoader 使用。每个样本是 (sentence, id)。 '''
    def __init__(self, sentences, ids):
        self.sentences = sentences
        self.ids = ids
    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, idx):
        return self.sentences[idx], self.ids[idx]

def collate_word_nll(batch, tokenizer, max_length):
    ''' 将一个 batch 的句子进行动态 padding 和 tokenization。由于每个句子长度不同，
        通过 padding=True 填充到本 batch 中最长长度（不超过 max_length），返回统一的张量。 '''
    sentences, ids = zip(*batch)
    enc = tokenizer(
        list(sentences),
        truncation=True,
        max_length=max_length,
        padding=True,  
        return_tensors='pt'
    )
    return enc['input_ids'], enc['attention_mask'], sentences, ids

def compute_word_nll_batch(model, tokenizer, sentences, ids, max_length, batch_size, device):
    dataset = WordNLLDataset(sentences, ids)
    collate_fn = lambda b: collate_word_nll(b, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, pin_memory=True, num_workers=0)
    model.eval()
    result_tokens = {}
    result_nlls = {}
    all_seq_nll = []
    with torch.no_grad():
        for input_ids, attention_mask, orig_sentences, orig_ids in tqdm(dataloader, desc="批处理词级NLL"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            token_nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            token_nll = token_nll.view(shift_labels.size())
            for i in range(input_ids.size(0)):
                real_len = attention_mask[i].sum().item()
                if real_len <= 1:
                    continue
                nll_values = token_nll[i, :real_len-1].cpu().tolist()
                token_ids = input_ids[i, 1:real_len].cpu().tolist()
                token_strs = [tokenizer.decode([tid], skip_special_tokens=False).strip() for tid in token_ids]
                for j, ts in enumerate(token_strs):
                    if not ts:
                        alt = tokenizer.convert_ids_to_tokens(token_ids[j])
                        if isinstance(alt, bytes):
                            alt = alt.decode('utf-8', errors='replace')
                        token_strs[j] = alt
                sid = orig_ids[i]
                result_tokens[sid] = token_strs
                result_nlls[sid] = nll_values
                avg_seq_nll = sum(nll_values) / len(nll_values) if nll_values else float('nan')
                all_seq_nll.append((sid, avg_seq_nll))
    return result_tokens, result_nlls, all_seq_nll

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

    # 从元数据获取步骤02的输出
    metadata_path = task_dir / "run_metadata.json"
    if not metadata_path.exists():
        logger.error(f"元数据文件不存在: {metadata_path}")
        sys.exit(1)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    if '02_filter_high_nll' not in metadata:
        logger.error("缺少步骤02的记录，请先运行步骤02")
        sys.exit(1)

    # 使用辅助函数获取步骤02的输出路径
    output_csv_rel = get_step_output(metadata, '02_filter_high_nll')
    if output_csv_rel is None:
        logger.error("无法找到步骤02的输出文件路径")
        sys.exit(1)
    project_root = base_dir.parent
    input_csv = project_root / Path(output_csv_rel) if not Path(output_csv_rel).is_absolute() else Path(output_csv_rel)

    # 使用辅助函数获取采样比例
    sample_ratio = get_step_sample_ratio(metadata, '01_compute_sentence_nll')

    output_csv = Path(step_cfg.get('output_csv', 'outputs/word_nll_details.csv'))
    if not output_csv.is_absolute():
        output_csv = task_dir / output_csv
    if sample_ratio < 1.0:
        output_csv = output_csv.with_name(f"{output_csv.stem}_sample_{int(sample_ratio*100)}{output_csv.suffix}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    max_seq_len = step_cfg.get('max_seq_len', 512)
    model_name = step_cfg.get('model_name', 'uer/gpt2-chinese-cluecorpussmall')
    gpu_ids = step_cfg.get('gpu_ids', [6, 7])
    batch_size = step_cfg.get('batch_size', 32)

    logger.info(f"输入CSV: {input_csv}")
    logger.info(f"输出CSV: {output_csv}")
    logger.info(f"采样比例: {sample_ratio}, 批大小: {batch_size}")

    if not input_csv.exists():
        logger.error(f"输入文件不存在: {input_csv}")
        sys.exit(1)

    df = pd.read_csv(input_csv)
    sentences = df['sentence'].tolist()
    ids = df['id'].tolist()
    logger.info(f"共加载 {len(sentences)} 条高NLL句子")

    total_start = time.perf_counter()
    timing = {}

    # 1. 加载模型
    with TimedBlock("load_model", timing):
        if len(gpu_ids) > 1:
            logger.warning("步骤03建议使用单卡以避免 DataParallel 开销，将只用第一个 GPU")
        model, tokenizer, device = load_model_and_tokenizer(model_name, device_ids=[gpu_ids[0]])
    logger.info(f"模型加载完成，设备: {device}，耗时: {timing['load_model']:.2f}s")

    # 2. 批量推理
    with TimedBlock("inference", timing):
        token_dict, nll_dict, seq_avg = compute_word_nll_batch(
            model, tokenizer, sentences, ids, max_seq_len, batch_size, device
        )

    # 3. 构建结果并保存
    all_records = []
    for sid in ids:
        if sid not in token_dict:
            continue
        tokens = token_dict[sid]
        nlls = nll_dict[sid]
        sent_text = df[df['id'] == sid]['sentence'].iloc[0]
        for idx, (tok, nll) in enumerate(zip(tokens, nlls)):
            all_records.append({
                'sentence_id': sid,
                'token_index': idx,
                'token': tok,
                'nll': nll,
                'sentence': sent_text
            })
    result_df = pd.DataFrame(all_records)

    with TimedBlock("save_output", timing):
        result_df.to_csv(output_csv, index=False, encoding='utf-8')
        seq_avg_df = pd.DataFrame(seq_avg, columns=['id', 'seq_avg_nll'])
        seq_avg_csv = output_csv.parent / (output_csv.stem + "_seq_avg.csv")
        seq_avg_df.to_csv(seq_avg_csv, index=False)

    timing["total_sec"] = time.perf_counter() - total_start
    logger.info(f"推理耗时: {timing['inference']:.2f}s, 平均每秒处理 {len(sentences)/timing['inference']:.2f} 条句子")
    logger.info(f"保存结果耗时: {timing['save_output']:.2f}s")
    logger.info(f"总耗时: {timing['total_sec']:.2f}s")
    logger.info(f"词级NLL结果已保存至 {output_csv}，共 {len(all_records)} 个 token")

    # 构建当前运行的计时记录
    current_timing = {
        "timestamp": datetime.now().isoformat(),
        "load_model_sec": timing["load_model"],
        "inference_sec": timing["inference"],
        "save_output_sec": timing["save_output"],
        "total_sec": timing["total_sec"],
        "throughput_sentences_per_sec": len(sentences) / timing["inference"] if timing["inference"] > 0 else 0,
        "num_sentences": len(sentences),
        "num_tokens": len(all_records),
        "batch_size": batch_size,
        "max_seq_len": max_seq_len
    }

    latest_info = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "model_name": model_name,
        "gpu_ids": gpu_ids,
        "max_seq_len": max_seq_len,
        "batch_size": batch_size,
        "num_sentences": len(sentences),
        "num_tokens": len(all_records),
        "sample_ratio": sample_ratio,
        "timestamp": datetime.now().isoformat()
    }

    update_metadata_timing(metadata_path, "03_compute_word_nll", current_timing, latest_info)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    if '03_compute_word_nll' not in metadata:
        metadata['03_compute_word_nll'] = {}
    metadata['03_compute_word_nll'].update({
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "model_name": model_name,
        "gpu_ids": gpu_ids,
        "max_seq_len": max_seq_len,
        "batch_size": batch_size,
        "num_sentences": len(sentences),
        "num_tokens": len(all_records),
        "sample_ratio": sample_ratio,
        "timestamp": datetime.now().isoformat()
    })

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("步骤03完成")

if __name__ == "__main__":
    main()