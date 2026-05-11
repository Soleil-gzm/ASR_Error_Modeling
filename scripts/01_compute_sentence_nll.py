#!/usr/bin/env python3
"""
步骤01：计算句子级平均NLL（GPU批处理） - 支持缓存 tokenization 结果，计时历史记录
"""

import sys
import json
import argparse
import time
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utils import setup_logger, load_model_and_tokenizer

def compute_nll_from_tensors(model, input_ids, attention_mask, batch_size, device, desc="推理"):
    """从预计算的张量直接计算句子级平均 NLL，返回 NLL 列表和推理耗时"""
    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    model.eval()
    all_nll = []
    start_time = time.perf_counter()
    with torch.no_grad():
        for batch_input_ids, batch_mask in tqdm(dataloader, desc=desc):
            batch_input_ids = batch_input_ids.to(device)
            batch_mask = batch_mask.to(device)
            outputs = model(batch_input_ids, attention_mask=batch_mask, labels=batch_input_ids)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch_input_ids[:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            token_nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            token_nll = token_nll.view(shift_labels.size())
            mask = batch_mask[:, 1:].contiguous()
            seq_nll = (token_nll * mask).sum(dim=1) / mask.sum(dim=1)
            all_nll.extend(seq_nll.cpu().tolist())
    inference_time = time.perf_counter() - start_time
    return all_nll, inference_time

def save_chunk_cache(cache_dir, chunk_idx, input_ids, attention_mask, ids):
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'input_ids': input_ids.cpu(),
        'attention_mask': attention_mask.cpu(),
        'ids': ids
    }, cache_dir / f"chunk_{chunk_idx:04d}.pt")

def load_chunk_cache(cache_dir, chunk_idx):
    data = torch.load(cache_dir / f"chunk_{chunk_idx:04d}.pt", map_location='cpu')
    return data['input_ids'], data['attention_mask'], data['ids']

def exists_cache(cache_dir, expected_chunks=None):
    chunks = list(cache_dir.glob("chunk_*.pt"))
    if not chunks:
        return False
    if expected_chunks is not None and len(chunks) != expected_chunks:
        return False
    return True

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
    chunk_size = step_cfg.get('chunk_size', 50000)
    sample_ratio = step_cfg.get('sample_ratio', 1.0)
    sample_seed = step_cfg.get('sample_seed', 42)

    log_dir = task_dir / "logs"
    logger = setup_logger(log_dir, "01_compute_sentence_nll")
    logger.info(f"输入CSV: {input_csv}")
    logger.info(f"输出CSV: {output_csv}")

    if not input_csv.exists():
        logger.error(f"输入文件不存在: {input_csv}")
        sys.exit(1)

    total_start = time.perf_counter()

    # 1. 模型加载
    model_load_start = time.perf_counter()
    logger.info("加载模型...")
    model, tokenizer, device = load_model_and_tokenizer(model_name, device_ids=gpu_ids)
    model_load_time = time.perf_counter() - model_load_start
    logger.info(f"模型加载完成，设备: {device}，耗时: {model_load_time:.2f}s")

    # 缓存目录
    cache_dir = task_dir / "intermediate" / f"tokenized_cache_{model_name.replace('/', '_')}_{max_seq_len}"
    full_cache_exists = exists_cache(cache_dir)

    data_prep_start = time.perf_counter()

    if full_cache_exists:
        logger.info("发现全量缓存，将直接使用缓存（跳过tokenization）")
        all_input_ids = []
        all_attention_masks = []
        all_ids = []
        chunk_files = sorted(cache_dir.glob("chunk_*.pt"))
        load_cache_start = time.perf_counter()
        for ch_file in tqdm(chunk_files, desc="加载缓存分块"):
            idx = int(ch_file.stem.split('_')[1])
            input_ids, attn_mask, ids = load_chunk_cache(cache_dir, idx)
            all_input_ids.append(input_ids)
            all_attention_masks.append(attn_mask)
            all_ids.extend(ids)
        load_cache_time = time.perf_counter() - load_cache_start
        input_ids_all = torch.cat(all_input_ids, dim=0)
        attention_mask_all = torch.cat(all_attention_masks, dim=0)
        total = input_ids_all.size(0)

        if sample_ratio < 1.0:
            np.random.seed(sample_seed)
            indices = np.random.choice(total, size=int(total * sample_ratio), replace=False)
            indices.sort()
            input_ids_all = input_ids_all[indices]
            attention_mask_all = attention_mask_all[indices]
            sampled_ids = [all_ids[i] for i in indices]
            logger.info(f"采样后句子数: {len(sampled_ids)}")
        else:
            sampled_ids = all_ids
            logger.info(f"使用全量数据: {total} 条句子")
        data_prep_time = time.perf_counter() - data_prep_start
        logger.info(f"缓存加载+采样耗时: {data_prep_time:.2f}s (其中加载缓存 {load_cache_time:.2f}s)")
        tokenize_time = None
    else:
        logger.info("未发现缓存，将进行tokenization并创建缓存")
        csv_iter = pd.read_csv(input_csv, chunksize=chunk_size)
        all_input_ids = []
        all_attention_masks = []
        all_ids = []
        chunk_idx = 0
        tokenize_start = time.perf_counter()
        for chunk in tqdm(csv_iter, desc="Tokenizing and caching"):
            sentences = chunk['sentence'].tolist()
            ids = chunk['id'].tolist()
            enc = tokenizer(sentences, truncation=True, max_length=max_seq_len,
                            padding='max_length', return_tensors='pt')
            input_ids = enc['input_ids']
            attention_mask = enc['attention_mask']
            save_chunk_cache(cache_dir, chunk_idx, input_ids, attention_mask, ids)
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_ids.extend(ids)
            chunk_idx += 1
        tokenize_time = time.perf_counter() - tokenize_start
        logger.info(f"Tokenization 耗时: {tokenize_time:.2f}s")

        input_ids_all = torch.cat(all_input_ids, dim=0)
        attention_mask_all = torch.cat(all_attention_masks, dim=0)
        total = input_ids_all.size(0)
        logger.info(f"tokenization完成，总句子数: {total}")

        if sample_ratio < 1.0:
            np.random.seed(sample_seed)
            indices = np.random.choice(total, size=int(total * sample_ratio), replace=False)
            indices.sort()
            input_ids_all = input_ids_all[indices]
            attention_mask_all = attention_mask_all[indices]
            sampled_ids = [all_ids[i] for i in indices]
            logger.info(f"采样后句子数: {len(sampled_ids)}")
        else:
            sampled_ids = all_ids
        data_prep_time = time.perf_counter() - data_prep_start
        logger.info(f"数据准备总耗时: {data_prep_time:.2f}s (其中tokenization {tokenize_time:.2f}s)")

    # 2. 推理
    inference_start = time.perf_counter()
    nll_scores, inference_time = compute_nll_from_tensors(
        model, input_ids_all, attention_mask_all, batch_size, device, desc="计算句子NLL"
    )
    logger.info(f"推理耗时: {inference_time:.2f}s, 平均每秒处理 {len(sampled_ids)/inference_time:.2f} 条句子")

    # 3. 构建结果 DataFrame
    full_df = pd.read_csv(input_csv)
    id_to_sent = dict(zip(full_df['id'], full_df['sentence']))
    df = pd.DataFrame({'id': sampled_ids, 'sentence': [id_to_sent[i] for i in sampled_ids], 'nll': nll_scores})

    if sample_ratio < 1.0:
        stem = output_csv.stem
        output_csv = output_csv.with_name(f"{stem}_sample_{int(sample_ratio*100)}{output_csv.suffix}")
    save_start = time.perf_counter()
    df.to_csv(output_csv, index=False, encoding='utf-8')
    save_time = time.perf_counter() - save_start
    logger.info(f"结果保存至 {output_csv}，耗时 {save_time:.2f}s")

    total_time = time.perf_counter() - total_start
    logger.info(f"总耗时: {total_time:.2f}s")

    # 准备计时信息
    current_timing = {
        "timestamp": datetime.now().isoformat(),
        "load_model_sec": model_load_time,
        "data_preparation_sec": data_prep_time,
        "inference_sec": inference_time,
        "save_output_sec": save_time,
        "total_sec": total_time,
        "throughput_sentences_per_sec": len(sampled_ids) / inference_time if inference_time > 0 else 0,
        "tokenization_sec": tokenize_time if tokenize_time is not None else None
    }

    # 更新元数据（历史追加模式）
    metadata_path = task_dir / "run_metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

    # 确保结构存在
    if "01_compute_sentence_nll" not in metadata:
        metadata["01_compute_sentence_nll"] = {}
    if "timing_history" not in metadata["01_compute_sentence_nll"]:
        metadata["01_compute_sentence_nll"]["timing_history"] = []
    metadata["01_compute_sentence_nll"]["timing_history"].append(current_timing)

    # 保存最新运行的关键配置（可选）
    metadata["01_compute_sentence_nll"]["latest"] = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "model_name": model_name,
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
        "gpu_ids": gpu_ids,
        "chunk_size": chunk_size,
        "sample_ratio": sample_ratio,
        "sample_seed": sample_seed,
        "num_sentences": len(df),
        "used_cache": full_cache_exists,
        "timestamp": datetime.now().isoformat()
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("步骤01完成")

if __name__ == "__main__":
    main()