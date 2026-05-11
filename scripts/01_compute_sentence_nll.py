#!/usr/bin/env python3
"""
步骤01：计算句子级平均NLL（GPU批处理） - 支持缓存 tokenization 结果
"""

import sys
import json
import argparse
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
    """从预计算的张量直接计算句子级平均 NLL"""
    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    model.eval()
    all_nll = []
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
    return all_nll

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

    # 先加载模型（无论是否使用缓存，都需要模型）
    logger.info("加载模型...")
    model, tokenizer, device = load_model_and_tokenizer(model_name, device_ids=gpu_ids)
    logger.info(f"模型加载完成，设备: {device}")

    # 构建缓存目录
    cache_dir = task_dir / "intermediate" / f"tokenized_cache_{model_name.replace('/', '_')}_{max_seq_len}"
    full_cache_exists = exists_cache(cache_dir)

    if full_cache_exists:
        logger.info("发现全量缓存，将直接使用缓存（跳过tokenization）")
        # 加载所有分块
        all_input_ids = []
        all_attention_masks = []
        all_ids = []
        chunk_files = sorted(cache_dir.glob("chunk_*.pt"))
        for ch_file in tqdm(chunk_files, desc="加载缓存分块"):
            idx = int(ch_file.stem.split('_')[1])
            input_ids, attn_mask, ids = load_chunk_cache(cache_dir, idx)
            all_input_ids.append(input_ids)
            all_attention_masks.append(attn_mask)
            all_ids.extend(ids)
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

        # 计算 NLL
        nll_scores = compute_nll_from_tensors(model, input_ids_all, attention_mask_all, batch_size, device, desc="计算句子NLL")

        # 重建 DataFrame（需要句子内容）
        full_df = pd.read_csv(input_csv)
        id_to_sent = dict(zip(full_df['id'], full_df['sentence']))
        df = pd.DataFrame({'id': sampled_ids, 'sentence': [id_to_sent[i] for i in sampled_ids], 'nll': nll_scores})

    else:
        logger.info("未发现缓存，将进行tokenization并创建缓存")
        # 分块读取CSV，进行 tokenization 并保存缓存
        csv_iter = pd.read_csv(input_csv, chunksize=chunk_size)
        all_input_ids = []
        all_attention_masks = []
        all_ids = []
        chunk_idx = 0
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

        # 计算 NLL
        nll_scores = compute_nll_from_tensors(model, input_ids_all, attention_mask_all, batch_size, device, desc="计算句子NLL")

        full_df = pd.read_csv(input_csv)
        id_to_sent = dict(zip(full_df['id'], full_df['sentence']))
        df = pd.DataFrame({'id': sampled_ids, 'sentence': [id_to_sent[i] for i in sampled_ids], 'nll': nll_scores})

    # 保存最终 CSV
    if sample_ratio < 1.0:
        stem = output_csv.stem
        output_csv = output_csv.with_name(f"{stem}_sample_{int(sample_ratio*100)}{output_csv.suffix}")
    df.to_csv(output_csv, index=False, encoding='utf-8')
    logger.info(f"结果已保存至 {output_csv}")

    # 更新元数据
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
        "sample_ratio": sample_ratio,
        "sample_seed": sample_seed,
        "num_sentences": len(df),
        "timestamp": datetime.now().isoformat(),
        "used_cache": full_cache_exists
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("步骤01完成")

if __name__ == "__main__":
    main()