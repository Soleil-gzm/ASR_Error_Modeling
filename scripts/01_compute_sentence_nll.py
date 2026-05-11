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
    """从预计算的张量直接计算句子级平均 NLL，避免重新 tokenization"""
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
    """保存一个分块的缓存"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'input_ids': input_ids.cpu(),
        'attention_mask': attention_mask.cpu(),
        'ids': ids
    }, cache_dir / f"chunk_{chunk_idx:04d}.pt")

def load_chunk_cache(cache_dir, chunk_idx):
    """加载一个分块的缓存"""
    data = torch.load(cache_dir / f"chunk_{chunk_idx:04d}.pt", map_location='cpu')
    return data['input_ids'], data['attention_mask'], data['ids']

def exists_cache(cache_dir, expected_chunks=None):
    """检查缓存是否存在（至少有一个分块）"""
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

    # 输入输出路径
    input_csv = Path(step_cfg.get('input_csv', 'intermediate/all_sentences.csv'))
    if not input_csv.is_absolute():
        input_csv = task_dir / input_csv
    output_csv = Path(step_cfg.get('output_csv', 'intermediate/sentence_nll.csv'))
    if not output_csv.is_absolute():
        output_csv = task_dir / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # 模型参数
    model_name = step_cfg.get('model_name', 'uer/gpt2-chinese-cluecorpussmall')
    batch_size = step_cfg.get('batch_size', 64)
    max_seq_len = step_cfg.get('max_seq_len', 512)
    gpu_ids = step_cfg.get('gpu_ids', [6, 7])
    chunk_size = step_cfg.get('chunk_size', 50000)
    num_workers = step_cfg.get('num_workers', 4)

    # 采样参数
    sample_ratio = step_cfg.get('sample_ratio', 1.0)
    sample_seed = step_cfg.get('sample_seed', 42)

    # 日志
    log_dir = task_dir / "logs"
    logger = setup_logger(log_dir, "01_compute_sentence_nll")
    logger.info(f"输入CSV: {input_csv}")
    logger.info(f"输出CSV: {output_csv}")

    if not input_csv.exists():
        logger.error(f"输入文件不存在: {input_csv}")
        sys.exit(1)

    # 读取CSV（仅为了获取总行数和句子内容，但为了内存，我们采用分块处理）
    # 注意：缓存需要知道所有句子的id和内容，但我们可以通过分块读取CSV同时生成缓存
    # 这里先确定总行数（可选，用于预估）
    total_lines = sum(1 for _ in open(input_csv)) - 1  # 减去标题行
    logger.info(f"原始句子总数: {total_lines}")

    # 构建缓存目录（基于模型名和max_seq_len，因为padding长度影响张量形状）
    cache_dir = task_dir / "intermediate" / f"tokenized_cache_{model_name.replace('/', '_')}_{max_seq_len}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 是否使用缓存？只要缓存存在且采样比例不影响形状（全量缓存存在）
    # 采样时我们需要从缓存中抽取子集，因此只要全量缓存存在，就可复用
    full_cache_exists = exists_cache(cache_dir)

    # 处理采样逻辑：如果采样比例<1.0且全量缓存存在，从缓存中采样；否则重新tokenization
    if full_cache_exists:
        logger.info("发现全量缓存，将直接使用缓存（跳过tokenization）")
        # 加载所有分块并合并（采样时需随机抽取）
        all_input_ids = []
        all_attention_masks = []
        all_ids = []
        chunk_files = sorted(cache_dir.glob("chunk_*.pt"))
        for ch_file in tqdm(chunk_files, desc="加载缓存分块"):
            input_ids, attention_mask, ids = load_chunk_cache(cache_dir, int(ch_file.stem.split('_')[1]))
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_ids.extend(ids)
        # 合并为单个张量（注意内存，如果太大可逐块采样，但先合并简化）
        input_ids_all = torch.cat(all_input_ids, dim=0)
        attention_mask_all = torch.cat(all_attention_masks, dim=0)
        total = input_ids_all.size(0)

        if sample_ratio < 1.0:
            # 采样
            np.random.seed(sample_seed)
            indices = np.random.choice(total, size=int(total * sample_ratio), replace=False)
            indices.sort()
            input_ids_all = input_ids_all[indices]
            attention_mask_all = attention_mask_all[indices]
            # 注意：ids 也需要采样，但这里我们只保留 sampled 的 ids
            sampled_ids = [all_ids[i] for i in indices]
            logger.info(f"采样后句子数: {len(sampled_ids)}")
            # 重新生成 DataFrame（仅用于保存结果，实际推理使用张量）
            # 但由于我们保留了 sampled_ids，可以在计算NLL后重建df
        else:
            sampled_ids = all_ids
            logger.info(f"使用全量数据: {total} 条句子")

        # 直接使用张量计算NLL
        logger.info("开始计算NLL（使用缓存张量）")
        nll_scores = compute_nll_from_tensors(
            model, input_ids_all, attention_mask_all, batch_size, device, desc="计算句子NLL"
        )
        # 构建结果DataFrame
        df = pd.DataFrame({'id': sampled_ids, 'sentence': [""]*len(sampled_ids), 'nll': nll_scores})
        # 注意：句子原文未保存，需要额外加载CSV填充，但为了简单，可以提前在读取CSV时保留句子内容
        # 这里简化：重新读取CSV，根据id获取sentence（效率低，但采样后句子少，可接受）
        if sample_ratio < 1.0:
            # 重新读取CSV，快速建立 id->sentence 映射
            full_df = pd.read_csv(input_csv)
            id_to_sent = dict(zip(full_df['id'], full_df['sentence']))
            df['sentence'] = df['id'].map(id_to_sent)
        else:
            # 全量时可以直接从缓存中的ids无法直接得到sentence，同样需要读取原CSV
            full_df = pd.read_csv(input_csv)
            id_to_sent = dict(zip(full_df['id'], full_df['sentence']))
            df['sentence'] = df['id'].map(id_to_sent)

    else:
        logger.info("未发现缓存，将进行tokenization并创建缓存")
        # 需要加载模型（因为后续计算需要）
        model, tokenizer, device = load_model_and_tokenizer(model_name, device_ids=gpu_ids)
        logger.info(f"模型加载完成，设备: {device}")

        # 分块读取CSV，tokenization，同时保存缓存
        csv_iter = pd.read_csv(input_csv, chunksize=chunk_size)
        chunk_idx = 0
        all_input_ids = []
        all_attention_masks = []
        all_ids = []  # 收集所有id，用于最终df
        for chunk in tqdm(csv_iter, desc="处理分块"):
            sentences = chunk['sentence'].tolist()
            ids = chunk['id'].tolist()
            # Tokenization
            enc = tokenizer(sentences, truncation=True, max_length=max_seq_len,
                            padding='max_length', return_tensors='pt')
            input_ids = enc['input_ids']
            attention_mask = enc['attention_mask']
            # 保存缓存
            save_chunk_cache(cache_dir, chunk_idx, input_ids, attention_mask, ids)
            # 收集用于后续计算（如果采样比例<1.0，可能不需要全量，但为了简单，先全量收集再采样）
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_ids.extend(ids)
            chunk_idx += 1

        # 合并所有分块（注意内存）
        input_ids_all = torch.cat(all_input_ids, dim=0)
        attention_mask_all = torch.cat(all_attention_masks, dim=0)
        total = input_ids_all.size(0)
        logger.info(f"tokenization完成，总句子数: {total}")

        # 采样（如果需要）
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

        # 计算NLL
        nll_scores = compute_nll_from_tensors(
            model, input_ids_all, attention_mask_all, batch_size, device, desc="计算句子NLL"
        )

        # 构建结果DataFrame
        if sample_ratio < 1.0 or sample_ratio == 1.0:
            # 需要句子内容，重新读取CSV构建映射（或提前保存）
            full_df = pd.read_csv(input_csv)
            id_to_sent = dict(zip(full_df['id'], full_df['sentence']))
            df = pd.DataFrame({'id': sampled_ids, 'sentence': [id_to_sent[i] for i in sampled_ids], 'nll': nll_scores})

    # 保存结果CSV
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
        "num_workers": num_workers,
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