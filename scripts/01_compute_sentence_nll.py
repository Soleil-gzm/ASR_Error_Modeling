#!/usr/bin/env python3
"""
步骤01：计算句子级平均NLL（GPU批处理） - 缓存 tokenization，历史计时
集成缓存健康检查，自动失效和重建
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
from scripts.utils.timer import TimedBlock, update_metadata_timing
from scripts.utils.cache import is_cache_valid, write_cache_meta, invalidate_cache

# ---------- 辅助函数 ----------
def compute_nll_from_tensors(model, input_ids, attention_mask, batch_size, device, desc="推理"):
    """从预计算的张量直接计算句子级平均 NLL，返回 NLL 列表和推理耗时"""

    ''' num_workers=0  默认为0
        这里的 dataset 是 TensorDataset，里面已经存放了准备好的 input_ids 和 attention_mask（都是 PyTorch 张量）。
        从内存张量中加载数据到 GPU 是非常快的操作（仅仅是切片和复制），几乎没有 CPU 开销，因此：
        用多进程（num_workers>0）并不会带来性能提升，反而会引入额外的进程间通信开销、增加内存占用，
        并且可能因为子进程继承 CUDA 上下文而引发错误（尤其在 GPU 环境下，通常要求主进程创建 CUDA 张量，子进程不应再初始化 CUDA）。 
    '''
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
    ''' 将一批 tokenization 结果（张量）和对应的原始句子 ID 保存到 .pt 文件，文件名如 chunk_0000.pt。
        用于分块缓存，避免一次将全部张量加载到内存。 '''
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'input_ids': input_ids.cpu(),
        'attention_mask': attention_mask.cpu(),
        'ids': ids
    }, cache_dir / f"chunk_{chunk_idx:04d}.pt")

def load_chunk_cache(cache_dir, chunk_idx):
    ''' 加载之前保存的缓存分块，返回 (input_ids, attention_mask, ids)，张量在 CPU 上（便于后续拼接）。 
    
        为什么张量放在CPU上？
        A：节省 GPU 显存，先放在 CPU 上（系统内存），拼接成完整张量后，在推理阶段再按 batch 移到 GPU，可以显著降低显存峰值占用。
           便于拼接操作，CPU 内存通常远大于 GPU 显存，可以容纳整个数据集。若在 GPU 上拼接，会额外产生显存分配和拷贝开销，且容易因显存不足而失败。
           延迟传输到 GPU 的常见模式，在 PyTorch 数据处理流程中，最佳实践通常是：
                数据加载和预处理在 CPU 上完成（使用 TensorDataset、DataLoader）。
                在训练/推理循环中，通过 DataLoader 迭代 batch，再将每个 batch 调用 .to(device) 异步传输到 GPU。
                这种模式可以隐藏数据传输延迟，并与 GPU 计算流水线并行。
           缓存复用场景的考虑：
                缓存的目的之一是在多次运行中重复使用。不同运行可能使用不同的 GPU（如单卡 vs 多卡），甚至不同的设备（CPU fallback）。
                将缓存数据保存为 CPU 张量是设备无关的，最大程度保持通用性。如果保存为 GPU 张量，加载时必须依赖原来的 GPU 环境，导致无法灵活迁移。
    '''
    data = torch.load(cache_dir / f"chunk_{chunk_idx:04d}.pt", map_location='cpu')
    return data['input_ids'], data['attention_mask'], data['ids']

# ---------- 主函数 ----------
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
    max_seq_len = step_cfg.get('max_seq_len', 256)
    gpu_ids = step_cfg.get('gpu_ids', [7])
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
    timing = {}

    # 1. 加载模型
    with TimedBlock("load_model", timing):
        logger.info("加载模型...")
        model, tokenizer, device = load_model_and_tokenizer(model_name, device_ids=gpu_ids)
    logger.info(f"模型加载完成，设备: {device}，耗时: {timing['load_model']:.2f}s")

    # 缓存目录（已包含模型名和长度，保证不同参数隔离）
    cache_dir = task_dir / "intermediate" / f"tokenized_cache_{model_name.replace('/', '_')}_{max_seq_len}"

    # 定义当前缓存的参数关键字段
    current_cache_params = {
        "model_name": model_name,
        "max_seq_len": max_seq_len,
        # 可选：添加 tokenizer 类名以检测 tokenizer 变化
        "tokenizer_class": tokenizer.__class__.__name__,
    }

    # 检查缓存是否有效（参数一致且分块完整）
    if is_cache_valid(cache_dir, current_cache_params):
        logger.info("发现有效缓存，将直接使用缓存（跳过tokenization）")
        full_cache_exists = True
    else:
        logger.info("缓存无效或不存在，将重新生成缓存")
        invalidate_cache(cache_dir)   # 删除无效缓存（若有）
        full_cache_exists = False

    data_prep_start = time.perf_counter()

    if full_cache_exists:
        # 加载缓存
        all_input_ids = []
        all_attention_masks = []
        all_ids = []
        chunk_files = sorted(cache_dir.glob("chunk_*.pt"))
        load_cache_start = time.perf_counter()
        for ch_file in tqdm(chunk_files, desc="加载缓存分块"):
            # 从文件名提取 chunk 索引（格式 chunk_0000.pt）
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
        # 重新生成缓存
        logger.info("未发现有效缓存，将进行tokenization并创建缓存")
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

        # 保存缓存元信息（包括分块数量）
        write_cache_meta(cache_dir, current_cache_params, chunk_idx)

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

    # 将数据准备耗时存储到 timing 字典（用于元数据）
    timing["data_preparation"] = data_prep_time

    # 3. 推理
    with TimedBlock("inference", timing):
        nll_scores, inference_time = compute_nll_from_tensors(
            model, input_ids_all, attention_mask_all, batch_size, device, desc="计算句子NLL"
        )
    logger.info(f"推理耗时: {timing['inference']:.2f}s, 平均每秒处理 {len(sampled_ids)/timing['inference']:.2f} 条句子")

    # 4. 构建结果 DataFrame 并保存
    full_df = pd.read_csv(input_csv)
    id_to_sent = dict(zip(full_df['id'], full_df['sentence']))
    df = pd.DataFrame({'id': sampled_ids, 'sentence': [id_to_sent[i] for i in sampled_ids], 'nll': nll_scores})

    if sample_ratio < 1.0:
        stem = output_csv.stem
        output_csv = output_csv.with_name(f"{stem}_sample_{int(sample_ratio*100)}{output_csv.suffix}")
    with TimedBlock("save_output", timing):
        df.to_csv(output_csv, index=False, encoding='utf-8')
    logger.info(f"结果保存至 {output_csv}，耗时 {timing['save_output']:.2f}s")

    timing["total_sec"] = time.perf_counter() - total_start
    logger.info(f"总耗时: {timing['total_sec']:.2f}s")

    # 构建当前运行的计时记录（含模型配置和采样信息）
    current_timing = {
        "timestamp": datetime.now().isoformat(),
        "load_model_sec": timing["load_model"],
        "data_preparation_sec": timing["data_preparation"],
        "inference_sec": timing["inference"],
        "save_output_sec": timing["save_output"],
        "total_sec": timing["total_sec"],
        "throughput_sentences_per_sec": len(sampled_ids) / timing["inference"] if timing["inference"] > 0 else 0,
        "tokenization_sec": tokenize_time if tokenize_time is not None else None,
        "used_cache": full_cache_exists
    }

    # 最新一次运行的关键配置
    latest_info = {
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

    metadata_path = task_dir / "run_metadata.json"
    update_metadata_timing(metadata_path, "01_compute_sentence_nll", current_timing, latest_info)

    logger.info("步骤01完成")

if __name__ == "__main__":
    main()