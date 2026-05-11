#!/usr/bin/env python3
"""
步骤03：对高NLL句子进行词级NLL计算（批处理版本）
- 支持批量推理，显著提升 GPU 利用率
- 自动 padding 和对齐，返回每个 token 的 NLL
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utils import setup_logger, load_model_and_tokenizer

class WordNLLDataset(Dataset):
    """自定义数据集，存储句子列表，用于 DataLoader 批处理"""
    def __init__(self, sentences, ids):
        self.sentences = sentences
        self.ids = ids
    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, idx):
        return self.sentences[idx], self.ids[idx]

def collate_word_nll(batch, tokenizer, max_length):
    """
    自定义批处理函数：对 batch 内的句子进行 padding，返回 input_ids, attention_mask, 原始句子和 ids
    """
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
    """
    批处理计算每个句子的 token 级 NLL
    返回：字典 {id: (token_list, nll_list)} 和每个句子的整体平均 NLL (可选)
    """
    dataset = WordNLLDataset(sentences, ids)
    # 使用 lambda 包装 collate 函数，传递额外参数
    collate_fn = lambda b: collate_word_nll(b, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, pin_memory=True, num_workers=0)
    model.eval()
    result_tokens = {}
    result_nlls = {}
    all_seq_nll = []   # 可选：句子平均 NLL（可用于后续分析）
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
            token_nll = token_nll.view(shift_labels.size())  # (batch, seq_len-1)

            # 对每个句子，提取有效 token 的 NLL（忽略 padding 和第一个 token 的 NaN）
            for i in range(input_ids.size(0)):
                # 实际长度 = attention_mask[i] 中1的个数
                real_len = attention_mask[i].sum().item()
                # 有效 token 个数（忽略第一个特殊 token，如 [CLS] 或 [BOS]）
                # 注意：我们保留所有 token，但第一个 token 的 NLL 未定义，设为 NaN
                # 这里我们只取从位置1开始的 token（索引 1 到 real_len-1）
                seq_len = real_len  # 包括第一个特殊 token
                if seq_len <= 1:
                    # 空句子或只含特殊 token，跳过
                    continue
                # 提取该句子的 token NLL，长度 = real_len - 1（因为预测的是下一个 token，第一个 token 无 loss）
                nll_values = token_nll[i, :real_len-1].cpu().tolist()
                # 获取对应的 token 字符串（从 input_ids 解码）
                token_ids = input_ids[i, 1:real_len].cpu().tolist()  # 跳过第一个特殊 token
                token_strs = [tokenizer.decode([tid], skip_special_tokens=False).strip() for tid in token_ids]
                # 处理空字符串或 bytes（兼容 Qwen）
                for j, ts in enumerate(token_strs):
                    if not ts:
                        # 尝试用 convert_ids_to_tokens 并解码
                        alt = tokenizer.convert_ids_to_tokens(token_ids[j])
                        if isinstance(alt, bytes):
                            alt = alt.decode('utf-8', errors='replace')
                        token_strs[j] = alt
                # 存储结果
                sid = orig_ids[i]
                result_tokens[sid] = token_strs
                result_nlls[sid] = nll_values
                # 可选：句子平均 NLL（只计算有效 token，忽略第一个 token）
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

    input_csv_rel = Path(metadata['02_filter_high_nll']['output_csv'])
    project_root = base_dir.parent
    input_csv = project_root / input_csv_rel if not input_csv_rel.is_absolute() else input_csv_rel

    sample_ratio = metadata.get('01_compute_sentence_nll', {}).get('sample_ratio', 1.0)

    # 输出文件路径
    output_csv = Path(step_cfg.get('output_csv', 'outputs/word_nll_details.csv'))
    if not output_csv.is_absolute():
        output_csv = task_dir / output_csv
    if sample_ratio < 1.0:
        output_csv = output_csv.with_name(f"{output_csv.stem}_sample_{int(sample_ratio*100)}{output_csv.suffix}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # 参数
    max_seq_len = step_cfg.get('max_seq_len', 512)
    model_name = step_cfg.get('model_name', 'uer/gpt2-chinese-cluecorpussmall')
    gpu_ids = step_cfg.get('gpu_ids', [6, 7])
    batch_size = step_cfg.get('batch_size', 32)   # 批处理大小，根据显存调整
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

    # 加载模型（建议单卡，因为批处理后单卡足够）
    if len(gpu_ids) > 1:
        logger.warning("步骤03建议使用单卡以避免 DataParallel 开销，将只用第一个 GPU")
    model, tokenizer, device = load_model_and_tokenizer(model_name, device_ids=[gpu_ids[0]])
    logger.info(f"模型加载完成，设备: {device}")

    # 批量计算词级 NLL
    token_dict, nll_dict, seq_avg = compute_word_nll_batch(
        model, tokenizer, sentences, ids, max_seq_len, batch_size, device
    )

    # 构建详细记录列表
    all_records = []
    for sid in ids:   # 保持原始顺序
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
    result_df.to_csv(output_csv, index=False, encoding='utf-8')
    logger.info(f"词级NLL结果已保存至 {output_csv}，共 {len(result_df)} 个 token")

    # 可选：保存句子平均 NLL（用于后续）
    seq_avg_df = pd.DataFrame(seq_avg, columns=['id', 'seq_avg_nll'])
    seq_avg_csv = output_csv.parent / (output_csv.stem + "_seq_avg.csv")
    seq_avg_df.to_csv(seq_avg_csv, index=False)
    logger.info(f"句子平均NLL已保存至 {seq_avg_csv}")

    # 更新元数据
    metadata['03_compute_word_nll'] = {
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
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("步骤03完成")

if __name__ == "__main__":
    main()