#!/usr/bin/env python3
"""
步骤01：计算句子级平均负对数似然（NLL）并保存
- 加载预训练中文GPT-2模型
- 使用DataParallel在多个GPU上并行
- 输出每个句子的平均NLL（越低表示越符合语言模型，越高越可疑）
"""

import csv
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# 尝试导入自定义日志模块
try:
    from scripts.utils.logger import setup_logger
except ImportError:
    import logging
    def setup_logger(log_dir, name):
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{name}.log"
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        if logger.handlers:
            logger.handlers.clear()
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger


class SentenceDataset(Dataset):
    """数据集：封装句子列表，用于DataLoader"""
    def __init__(self, sentences, tokenizer, max_length=512):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        text = self.sentences[idx]
        # 编码并截断
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        # 返回 input_ids 和 attention_mask
        return enc['input_ids'][0], enc['attention_mask'][0]


def compute_sentence_nll_batch(model, tokenizer, sentences, batch_size=32, max_length=512, device='cuda'):
    """
    批量计算每条句子的平均负对数似然
    返回：NLL列表（float）
    """
    dataset = SentenceDataset(sentences, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    all_nll = []
    total_batches = len(dataloader)

    with torch.no_grad():
        for input_ids, attention_mask in tqdm(dataloader, desc="计算NLL", total=total_batches):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # 前向传播，获取logits
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            logits = outputs.logits  # (batch, seq_len, vocab_size)

            # 移动一位，计算每个token的交叉熵损失（reduction='none'）
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            token_nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            token_nll = token_nll.view(shift_labels.size())  # (batch, seq_len-1)

            # 忽略padding部分：attention_mask中第1个位置之后为1的地方才是有效token
            mask = attention_mask[:, 1:].contiguous()  # (batch, seq_len-1)
            seq_nll = (token_nll * mask).sum(dim=1) / mask.sum(dim=1)  # (batch,)

            all_nll.extend(seq_nll.cpu().tolist())

    return all_nll


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", type=str, help="全局配置JSON字符串")
    args = parser.parse_args()

    if not args.config_json:
        print("错误：必须提供 --config_json 参数")
        sys.exit(1)

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
    use_gpu = step_cfg.get('use_gpu', True)
    gpu_ids = step_cfg.get('gpu_ids', [6, 7])  # 使用6、7号GPU

    # 设置日志
    log_dir = task_dir / "logs"
    logger = setup_logger(log_dir, "01_compute_sentence_nll")
    logger.info(f"任务目录: {task_dir}")
    logger.info(f"输入CSV: {input_csv}")
    logger.info(f"输出CSV: {output_csv}")
    logger.info(f"模型: {model_name}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"最大序列长度: {max_seq_len}")
    logger.info(f"使用GPU: {use_gpu}, GPU IDs: {gpu_ids}")

    # 读取输入CSV
    if not input_csv.exists():
        logger.error(f"输入文件不存在: {input_csv}")
        sys.exit(1)

    df = pd.read_csv(input_csv)
    sentences = df['sentence'].tolist()
    logger.info(f"共加载 {len(sentences)} 条句子")

    # 加载模型和分词器
    logger.info(f"正在加载模型 {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 添加pad_token（如果不存在）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 设置设备
    if use_gpu and torch.cuda.is_available():
        # 将模型放到第一个GPU上，然后用DataParallel包装
        device = torch.device(f"cuda:{gpu_ids[0]}")
        model = model.to(device)
        if len(gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
            logger.info(f"使用 DataParallel 在 {gpu_ids} 上并行")
        else:
            logger.info(f"使用单GPU: {gpu_ids[0]}")
    else:
        device = torch.device("cpu")
        logger.info("使用CPU")
    logger.info("模型加载完成")

    # 计算NLL
    logger.info("开始计算句子NLL...")
    nll_scores = compute_sentence_nll_batch(
        model, tokenizer, sentences,
        batch_size=batch_size,
        max_length=max_seq_len,
        device=device
    )
    logger.info(f"计算完成，共 {len(nll_scores)} 条分数")

    # 添加NLL列到DataFrame
    df['nll'] = nll_scores

    # 保存结果
    df.to_csv(output_csv, index=False, encoding='utf-8')
    logger.info(f"结果已保存至: {output_csv}")

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
        "use_gpu": use_gpu,
        "gpu_ids": gpu_ids,
        "num_sentences": len(sentences),
        "timestamp": datetime.now().isoformat()
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("步骤01完成")


if __name__ == "__main__":
    main()