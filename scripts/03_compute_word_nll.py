#!/usr/bin/env python3
"""
步骤03：对高NLL句子进行词（字）级NLL计算
- 读取高NLL句子CSV
- 对每条句子，遍历每个位置，计算每个token的负对数似然
- 输出每个字（或词）的NLL分数，便于定位具体错误
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

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


def compute_word_nll(model, tokenizer, sentence, device='cuda'):
    """
    计算句子中每个token的负对数似然
    返回: tokens (list of str), nlls (list of float)
    """
    # 编码
    inputs = tokenizer(sentence, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # 计算每个预测位置的损失（预测下一个token）
    # 对于位置 i，损失是预测 token_{i+1} 与真实 token_{i+1} 的交叉熵
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    token_nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    token_nll = token_nll.view(shift_labels.size())  # (1, seq_len-1)

    # 获取token字符串（第一个token通常是[CLS]或[BOS]，此处我们取所有token）
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    # 第一个token没有对应的NLL（因为没有前一个token预测它），所以NLL列表比tokens少1
    # 通常我们忽略第一个token（如BOS）的NLL
    nll_list = [float('nan')] * len(tokens)   # 占位
    # 填充对应位置的NLL：第 i 个token（i>=1）的NLL对应 shift_labels 的第 i-1 个
    for i in range(1, len(tokens)):
        nll_list[i] = token_nll[0, i-1].item()

    # 可选：过滤掉特殊token（如[SEP]），保留所有
    return tokens, nll_list


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

    step_cfg = config['steps']['03_compute_word_nll']
    input_csv = Path(step_cfg.get('input_csv', 'intermediate/high_nll_sentences.csv'))
    if not input_csv.is_absolute():
        input_csv = task_dir / input_csv
    output_csv = Path(step_cfg.get('output_csv', 'outputs/word_nll_details.csv'))
    if not output_csv.is_absolute():
        # 输出文件进入带时间戳的outputs子目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = task_dir / f"outputs/{timestamp}_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_csv = output_dir / "word_nll_details.csv"
    else:
        output_csv.parent.mkdir(parents=True, exist_ok=True)

    model_name = step_cfg.get('model_name', 'uer/gpt2-chinese-cluecorpussmall')
    batch_mode = step_cfg.get('batch_mode', False)   # 是否批量处理（暂不实现，单句循环）
    use_gpu = step_cfg.get('use_gpu', True)
    gpu_ids = step_cfg.get('gpu_ids', [6, 7])

    # 设置日志
    log_dir = task_dir / "logs"
    logger = setup_logger(log_dir, "03_compute_word_nll")
    logger.info(f"任务目录: {task_dir}")
    logger.info(f"输入CSV: {input_csv}")
    logger.info(f"输出CSV: {output_csv}")
    logger.info(f"模型: {model_name}")

    # 检查输入
    if not input_csv.exists():
        logger.error(f"输入文件不存在: {input_csv}")
        sys.exit(1)

    df = pd.read_csv(input_csv)
    sentences = df['sentence'].tolist()
    sentence_ids = df['id'].tolist()
    logger.info(f"共加载 {len(sentences)} 条高NLL句子")

    # 加载模型
    logger.info(f"正在加载模型 {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 设置设备
    if use_gpu and torch.cuda.is_available():
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
    model.eval()

    # 逐句计算词级NLL
    all_records = []
    for sent_id, sentence in tqdm(zip(sentence_ids, sentences), total=len(sentences), desc="逐句分析"):
        tokens, nlls = compute_word_nll(model, tokenizer, sentence, device)
        # 记录每个token的信息
        for idx, (token, nll) in enumerate(zip(tokens, nlls)):
            # 跳过nan（第一个token）
            if pd.isna(nll):
                continue
            all_records.append({
                "sentence_id": sent_id,
                "token_index": idx,
                "token": token,
                "nll": nll,
                "sentence": sentence
            })

    logger.info(f"共产生 {len(all_records)} 条 token 级记录")

    # 保存为CSV
    result_df = pd.DataFrame(all_records)
    result_df.to_csv(output_csv, index=False, encoding='utf-8')
    logger.info(f"已保存至 {output_csv}")

    # 更新元数据
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