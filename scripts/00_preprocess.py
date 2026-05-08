#!/usr/bin/env python3
"""
步骤00：预处理原始ASR转录文本
- 递归扫描 datas/original 目录下所有 .txt 文件
- 提取每行对话内容（去除“说话人X: ”前缀）
- 可选：根据标点分句（默认按行分句）
- 过滤过短句子
- 输出 CSV 文件：id, file_path, speaker, sentence
"""

import csv
import re
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# 尝试导入自定义日志模块（如果存在，否则使用基础日志）
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

def extract_sentences_from_file(file_path, remove_speaker_prefix=True, split_by_punct=False):
    """
    读取单个txt文件，返回句子列表（每句为一个字符串）
    """
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 去除说话人前缀（如“说话人2: ”、“说话人1：”）
            if remove_speaker_prefix:
                # 匹配“说话人数字:”或“说话人数字：”
                line = re.sub(r'^说话人\d+[：:]', '', line).strip()
            if not line:
                continue
            # 按标点分句（可选，默认不启用，保留整行作为一句）
            if split_by_punct:
                # 按句号、问号、感叹号、分号分割
                parts = re.split(r'[。？!；]', line)
                for part in parts:
                    part = part.strip()
                    if part:
                        sentences.append(part)
            else:
                sentences.append(line)
    return sentences

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

    # 获取步骤配置
    step_cfg = config['steps']['00_preprocess']
    raw_data_dir = Path(config['paths']['input']['raw_data_dir'])
    output_csv = Path(step_cfg.get('output_csv', 'intermediate/all_sentences.csv'))
    if not output_csv.is_absolute():
        output_csv = task_dir / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    min_sentence_len = step_cfg.get('min_sentence_len', 3)
    remove_speaker_prefix = step_cfg.get('remove_speaker_prefix', True)
    split_by_punct = step_cfg.get('split_by_punct', False)

    # 设置日志
    log_dir = task_dir / "logs"
    logger = setup_logger(log_dir, "00_preprocess")
    logger.info(f"任务目录: {task_dir}")
    logger.info(f"原始数据目录: {raw_data_dir}")
    logger.info(f"输出CSV: {output_csv}")
    logger.info(f"最小句子长度: {min_sentence_len}")

    # 递归查找所有 .txt 文件
    txt_files = list(raw_data_dir.rglob("*.txt"))
    logger.info(f"找到 {len(txt_files)} 个txt文件")

    if not txt_files:
        logger.warning("未找到任何txt文件，请检查 raw_data_dir 路径")
        sys.exit(0)

    # 处理每个文件
    all_sentences = []  # 每条记录为一个字典
    sentence_id = 0
    for file_path in txt_files:
        relative_path = str(file_path.relative_to(raw_data_dir))
        logger.debug(f"处理文件: {relative_path}")
        sentences = extract_sentences_from_file(file_path, remove_speaker_prefix, split_by_punct)
        for sent in sentences:
            if len(sent) < min_sentence_len:
                continue
            # 提取说话人（如果需要，可以从原行保留，但这里简化）
            speaker = "unknown"  # 可根据需要从文件名或内容提取，暂不实现
            all_sentences.append({
                "id": sentence_id,
                "file_path": relative_path,
                "speaker": speaker,
                "sentence": sent
            })
            sentence_id += 1

    logger.info(f"共提取 {len(all_sentences)} 条有效句子")

    # 写入CSV
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ["id", "file_path", "speaker", "sentence"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_sentences)

    logger.info(f"CSV文件已保存: {output_csv}")

    # 可选：保存一个简要统计到 metadata
    metadata_path = task_dir / "run_metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    metadata['00_preprocess'] = {
        "num_files": len(txt_files),
        "num_sentences": len(all_sentences),
        "min_sentence_len": min_sentence_len,
        "output_csv": str(output_csv),
        "timestamp": datetime.now().isoformat()
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    main()