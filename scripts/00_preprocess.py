#!/usr/bin/env python3
"""
步骤00：预处理原始ASR转录文本（带计时和历史记录）
"""

import csv
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utils import setup_logger, extract_sentences_from_file
from scripts.utils.timer import TimedBlock, update_metadata_timing

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", type=str, required=True)
    args = parser.parse_args()

    config = json.loads(args.config_json)
    task_name = config['task_name']
    base_dir = Path(config['paths']['output']['base_dir'])
    task_dir = base_dir / task_name

    step_cfg = config['steps']['00_preprocess']
    raw_data_dir = Path(config['paths']['input']['raw_data_dir'])
    output_csv = Path(step_cfg.get('output_csv', 'intermediate/all_sentences.csv'))
    if not output_csv.is_absolute():
        output_csv = task_dir / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    min_sentence_len = step_cfg.get('min_sentence_len', 3)
    remove_speaker_prefix = step_cfg.get('remove_speaker_prefix', True)
    split_by_punct = step_cfg.get('split_by_punct', False)

    log_dir = task_dir / "logs"
    logger = setup_logger(log_dir, "00_preprocess")
    logger.info(f"任务目录: {task_dir}")
    logger.info(f"原始数据目录: {raw_data_dir}")
    logger.info(f"输出CSV: {output_csv}")
    logger.info(f"参数: min_len={min_sentence_len}, remove_prefix={remove_speaker_prefix}, split_by_punct={split_by_punct}")

    total_start = time.perf_counter()
    timing = {}

    # 1. 扫描所有txt文件
    with TimedBlock("scan_files", timing):
        txt_files = list(raw_data_dir.rglob("*.txt"))
    logger.info(f"找到 {len(txt_files)} 个txt文件")

    if not txt_files:
        logger.warning("未找到任何txt文件")
        sys.exit(0)

    # 2. 提取句子
    with TimedBlock("extract_sentences", timing):
        all_sentences = []
        sentence_id = 0
        for file_path in txt_files:
            relative_path = str(file_path.relative_to(raw_data_dir))
            logger.debug(f"处理文件: {relative_path}")
            sentences = extract_sentences_from_file(
                file_path,
                remove_prefix=remove_speaker_prefix,
                split_by_punct=split_by_punct,
                min_len=min_sentence_len
            )
            for sent in sentences:
                all_sentences.append({
                    "id": sentence_id,
                    "file_path": relative_path,
                    "speaker": "unknown",
                    "sentence": sent
                })
                sentence_id += 1
    logger.info(f"共提取 {len(all_sentences)} 条有效句子")

    # 3. 写入CSV
    with TimedBlock("write_csv", timing):
        with open(output_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["id", "file_path", "speaker", "sentence"])
            writer.writeheader()
            writer.writerows(all_sentences)

    timing["total_sec"] = time.perf_counter() - total_start
    logger.info(f"总耗时: {timing['total_sec']:.2f}秒")
    logger.info(f"扫描文件耗时: {timing['scan_files']:.2f}秒")
    logger.info(f"提取句子耗时: {timing['extract_sentences']:.2f}秒")
    logger.info(f"写入CSV耗时: {timing['write_csv']:.2f}秒")

    # 构建当前运行的计时记录
    current_timing = {
        "timestamp": datetime.now().isoformat(),
        "scan_files_sec": timing["scan_files"],
        "extract_sentences_sec": timing["extract_sentences"],
        "write_csv_sec": timing["write_csv"],
        "total_sec": timing["total_sec"],
        "num_files": len(txt_files),
        "num_sentences": len(all_sentences),
        "min_sentence_len": min_sentence_len,
        "remove_speaker_prefix": remove_speaker_prefix,
        "split_by_punct": split_by_punct
    }

    # 最新一次运行的关键信息
    latest_info = {
        "output_csv": str(output_csv),
        "num_sentences": len(all_sentences),
        "min_sentence_len": min_sentence_len,
        "timestamp": datetime.now().isoformat()
    }

    metadata_path = task_dir / "run_metadata.json"
    update_metadata_timing(metadata_path, "00_preprocess", current_timing, latest_info)

    logger.info("步骤00完成")

if __name__ == "__main__":
    main()