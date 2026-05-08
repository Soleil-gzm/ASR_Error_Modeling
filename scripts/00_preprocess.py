#!/usr/bin/env python3
"""
步骤00：预处理原始ASR转录文本
"""

import csv
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utils import setup_logger, extract_sentences_from_file



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

    txt_files = list(raw_data_dir.rglob("*.txt"))
    logger.info(f"找到 {len(txt_files)} 个txt文件")

    if not txt_files:
        logger.warning("未找到任何txt文件")
        sys.exit(0)

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

    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "file_path", "speaker", "sentence"])
        writer.writeheader()
        writer.writerows(all_sentences)

    # 更新元数据
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

    logger.info("步骤00完成")

if __name__ == "__main__":
    main()