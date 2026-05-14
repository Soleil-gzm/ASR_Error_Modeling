#!/usr/bin/env python3
"""
过滤低频词对，生成前置词统计表（prev_word, total_occurrences, unique_abnormal, abnormal_words）
支持：
   - 根据词对频次过滤（min_count）
   - 根据前置词对应的异常词种类数过滤（min_unique_abnormal）
   - 可选：过滤掉前后都是纯数字的词对
   - 异常词后自动附加出现概率（括号内小数），并按概率降序排列
   - 统一日志输出，自动将日志放入 work/{task_name}/logs/
用法:
    python filter_pairs.py --input <噪声对文件> --output <输出目录> [--min_count 2] [--min_unique_abnormal 2] [--drop_digit_pairs]
"""

import sys
import logging
import argparse
from pathlib import Path
import pandas as pd

# 导入项目统一日志和过滤函数
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import setup_logger
from utils.pair_filters import filter_digit_pairs, filter_by_min_count, aggregate_by_prev

def get_task_dir_from_input(input_path: Path, base_dir_name: str = "work") -> Path:
    """
    从输入文件路径推断任务目录。
    例如：work/test_Qwen_pt/outputs/... -> work/test_Qwen_pt
    """
    parts = input_path.absolute().parts
    try:
        idx = parts.index(base_dir_name)
        if idx + 1 < len(parts):
            return Path(*parts[:idx+2])
    except ValueError:
        pass
    return Path.cwd() / base_dir_name / "unknown_task"

def main():
    # ================== 可修改的硬编码默认值 ==================
    DEFAULT_INPUT = "work/test_Qwen_pt/outputs/sample_20_analysis/prev_window_1/noise_pairs.csv"
    DEFAULT_OUTPUT = "work/test_Qwen_pt/outputs/sample_20_analysis/prev_clean"
    DEFAULT_MIN_COUNT = 2
    DEFAULT_MIN_UNIQUE_ABNORMAL = 2
    DEFAULT_DROP_DIGIT_PAIRS = True
    # ========================================================

    parser = argparse.ArgumentParser(description="从noise_pairs.csv中过滤低频词对，生成前置词统计")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT,
                        help=f"输入的CSV文件路径（默认: {DEFAULT_INPUT}）")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help=f"输出目录（默认: {DEFAULT_OUTPUT}）")
    parser.add_argument("--min_count", type=int, default=DEFAULT_MIN_COUNT,
                        help=f"最小频次阈值，保留出现次数≥该值的词对（默认: {DEFAULT_MIN_COUNT}）")
    parser.add_argument("--min_unique_abnormal", type=int, default=DEFAULT_MIN_UNIQUE_ABNORMAL,
                        help=f"最小异常词种类数，保留前置词对应的不同异常词数量≥该值（默认: {DEFAULT_MIN_UNIQUE_ABNORMAL}）")
    parser.add_argument("--drop_digit_pairs", action='store_true', default=DEFAULT_DROP_DIGIT_PAIRS,
                        help="是否过滤掉前置词和异常词都是纯数字的词对（默认启用）")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="自定义日志目录（若不指定，则从输入文件路径自动推断 task_dir 下的 logs）")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="控制台日志级别（默认 INFO）")
    args = parser.parse_args()

    # 确定日志目录
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        input_path = Path(args.input)
        task_dir = get_task_dir_from_input(input_path)
        log_dir = task_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志：文件级别 DEBUG，控制台级别按参数
    logger = setup_logger(log_dir, "filter_pairs", 
                          level=logging.DEBUG,
                          console_level=getattr(logging, args.log_level))
    logger.info(f"开始处理，输入文件: {args.input}")
    logger.info(f"日志文件目录: {log_dir}")

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取数据
    try:
        df = pd.read_csv(input_path)
        logger.info(f"原始词对数量: {len(df)}")
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        sys.exit(1)

    # 检查列名
    if 'prev_word' not in df.columns or 'abnormal_word' not in df.columns:
        logger.error("CSV文件必须包含 'prev_word' 和 'abnormal_word' 列")
        sys.exit(1)

    # 1. 数字对过滤
    if args.drop_digit_pairs:
        before = len(df)
        df = filter_digit_pairs(df)
        logger.info(f"数字对过滤后剩余 {len(df)} 条 (移除 {before - len(df)})")
    else:
        logger.info("跳过数字对过滤")

    # 2. 频次过滤
    df_counts = filter_by_min_count(df, args.min_count)
    logger.info(f"频次过滤后唯一词对数量: {len(df_counts)} (要求出现≥{args.min_count})")

    # 3. 按前置词聚合
    grouped = aggregate_by_prev(df_counts, with_prob=True)

    # 4. 按异常词种类过滤
    before = len(grouped)
    grouped = grouped[grouped['unique_abnormal'] >= args.min_unique_abnormal]
    logger.info(f"异常词种类过滤后剩余前置词数量: {len(grouped)} (要求 unique_abnormal ≥ {args.min_unique_abnormal})")

    # 输出到CSV
    output_file = output_dir / "prev_clean_summary.csv"
    grouped.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"结果已保存至: {output_file}")

if __name__ == "__main__":
    main()