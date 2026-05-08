#!/usr/bin/env python3
"""
步骤04：统计分析，生成报告
- 输入：词级NLL详表（03步骤输出）、句子级NLL表（可选）
- 输出：高频可疑词、NLL分布图、错误模式摘要、汇总报告
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_nll_distribution(df, output_dir, logger):
    """绘制句子级NLL分布直方图和箱线图"""
    if df is None or 'nll' not in df.columns:
        logger.warning("无句子级NLL数据，跳过分布图")
        return
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(df['nll'], bins=50, alpha=0.7, color='steelblue')
    plt.xlabel('句子平均NLL')
    plt.ylabel('频数')
    plt.title('句子平均NLL分布')
    plt.subplot(1, 2, 2)
    plt.boxplot(df['nll'], vert=False)
    plt.xlabel('句子平均NLL')
    plt.title('NLL箱线图')
    plt.tight_layout()
    out_path = output_dir / "nll_distribution.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"分布图已保存: {out_path}")

def plot_top_suspicious_tokens(token_stats, output_dir, top_n=30, logger=None):
    """绘制top N高平均NLL的词（出现次数>=min_count）"""
    # 过滤出现次数少的词
    min_count = 5
    filtered = [(w, avg_nll, cnt) for w, avg_nll, cnt in token_stats if cnt >= min_count]
    filtered.sort(key=lambda x: x[1], reverse=True)
    top = filtered[:top_n]
    if not top:
        logger.warning("无可视化的高频可疑词")
        return
    words = [w for w, _, _ in top]
    avg_nlls = [avg for _, avg, _ in top]
    plt.figure(figsize=(10, 6))
    plt.barh(words, avg_nlls, color='coral')
    plt.xlabel('平均NLL')
    plt.title(f'平均NLL最高的{len(top)}个词（出现≥{min_count}次）')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    out_path = output_dir / "top_suspicious_words.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    if logger:
        logger.info(f"Top可疑词图已保存: {out_path}")

def classify_error(token):
    """简单的错误模式分类（基于规则）"""
    if token.isdigit():
        return "数字"
    if token in "，。！？；：、“”‘’《》【】（）":
        return "标点符号"
    # 常见语气词或重复字
    if token in ["嗯", "啊", "哦", "哎", "呀"]:
        return "语气词"
    if len(token) == 1 and '\u4e00' <= token <= '\u9fff':
        # 中文字符，但未分类
        return "汉字"
    if token.isalpha():
        return "英文字母"
    return "其他"

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

    step_cfg = config['steps']['04_statistics']
    word_nll_csv = Path(step_cfg.get('input_word_csv', 'outputs/word_nll_details.csv'))
    if not word_nll_csv.is_absolute():
        # 需要加上时间戳子目录？实际路径可能包含时间戳，用户需指定完整相对路径或绝对路径
        # 简化：假设word_nll_csv是相对于task_dir的路径，但通常包含outputs/{timestamp}_analysis/前缀
        # 在配置中必须明确给出相对路径（如"outputs/20250508_120000_analysis/word_nll_details.csv"）
        # 我们使用task_dir / word_nll_csv
        word_nll_csv = task_dir / word_nll_csv
    sentence_nll_csv = step_cfg.get('input_sentence_csv', 'intermediate/sentence_nll.csv')
    if sentence_nll_csv and not Path(sentence_nll_csv).is_absolute():
        sentence_nll_csv = task_dir / sentence_nll_csv
    output_dir = Path(step_cfg.get('output_dir', 'outputs/report'))
    if not output_dir.is_absolute():
        # 添加时间戳子目录（如果用户未指定绝对路径，则自动创建时间戳子目录）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = task_dir / f"outputs/{timestamp}_analysis/report"
    output_dir.mkdir(parents=True, exist_ok=True)

    top_k = step_cfg.get('top_k_suspicious_words', 30)
    min_occurrence = step_cfg.get('min_occurrence', 3)  # 统计词至少出现次数
    generate_plots = step_cfg.get('generate_plots', True)

    # 日志
    log_dir = task_dir / "logs"
    logger = setup_logger(log_dir, "04_statistics")
    logger.info(f"任务目录: {task_dir}")
    logger.info(f"词级NLL文件: {word_nll_csv}")
    logger.info(f"输出报告目录: {output_dir}")

    # 1. 加载词级数据
    if not word_nll_csv.exists():
        logger.error(f"词级NLL文件不存在: {word_nll_csv}")
        sys.exit(1)
    word_df = pd.read_csv(word_nll_csv)
    logger.info(f"加载 {len(word_df)} 条词级记录")

    # 2. 基础统计
    total_tokens = len(word_df)
    avg_nll_all = word_df['nll'].mean()
    median_nll = word_df['nll'].median()
    high_nll_threshold = word_df['nll'].quantile(0.95)
    high_nll_tokens = word_df[word_df['nll'] >= high_nll_threshold]

    logger.info(f"总token数: {total_tokens}")
    logger.info(f"平均NLL: {avg_nll_all:.4f}, 中位数: {median_nll:.4f}, 95分位数: {high_nll_threshold:.4f}")
    logger.info(f"NLL ≥ 95分位数的token数: {len(high_nll_tokens)} ({len(high_nll_tokens)/total_tokens*100:.2f}%)")

    # 3. 统计每个token的平均NLL和出现次数
    token_stats = []
    token_groups = word_df.groupby('token')['nll'].agg(['mean', 'count']).reset_index()
    token_groups.columns = ['token', 'avg_nll', 'count']
    token_groups = token_groups[token_groups['count'] >= min_occurrence]
    token_groups = token_groups.sort_values('avg_nll', ascending=False)
    token_stats = token_groups[['token', 'avg_nll', 'count']].values.tolist()
    top_tokens = token_groups.head(top_k)
    logger.info(f"共 {len(token_groups)} 种token满足出现≥{min_occurrence}次")
    logger.info("Top 10 高平均NLL词:")
    for i, row in top_tokens.head(10).iterrows():
        logger.info(f"  {row['token']}: avg_nll={row['avg_nll']:.4f}, 出现次数={row['count']}")

    # 4. 错误模式分类（基于规则）
    error_categories = defaultdict(lambda: {"count": 0, "total_nll": 0.0})
    for _, row in word_df.iterrows():
        token = row['token']
        nll = row['nll']
        cat = classify_error(token)
        error_categories[cat]["count"] += 1
        error_categories[cat]["total_nll"] += nll
    # 计算平均NLL
    cat_stats = []
    for cat, vals in error_categories.items():
        avg_nll = vals["total_nll"] / vals["count"] if vals["count"] > 0 else 0
        cat_stats.append((cat, vals["count"], avg_nll))
    cat_stats.sort(key=lambda x: x[1], reverse=True)  # 按出现次数排序
    logger.info("错误类别统计（按次数）:")
    for cat, cnt, avg in cat_stats:
        logger.info(f"  {cat}: 出现次数={cnt}, 平均NLL={avg:.4f}")

    # 5. 可选：加载句子级NLL并绘制分布图
    sentence_df = None
    if sentence_nll_csv and Path(sentence_nll_csv).exists():
        sentence_df = pd.read_csv(sentence_nll_csv)
        logger.info(f"加载句子级NLL数据: {len(sentence_df)} 条")
    else:
        logger.warning("未提供句子级NLL文件，跳过分布图")

    if generate_plots:
        plot_nll_distribution(sentence_df, output_dir, logger)
        plot_top_suspicious_tokens(token_stats, output_dir, top_k, logger)

    # 6. 保存报告文件
    # 6.1 高频可疑词CSV
    top_words_csv = output_dir / "top_suspicious_words.csv"
    top_tokens.to_csv(top_words_csv, index=False)
    logger.info(f"高频可疑词列表保存至: {top_words_csv}")

    # 6.2 错误类别统计CSV
    cat_df = pd.DataFrame(cat_stats, columns=["category", "count", "avg_nll"])
    cat_csv = output_dir / "error_categories.csv"
    cat_df.to_csv(cat_csv, index=False)
    logger.info(f"错误类别统计保存至: {cat_csv}")

    # 6.3 汇总报告（JSON）
    summary = {
        "total_tokens": total_tokens,
        "avg_nll": avg_nll_all,
        "median_nll": median_nll,
        "percentile_95_nll": high_nll_threshold,
        "high_nll_tokens_ratio": len(high_nll_tokens)/total_tokens,
        "top_suspicious_words": top_tokens.to_dict(orient="records"),
        "error_categories": cat_stats,
        "timestamp": datetime.now().isoformat()
    }
    summary_json = output_dir / "summary_report.json"
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"汇总报告保存至: {summary_json}")

    # 7. 可选：输出Markdown报告（简单版本）
    md_path = output_dir / "analysis_report.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# ASR转录错误分析报告\n\n")
        f.write(f"**任务名称**: {task_name}\n")
        f.write(f"**生成时间**: {datetime.now().isoformat()}\n\n")
        f.write("## 1. 总体统计\n\n")
        f.write(f"- 总Token数: {total_tokens}\n")
        f.write(f"- 平均NLL: {avg_nll_all:.4f}\n")
        f.write(f"- NLL中位数: {median_nll:.4f}\n")
        f.write(f"- NLL 95%分位数: {high_nll_threshold:.4f}\n")
        f.write(f"- NLL≥95分位数的Token占比: {len(high_nll_tokens)/total_tokens*100:.2f}%\n\n")
        f.write("## 2. 高频可疑词（Top 10）\n\n")
        f.write("| Token | 平均NLL | 出现次数 |\n")
        f.write("|-------|---------|----------|\n")
        for _, row in top_tokens.head(10).iterrows():
            f.write(f"| {row['token']} | {row['avg_nll']:.4f} | {row['count']} |\n")
        f.write("\n## 3. 错误类别分布\n\n")
        f.write("| 类别 | 出现次数 | 平均NLL |\n")
        f.write("|------|----------|---------|\n")
        for cat, cnt, avg in cat_stats:
            f.write(f"| {cat} | {cnt} | {avg:.4f} |\n")
        f.write("\n## 4. 结论与建议\n\n")
        f.write("(根据以上数据，请人工分析主要错误模式并给出改进建议。)\n")
    logger.info(f"Markdown报告保存至: {md_path}")

    # 更新元数据
    metadata_path = task_dir / "run_metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    metadata['04_statistics'] = {
        "output_dir": str(output_dir),
        "total_tokens": total_tokens,
        "avg_nll": avg_nll_all,
        "timestamp": datetime.now().isoformat()
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("步骤04完成")

if __name__ == "__main__":
    main()