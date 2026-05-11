#!/usr/bin/env python3
"""
步骤04：统计分析，生成报告
支持从元数据动态获取步骤03的输出，自动适配采样比例，报告输出也置于带采样比例的子目录
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utils import setup_logger

def classify_error(token):
    if token.isdigit():
        return "数字"
    if token in "，。！？；：、“”‘’《》【】（）":
        return "标点符号"
    if token in ["嗯", "啊", "哦", "哎", "呀"]:
        return "语气词"
    if len(token) == 1 and '\u4e00' <= token <= '\u9fff':
        return "汉字"
    if token.isalpha():
        return "英文字母"
    return "其他"

def plot_nll_distribution(df, output_dir, logger):
    if df is None or 'nll' not in df.columns:
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

def plot_top_suspicious_tokens(token_stats, output_dir, top_n=30, min_count=5, logger=None):
    filtered = [(w, avg, cnt) for w, avg, cnt in token_stats if cnt >= min_count]
    filtered.sort(key=lambda x: x[1], reverse=True)
    top = filtered[:top_n]
    if not top:
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", type=str, required=True)
    args = parser.parse_args()

    config = json.loads(args.config_json)
    task_name = config['task_name']
    base_dir = Path(config['paths']['output']['base_dir'])
    task_dir = base_dir / task_name

    step_cfg = config['steps']['04_statistics']

    # 从元数据获取步骤03的输出路径（自动适配采样）
    metadata_path = task_dir / "run_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        default_word_csv = Path(metadata['03_compute_word_nll']['output_csv'])
        default_sentence_csv = Path(metadata['01_compute_sentence_nll']['output_csv'])
        sample_ratio = metadata['01_compute_sentence_nll'].get('sample_ratio', 1.0)
    else:
        default_word_csv = task_dir / "outputs/word_nll_details.csv"
        default_sentence_csv = task_dir / "intermediate/sentence_nll.csv"
        sample_ratio = 1.0

    word_nll_csv = Path(step_cfg.get('input_word_csv', default_word_csv))
    if not word_nll_csv.is_absolute():
        word_nll_csv = task_dir / word_nll_csv

    sentence_nll_csv = step_cfg.get('input_sentence_csv', default_sentence_csv)
    if sentence_nll_csv and not Path(sentence_nll_csv).is_absolute():
        sentence_nll_csv = task_dir / sentence_nll_csv

    output_dir = Path(step_cfg.get('output_dir', 'outputs/report'))
    if not output_dir.is_absolute():
        # 如果采样，输出目录也加上采样比例标识
        if sample_ratio < 1.0:
            output_dir = task_dir / f"outputs/sample_{int(sample_ratio*100)}_analysis/report"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = task_dir / f"outputs/{timestamp}_analysis/report"
    output_dir.mkdir(parents=True, exist_ok=True)

    top_k = step_cfg.get('top_k_suspicious_words', 30)
    min_occurrence = step_cfg.get('min_occurrence', 3)
    generate_plots = step_cfg.get('generate_plots', True)

    log_dir = task_dir / "logs"
    logger = setup_logger(log_dir, "04_statistics")
    logger.info(f"词级NLL文件: {word_nll_csv}")
    logger.info(f"输出报告目录: {output_dir}")

    if not word_nll_csv.exists():
        logger.error(f"词级NLL文件不存在: {word_nll_csv}")
        sys.exit(1)

    word_df = pd.read_csv(word_nll_csv)
    logger.info(f"加载 {len(word_df)} 条词级记录")

    total_tokens = len(word_df)
    avg_nll_all = word_df['nll'].mean()
    median_nll = word_df['nll'].median()
    high_thresh = word_df['nll'].quantile(0.95)
    high_nll_tokens = word_df[word_df['nll'] >= high_thresh]

    logger.info(f"总token数: {total_tokens}")
    logger.info(f"平均NLL: {avg_nll_all:.4f}, 中位数: {median_nll:.4f}, 95分位数: {high_thresh:.4f}")

    # 统计每个token的平均NLL
    token_groups = word_df.groupby('token')['nll'].agg(['mean', 'count']).reset_index()
    token_groups.columns = ['token', 'avg_nll', 'count']
    token_groups = token_groups[token_groups['count'] >= min_occurrence]
    token_groups = token_groups.sort_values('avg_nll', ascending=False)
    token_stats = token_groups[['token', 'avg_nll', 'count']].values.tolist()
    top_tokens = token_groups.head(top_k)

    # 错误类别分类
    error_categories = defaultdict(lambda: {"count": 0, "total_nll": 0.0})
    for _, row in word_df.iterrows():
        cat = classify_error(row['token'])
        error_categories[cat]["count"] += 1
        error_categories[cat]["total_nll"] += row['nll']
    cat_stats = []
    for cat, vals in error_categories.items():
        avg_nll = vals["total_nll"] / vals["count"]
        cat_stats.append((cat, vals["count"], avg_nll))
    cat_stats.sort(key=lambda x: x[1], reverse=True)

    # 绘制图表
    if generate_plots:
        sent_df = None
        if sentence_nll_csv and Path(sentence_nll_csv).exists():
            sent_df = pd.read_csv(sentence_nll_csv)
        plot_nll_distribution(sent_df, output_dir, logger)
        if token_stats:
            plot_top_suspicious_tokens(token_stats, output_dir, top_k, min_count=5, logger=logger)

    # 保存输出
    top_tokens.to_csv(output_dir / "top_suspicious_words.csv", index=False)
    pd.DataFrame(cat_stats, columns=["category", "count", "avg_nll"]).to_csv(output_dir / "error_categories.csv", index=False)

    summary = {
        "total_tokens": total_tokens,
        "avg_nll": avg_nll_all,
        "median_nll": median_nll,
        "percentile_95_nll": high_thresh,
        "high_nll_tokens_ratio": len(high_nll_tokens)/total_tokens,
        "top_suspicious_words": top_tokens.to_dict(orient="records"),
        "error_categories": cat_stats,
        "sample_ratio": sample_ratio,
        "timestamp": datetime.now().isoformat()
    }
    with open(output_dir / "summary_report.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Markdown报告
    md_path = output_dir / "analysis_report.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# ASR转录错误分析报告\n\n")
        f.write(f"**任务名称**: {task_name}\n")
        f.write(f"**采样比例**: {sample_ratio*100:.1f}%\n")
        f.write(f"**生成时间**: {datetime.now().isoformat()}\n\n")
        f.write("## 1. 总体统计\n\n")
        f.write(f"- 总Token数: {total_tokens}\n")
        f.write(f"- 平均NLL: {avg_nll_all:.4f}\n")
        f.write(f"- NLL中位数: {median_nll:.4f}\n")
        f.write(f"- NLL 95%分位数: {high_thresh:.4f}\n")
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

    logger.info(f"报告已保存至 {output_dir}")

    # 更新元数据
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    metadata['04_statistics'] = {
        "output_dir": str(output_dir),
        "total_tokens": total_tokens,
        "avg_nll": avg_nll_all,
        "sample_ratio": sample_ratio,
        "timestamp": datetime.now().isoformat()
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("步骤04完成")

if __name__ == "__main__":
    main()