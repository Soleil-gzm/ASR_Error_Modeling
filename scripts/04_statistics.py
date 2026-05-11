#!/usr/bin/env python3
"""
步骤04：统计分析，生成报告（支持自动词级聚合）
- 根据模型类型自动将 token 级 NLL 聚合为词级（仅对 GPT-2 等字级模型）
- 输出词级统计、图表和报告
"""

import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jieba

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utils import setup_logger
from scripts.utils.timer import TimedBlock, update_metadata_timing

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm

# --- 中文字体配置（稳健版）---
# 方法1：尝试直接设置已知中文字体名（优先）
try:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 测试字体是否可用
    fig, ax = plt.subplots()
    ax.set_title('测试')
    plt.close(fig)
except:
    # 方法2：手动添加字体文件路径
    font_paths = [
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/System/Library/Fonts/PingFang.ttc'  # macOS
    ]
    added = False
    for path in font_paths:
        if fm.findfont(path, fallback_to_default=False):
            fm.fontManager.addfont(path)
            font_name = fm.FontProperties(fname=path).get_name()
            plt.rcParams['font.sans-serif'] = [font_name]
            added = True
            break
    if not added:
        print("警告：未找到中文字体，图表将无法显示中文")
    plt.rcParams['axes.unicode_minus'] = False
# --- 配置结束 ---

# ------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------
def classify_error(token):
    """分类错误类型，token 可能为字符串或数值，需先处理"""
    if not isinstance(token, str):
        return "其他"
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

def plot_top_suspicious_words(word_stats, output_dir, top_n=30, min_count=5, logger=None):
    """word_stats: list of (word, avg_nll, count)"""
    filtered = [(w, avg, cnt) for w, avg, cnt in word_stats if cnt >= min_count]
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

# ------------------------------------------------------------
# 词级聚合函数（用于 GPT-2 等字级模型）
# ------------------------------------------------------------
def aggregate_tokens_to_words(df, logger=None):
    """
    将字级 token 的 NLL 按 jieba 分词结果聚合为词级
    df: 包含 token 级记录（列：sentence_id, token_index, token, nll, sentence）
    返回：词级 DataFrame，列：sentence_id, word, avg_nll, sentence
    """
    if logger:
        logger.info("正在将 token 级 NLL 聚合为词级（使用 jieba）...")
    records = []
    for sent_id, group in df.groupby('sentence_id'):
        sentence = group['sentence'].iloc[0]
        token_rows = group[group['token_index'] > 0].sort_values('token_index')
        if token_rows.empty:
            continue
        tokens = token_rows['token'].tolist()
        nlls = token_rows['nll'].tolist()
        words = list(jieba.cut(sentence))
        idx = 0
        for w in words:
            length = len(w)
            if idx + length <= len(tokens):
                word_nlls = nlls[idx:idx+length]
                avg_nll = sum(word_nlls) / length if length > 0 else float('nan')
                records.append({
                    'sentence_id': sent_id,
                    'word': w,
                    'avg_nll': avg_nll,
                    'sentence': sentence
                })
                idx += length
            else:
                if logger:
                    logger.warning(f"句子 {sent_id} token 数量不足，部分词被忽略")
                break
    result_df = pd.DataFrame(records)
    # 确保 word 列是字符串类型
    result_df['word'] = result_df['word'].astype(str)
    if logger:
        logger.info(f"聚合完成，共 {len(result_df)} 个词级记录")
    return result_df

# ------------------------------------------------------------
# 主函数
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", type=str, required=True)
    args = parser.parse_args()

    config = json.loads(args.config_json)
    task_name = config['task_name']
    base_dir = Path(config['paths']['output']['base_dir'])
    task_dir = base_dir / task_name

    step_cfg = config['steps']['04_statistics']

    log_dir = task_dir / "logs"
    logger = setup_logger(log_dir, "04_statistics")

    # 从元数据获取步骤03的输出路径
    metadata_path = task_dir / "run_metadata.json"
    if not metadata_path.exists():
        logger.error(f"元数据文件不存在: {metadata_path}，请先运行步骤03")
        sys.exit(1)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    if '03_compute_word_nll' not in metadata:
        logger.error("元数据中缺少步骤03的记录，请先运行步骤03")
        sys.exit(1)

    word_nll_csv_rel = Path(metadata['03_compute_word_nll']['output_csv'])
    project_root = base_dir.parent
    if word_nll_csv_rel.is_absolute():
        word_nll_csv = word_nll_csv_rel
    else:
        word_nll_csv = project_root / word_nll_csv_rel

    model_name = metadata.get('03_compute_word_nll', {}).get('model_name', '')
    sample_ratio = metadata.get('01_compute_sentence_nll', {}).get('sample_ratio', 1.0)

    sentence_nll_csv = None
    if '01_compute_sentence_nll' in metadata:
        sent_rel = Path(metadata['01_compute_sentence_nll']['output_csv'])
        if sent_rel.is_absolute():
            sentence_nll_csv = sent_rel
        else:
            sentence_nll_csv = project_root / sent_rel

    output_dir = Path(step_cfg.get('output_dir', 'outputs/report'))
    if not output_dir.is_absolute():
        if sample_ratio < 1.0:
            output_dir = task_dir / f"outputs/sample_{int(sample_ratio*100)}_analysis/report"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = task_dir / f"outputs/{timestamp}_analysis/report"
    output_dir.mkdir(parents=True, exist_ok=True)

    top_k = step_cfg.get('top_k_suspicious_words', 30)
    min_occurrence = step_cfg.get('min_occurrence', 3)
    generate_plots = step_cfg.get('generate_plots', True)

    logger.info(f"词级NLL文件: {word_nll_csv}")
    logger.info(f"输出报告目录: {output_dir}")
    logger.info(f"采样比例: {sample_ratio}")
    logger.info(f"模型名称: {model_name}")

    if not word_nll_csv.exists():
        logger.error(f"词级NLL文件不存在: {word_nll_csv}")
        sys.exit(1)

    total_start = time.perf_counter()
    timing = {}

    # 1. 读取 token 级数据
    with TimedBlock("read_token_csv", timing):
        token_df = pd.read_csv(word_nll_csv)
    logger.info(f"加载 {len(token_df)} 条 token 级记录")

    # 2. 根据模型自动聚合
    with TimedBlock("aggregate_words", timing):
        need_aggregate = ('gpt2' in model_name.lower() and 'qwen' not in model_name.lower())
        if need_aggregate:
            logger.info("检测到 GPT-2 模型，将 token 级 NLL 聚合为词级")
            df = aggregate_tokens_to_words(token_df, logger)
        else:
            logger.info("模型 token 已是词级别，直接使用 token 作为词")
            df = token_df.rename(columns={'token': 'word', 'nll': 'avg_nll'})
            df = df[['sentence_id', 'word', 'avg_nll', 'sentence']]
            # 确保 word 列是字符串（消除可能的浮点数）
            df['word'] = df['word'].astype(str)
            
        # 保存聚合后的词级数据（方便后续使用）
        agg_suffix = f"_sample_{int(sample_ratio*100)}" if sample_ratio < 1.0 else ""
        agg_filename = f"word_level_aggregated{agg_suffix}.csv"
        aggregated_csv = output_dir / agg_filename
        df.to_csv(aggregated_csv, index=False, encoding='utf-8')
        logger.info(f"聚合后的词级数据已保存至 {aggregated_csv}")
        logger.info(f"分析数据共有 {len(df)} 条词级记录")

    if len(df) == 0:
        logger.error("没有有效词级记录，无法生成报告")
        sys.exit(1)

    # 3. 统计词级信息
    with TimedBlock("statistics", timing):
        total_words = len(df)
        avg_nll_all = df['avg_nll'].mean()
        median_nll = df['avg_nll'].median()
        high_thresh = df['avg_nll'].quantile(0.95)
        high_nll_words = df[df['avg_nll'] >= high_thresh]

        logger.info(f"总词数: {total_words}")
        logger.info(f"平均NLL: {avg_nll_all:.4f}, 中位数: {median_nll:.4f}, 95分位数: {high_thresh:.4f}")

        # 统计每个词的平均NLL和出现次数
        word_groups = df.groupby('word')['avg_nll'].agg(['mean', 'count']).reset_index()
        word_groups.columns = ['word', 'avg_nll', 'count']
        word_groups = word_groups[word_groups['count'] >= min_occurrence]
        word_groups = word_groups.sort_values('avg_nll', ascending=False)
        word_stats = word_groups[['word', 'avg_nll', 'count']].values.tolist()
        top_words = word_groups.head(top_k)

        # 错误类别分类（确保 word 是字符串）
        error_categories = defaultdict(lambda: {"count": 0, "total_nll": 0.0})
        for _, row in df.iterrows():
            word = row['word']
            if pd.isna(word) or not isinstance(word, str):
                continue
            cat = classify_error(word)
            error_categories[cat]["count"] += 1
            error_categories[cat]["total_nll"] += row['avg_nll']
        cat_stats = []
        for cat, vals in error_categories.items():
            avg_nll = vals["total_nll"] / vals["count"] if vals["count"] > 0 else 0
            cat_stats.append((cat, vals["count"], avg_nll))
        cat_stats.sort(key=lambda x: x[1], reverse=True)

    # 4. 绘图
    if generate_plots:
        with TimedBlock("generate_plots", timing):
            sent_df = None
            if sentence_nll_csv and sentence_nll_csv.exists():
                sent_df = pd.read_csv(sentence_nll_csv)
            plot_nll_distribution(sent_df, output_dir, logger)
            if word_stats:
                plot_top_suspicious_words(word_stats, output_dir, top_k, min_count=5, logger=logger)

    # 5. 保存输出
    with TimedBlock("save_output", timing):
        top_words.to_csv(output_dir / "top_suspicious_words.csv", index=False)
        pd.DataFrame(cat_stats, columns=["category", "count", "avg_nll"]).to_csv(output_dir / "error_categories.csv", index=False)

        summary = {
            "total_words": total_words,
            "avg_nll": avg_nll_all,
            "median_nll": median_nll,
            "percentile_95_nll": high_thresh,
            "high_nll_words_ratio": len(high_nll_words)/total_words if total_words > 0 else 0,
            "top_suspicious_words": top_words.to_dict(orient="records"),
            "error_categories": cat_stats,
            "sample_ratio": sample_ratio,
            "model_name": model_name,
            "aggregated": need_aggregate,
            "timestamp": datetime.now().isoformat()
        }
        with open(output_dir / "summary_report.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        md_path = output_dir / "analysis_report.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# ASR转录错误分析报告\n\n")
            f.write(f"**任务名称**: {task_name}\n")
            f.write(f"**模型**: {model_name}\n")
            f.write(f"**采样比例**: {sample_ratio*100:.1f}%\n")
            f.write(f"**词级聚合**: {'是' if need_aggregate else '否（模型已是词级）'}\n")
            f.write(f"**生成时间**: {datetime.now().isoformat()}\n\n")
            f.write("## 1. 总体统计\n\n")
            f.write(f"- 总词数: {total_words}\n")
            f.write(f"- 平均NLL: {avg_nll_all:.4f}\n")
            f.write(f"- NLL中位数: {median_nll:.4f}\n")
            f.write(f"- NLL 95%分位数: {high_thresh:.4f}\n")
            f.write(f"- NLL≥95分位数的词占比: {len(high_nll_words)/total_words*100:.2f}%\n\n")
            f.write("## 2. 高频可疑词（Top 10）\n\n")
            f.write("| 词 | 平均NLL | 出现次数 |\n")
            f.write("|----|---------|----------|\n")
            for _, row in top_words.head(10).iterrows():
                f.write(f"| {row['word']} | {row['avg_nll']:.4f} | {row['count']} |\n")
            f.write("\n## 3. 错误类别分布\n\n")
            f.write("| 类别 | 出现次数 | 平均NLL |\n")
            f.write("|------|----------|---------|\n")
            for cat, cnt, avg in cat_stats:
                f.write(f"| {cat} | {cnt} | {avg:.4f} |\n")

    timing["total_sec"] = time.perf_counter() - total_start
    logger.info(f"报告已保存至 {output_dir}")
    logger.info(f"总耗时: {timing['total_sec']:.2f}s")

    # 计时历史追加
    current_timing = {
        "timestamp": datetime.now().isoformat(),
        "read_token_csv_sec": timing.get("read_token_csv", 0),
        "aggregate_words_sec": timing.get("aggregate_words", 0),
        "statistics_sec": timing.get("statistics", 0),
        "generate_plots_sec": timing.get("generate_plots", 0),
        "save_output_sec": timing.get("save_output", 0),
        "total_sec": timing["total_sec"],
        "num_words": total_words,
        "aggregated": need_aggregate
    }
    latest_info = {
        "output_dir": str(output_dir),
        "total_words": total_words,
        "avg_nll": avg_nll_all,
        "sample_ratio": sample_ratio,
        "model_name": model_name,
        "aggregated": need_aggregate,
        "timestamp": datetime.now().isoformat()
    }
    update_metadata_timing(metadata_path, "04_statistics", current_timing, latest_info)

    # 重新加载元数据更新常规字段
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    if '04_statistics' not in metadata:
        metadata['04_statistics'] = {}
    metadata['04_statistics'].update({
        "output_dir": str(output_dir),
        "total_words": total_words,
        "avg_nll": avg_nll_all,
        "sample_ratio": sample_ratio,
        "model_name": model_name,
        "aggregated": need_aggregate,
        "timestamp": datetime.now().isoformat()
    })
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("步骤04完成")

if __name__ == "__main__":
    main()