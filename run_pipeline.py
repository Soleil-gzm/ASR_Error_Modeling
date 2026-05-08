#!/usr/bin/env python3
"""
ASR错误分析流水线 - 配置驱动，支持任务隔离、断点续跑
"""
import os, sys, yaml, json, argparse, subprocess, logging
from pathlib import Path
from datetime import datetime

# ---------- 日志 ----------
def setup_logger(task_dir, log_level="INFO"):
    log_dir = task_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger("Pipeline")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, log_level))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # 替换路径中的占位符 {task_name} 和 {timestamp}
    task_name = config['task_name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    def rec_replace(obj):
        if isinstance(obj, dict):
            return {k: rec_replace(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [rec_replace(item) for item in obj]
        elif isinstance(obj, str):
            return obj.format(task_name=task_name, timestamp=timestamp)
        else:
            return obj
    return rec_replace(config)

def get_step_done_flag(task_dir, step_name):
    return task_dir / f".step_{step_name}_done"

def is_step_completed(task_dir, step_name):
    return get_step_done_flag(task_dir, step_name).exists()

def mark_step_done(task_dir, step_name):
    get_step_done_flag(task_dir, step_name).touch()

def run_step(step_config, step_key, task_dir, global_config, logger):
    if not step_config.get('enabled', True):
        logger.info(f"步骤 {step_key} 已禁用，跳过")
        return True
    if global_config.get('resume', False) and is_step_completed(task_dir, step_key):
        logger.info(f"步骤 {step_key} 已完成，跳过（断点续跑）")
        return True

    logger.info(f"开始执行步骤: {step_key}")
    script_path = Path(step_config['script'])
    if not script_path.exists():
        logger.error(f"脚本不存在: {script_path}")
        return False

    # 构建命令行，通过 --config_json 传递完整配置（各子脚本需支持）
    cmd = [sys.executable, str(script_path), '--config_json', json.dumps(global_config)]
    # 可以为特定步骤添加额外参数（如batch_size），但也可以用config_json统一
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"步骤 {step_key} 执行失败，返回码 {result.returncode}")
        error_log = task_dir / "logs" / f"{step_key}_error.log"
        error_log.parent.mkdir(exist_ok=True)
        with open(error_log, 'w') as f:
            f.write(result.stderr)
        logger.error(f"详细错误保存至 {error_log}")
        return False
    logger.info(f"步骤 {step_key} 执行成功")
    mark_step_done(task_dir, step_key)
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--step", help="单独运行某步骤，如00_preprocess")
    args = parser.parse_args()

    config = load_config(args.config)
    task_name = config['task_name']
    base_dir = Path(config['paths']['output']['base_dir'])
    task_dir = base_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    # 日志
    logger = setup_logger(task_dir, config['logging']['level'])
    logger.info(f"任务启动: {task_name}")
    logger.info(f"工作目录: {task_dir}")
    logger.info(f"断点续跑: {config.get('resume', False)}")

    # 定义步骤顺序（与配置文件中的steps键名一致）
    steps_order = ['00_preprocess', '01_compute_sentence_nll', '02_filter_high_nll', '03_compute_word_nll', '04_statistics']

    if args.step:
        if args.step not in config['steps']:
            logger.error(f"未知步骤: {args.step}")
            sys.exit(1)
        steps_order = [args.step]

    for step_key in steps_order:
        step_cfg = config['steps'].get(step_key)
        if not step_cfg:
            logger.warning(f"配置中缺少步骤 {step_key}，跳过")
            continue
        success = run_step(step_cfg, step_key, task_dir, config, logger)
        if not success:
            logger.error(f"流水线终止于步骤 {step_key}")
            sys.exit(1)

    logger.info("流水线执行完毕")

if __name__ == "__main__":
    main()