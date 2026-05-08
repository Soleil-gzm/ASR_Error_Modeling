# scripts/utils/logger.py
"""
日志配置模块
提供统一的 setup_logger 函数，支持文件和控制台双输出
"""

import logging
from pathlib import Path

def setup_logger(log_dir, name, level=logging.DEBUG, console_level=logging.INFO):
    """
    设置日志记录器
    Args:
        log_dir: 日志文件目录（Path对象或字符串）
        name: 日志记录器名称（通常为脚本名或任务名）
        level: 文件日志级别
        console_level: 控制台日志级别
    Returns:
        logging.Logger 实例
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # 总开关设为DEBUG
    if logger.handlers:
        logger.handlers.clear()

    # 文件处理器
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(level)
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(console_level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger