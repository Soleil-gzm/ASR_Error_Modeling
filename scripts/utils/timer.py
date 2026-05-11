# scripts/utils/timer.py
import time
import json
from pathlib import Path
from contextlib import contextmanager

@contextmanager
def TimedBlock(name, timing_dict):
    """上下文管理器，记录代码块耗时并存储到 timing_dict[name]"""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    timing_dict[name] = elapsed

def update_metadata_timing(metadata_path, step_name, timing_dict, latest_info=None):
    """
    将计时信息追加到 run_metadata.json 中对应步骤的 timing_history
    Args:
        metadata_path: Path 对象
        step_name: 字符串，如 '00_preprocess'
        timing_dict: 包含计时字段的字典（应包含 'timestamp'）
        latest_info: 可选，最新一次运行的关键配置（会存入 'latest' 字段）
    """
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

    if step_name not in metadata:
        metadata[step_name] = {}
    if 'timing_history' not in metadata[step_name]:
        metadata[step_name]['timing_history'] = []
    metadata[step_name]['timing_history'].append(timing_dict)

    if latest_info:
        metadata[step_name]['latest'] = latest_info

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)