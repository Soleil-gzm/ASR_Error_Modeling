# scripts/utils/cache.py
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List

def get_cache_meta_path(cache_dir: Path) -> Path:
    """返回缓存元信息文件路径"""
    return cache_dir / "cache_meta.json"

def write_cache_meta(cache_dir: Path, params: Dict[str, Any], num_chunks: int) -> None:
    """保存缓存元信息，包括分块数量"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        **params,
        "num_chunks": num_chunks,
        "version": 1  # 缓存格式版本，以后升级时可增加
    }
    meta_path = get_cache_meta_path(cache_dir)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

def read_cache_meta(cache_dir: Path) -> Dict[str, Any]:
    """读取缓存元信息，若文件不存在返回空字典"""
    meta_path = get_cache_meta_path(cache_dir)
    if not meta_path.exists():
        return {}
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def is_cache_valid(cache_dir: Path, current_params: Dict[str, Any]) -> bool:
    """
    检查缓存是否有效：
    1. 目录存在且包含 cache_meta.json
    2. 元信息中的所有关键参数与 current_params 一致
    3. 所有分块文件都存在（假设分块文件命名 chunk_*.pt）
    """
    if not cache_dir.exists():
        return False
    cached_meta = read_cache_meta(cache_dir)
    if not cached_meta:
        return False

    # 比较关键参数（可根据需要增减）
    for key in current_params:
        if cached_meta.get(key) != current_params[key]:
            return False

    # 检查分块文件数量是否与记录一致
    expected_chunks = cached_meta.get("num_chunks", 0)
    chunk_files = list(cache_dir.glob("chunk_*.pt"))
    if len(chunk_files) != expected_chunks:
        return False

    # 可选：尝试读取第一个和最后一个块检查文件是否损坏
    # 这里简化，相信文件存在即完整
    return True

def invalidate_cache(cache_dir: Path) -> None:
    """删除整个缓存目录（当缓存损坏或参数不合适时调用）"""
    if cache_dir.exists():
        shutil.rmtree(cache_dir)