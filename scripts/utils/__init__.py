# scripts/utils/__init__.py
"""
utils 工具包
导出常用函数，方便其他模块导入
"""

from .logger import setup_logger
from .text_cleaner import (
    remove_speaker_prefix,
    is_valid_sentence,
    extract_sentences_from_line,
    extract_sentences_from_file,
    normalize_text
)
from .model_loader import load_model_and_tokenizer
from .nll_calculator import compute_sentence_nll_batch, compute_word_nll

__all__ = [
    'setup_logger',
    'remove_speaker_prefix',
    'is_valid_sentence',
    'extract_sentences_from_line',
    'extract_sentences_from_file',
    'normalize_text',
    'load_model_and_tokenizer',
    'compute_sentence_nll_batch',
    'compute_word_nll',
]