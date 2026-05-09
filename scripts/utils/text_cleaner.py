# scripts/utils/text_cleaner.py
"""
文本清洗模块
提供去除说话人前缀、按标点分句、过滤短句等功能
"""

import re
from pathlib import Path
import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw = f.read(10000)
    return chardet.detect(raw)['encoding']   

def remove_speaker_prefix(text: str) -> str:
    """
    去除行首的“说话人X:”或“说话人X：”前缀
    """
    return re.sub(r'^说话人\d+[：:]', '', text).strip()

def is_valid_sentence(text: str, min_len: int = 3) -> bool:
    """
    判断是否为有效句子（非空且长度足够）
    """
    return bool(text) and len(text.strip()) >= min_len

def extract_sentences_from_line(line: str, remove_prefix: bool = True, split_by_punct: bool = False) -> list:
    """
    从一行文本中提取句子列表
    Args:
        line: 原始行
        remove_prefix: 是否去除“说话人X:”前缀
        split_by_punct: 是否根据标点符号分句（默认按整行）
    Returns:
        句子列表（已去除空字符串）
    """
    if remove_prefix:
        line = remove_speaker_prefix(line)
    line = line.strip()
    if not line:
        return []
    if split_by_punct:
        # 按句号、问号、感叹号、分号分割
        parts = re.split(r'[。？!；]', line)
        sentences = [p.strip() for p in parts if p.strip()]
    else:
        sentences = [line]
    return sentences

    # encoding = detect_encoding(file_path)
    # with open(file_path, 'r', encoding=encoding, errors='replace') as f:
    #     if remove_prefix:
    #         line = remove_speaker_prefix(line)
    #     line = line.strip()
    #     if not line:
    #         return []
    #     if split_by_punct:
    #         # 按句号、问号、感叹号、分号分割
    #         parts = re.split(r'[。？!；]', line)
    #         sentences = [p.strip() for p in parts if p.strip()]
    #     else:
    #         sentences = [line]
    # return sentences

def extract_sentences_from_file(file_path: str, remove_prefix: bool = True,
                                split_by_punct: bool = False, min_len: int = 3) -> list:
    """
    从txt文件中提取所有有效句子
    Returns:
        句子字符串列表
    """
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            segs = extract_sentences_from_line(line, remove_prefix, split_by_punct)
            for s in segs:
                if is_valid_sentence(s, min_len):
                    sentences.append(s)
    return sentences

def normalize_text(text: str) -> str:
    """
    可选：文本标准化（如全角转半角、去除多余空格等）
    """
    # 全角数字/字母转半角（示例）
    text = re.sub(r'［', '[', text)
    text = re.sub(r'］', ']', text)
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip()