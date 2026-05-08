# scripts/utils/model_loader.py
"""
模型加载模块
加载预训练GPT-2中文模型，支持单卡或多卡DataParallel
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_name: str, cache_dir: str = None, device_ids: list = None):
    """
    加载模型和分词器，并配置多GPU包装
    Args:
        model_name: HuggingFace模型名称
        cache_dir: 模型缓存目录（可选）
        device_ids: GPU ID列表，如 [0,1] 或 [6,7]
    Returns:
        model, tokenizer, device
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

    # 添加 pad_token（如果不存在）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 确定设备
    if device_ids and torch.cuda.is_available():
        device = torch.device(f"cuda:{device_ids[0]}")
        model = model.to(device)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        device = torch.device("cpu")
        model = model.to(device)

    model.eval()
    return model, tokenizer, device