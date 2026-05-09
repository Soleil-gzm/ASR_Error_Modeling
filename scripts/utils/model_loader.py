# scripts/utils/model_loader.py
"""
模型加载模块
支持加载 GPT-2、Qwen、LLaMA 等因果语言模型
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_name: str, cache_dir: str = None, device_ids: list = None, trust_remote_code: bool = True):
    """
    加载模型和分词器，支持需要 trust_remote_code 的模型（如 Qwen）
    Args:
        model_name: HuggingFace 模型名称或本地路径
        cache_dir: 模型缓存目录（可选）
        device_ids: GPU ID 列表，如 [0,1] 或 [6,7]
        trust_remote_code: 是否允许执行自定义模型代码（Qwen、ChatGLM 等需要设为 True）
    Returns:
        model, tokenizer, device
    """
    # 加载 tokenizer（某些模型可能需要 use_fast=False）
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        use_fast=True
    )
    
    # 检查 tokenizer 是否加载成功，若 fast tokenizer 失败则回退到 slow
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            use_fast=False
        )
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32  # 节省显存
    )
    
    # 设置 pad_token（如果不存在）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # 同步到模型配置
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
    
    # 设备配置
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