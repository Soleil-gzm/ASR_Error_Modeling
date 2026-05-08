# scripts/utils/nll_calculator.py
"""
NLL计算模块
提供句子级NLL批处理和词级NLL的单句计算
"""

# scripts/utils/nll_calculator.py (修正版)

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class SentenceDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length=512):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        text = self.sentences[idx]
        # 关键修复：添加 padding='max_length'，使所有样本长度相同
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',          # <--- 添加此参数
            return_tensors='pt'
        )
        # 返回形状为 (max_length,) 的张量
        return enc['input_ids'][0], enc['attention_mask'][0]

def compute_sentence_nll_batch(model, tokenizer, sentences, batch_size=32,
                               max_length=512, device='cuda', desc="Computing NLL"):
    dataset = SentenceDataset(sentences, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_nll = []
    with torch.no_grad():
        for input_ids, attention_mask in tqdm(dataloader, desc=desc):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            token_nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            token_nll = token_nll.view(shift_labels.size())
            # 注意：由于 padding 的存在，attention_mask 的第 1 位后仍然有 padding 标记（值为0）
            mask = attention_mask[:, 1:].contiguous()   # 移除了第一个 [CLS] 或 BOS 吗？实际上 attention_mask 与 input_ids 对齐，第一个 token 是 [CLS]/[BOS] 也需要预测？通常我们忽略它，但这里 mask 对应 shift_labels 的位置。
            # 更严谨的做法：确保 mask 长度与 token_nll 一致
            seq_nll = (token_nll * mask).sum(dim=1) / mask.sum(dim=1)
            all_nll.extend(seq_nll.cpu().tolist())
    return all_nll

def compute_word_nll(model, tokenizer, sentence, device='cuda', max_length=512):
    """
    单句逐token的负对数似然
    添加 truncation 避免序列过长
    """
    inputs = tokenizer(
        sentence,
        return_tensors='pt',
        truncation=True,
        max_length=max_length
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        logits = outputs.logits
    
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    token_nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    token_nll = token_nll.view(shift_labels.size())
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    nll_list = [float('nan')] * len(tokens)
    for i in range(1, len(tokens)):
        nll_list[i] = token_nll[0, i-1].item()
    return tokens, nll_list


