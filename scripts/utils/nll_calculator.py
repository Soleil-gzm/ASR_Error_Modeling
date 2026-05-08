# scripts/utils/nll_calculator.py
"""
NLL计算模块
提供句子级NLL批处理和词级NLL的单句计算
"""

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
        enc = self.tokenizer(text, max_length=self.max_length, truncation=True, return_tensors='pt')
        return enc['input_ids'][0], enc['attention_mask'][0]

def compute_sentence_nll_batch(model, tokenizer, sentences, batch_size=32,
                               max_length=512, device='cuda', desc="Computing NLL"):
    """
    批量计算句子平均负对数似然
    Returns:
        nll_list: list of float
    """
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
            mask = attention_mask[:, 1:].contiguous()
            seq_nll = (token_nll * mask).sum(dim=1) / mask.sum(dim=1)
            all_nll.extend(seq_nll.cpu().tolist())
    return all_nll

def compute_word_nll(model, tokenizer, sentence, device='cuda'):
    """
    单句逐token的负对数似然
    Returns:
        tokens: list of str (长度 = token个数)
        nlls: list of float (长度 = token个数，第一个token的NLL为NaN)
    """
    inputs = tokenizer(sentence, return_tensors='pt')
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
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])          # 所有token
    nll_list = [float('nan')] * len(tokens)                        # 占位
    for i in range(1, len(tokens)):
        nll_list[i] = token_nll[0, i-1].item()
    return tokens, nll_list