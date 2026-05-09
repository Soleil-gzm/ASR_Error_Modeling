# scripts/utils/nll_calculator.py
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
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
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
            logits = outputs.logits     # 获取模型输出的 logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')     #损失函数内部会自动对 logits 做 softmax，然后计算负对数似然（NLL）
            token_nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            token_nll = token_nll.view(shift_labels.size())
            mask = attention_mask[:, 1:].contiguous()
            seq_nll = (token_nll * mask).sum(dim=1) / mask.sum(dim=1)
            all_nll.extend(seq_nll.cpu().tolist())
    return all_nll

def compute_word_nll(model, tokenizer, sentence, device='cuda', max_length=512):
    """
    单句逐token的负对数似然，支持 Qwen、GPT-2 等所有模型，中文输出完全正常
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
    
    # 关键修复：使用 decode 获得原始字符串（而不是 convert_ids_to_tokens）
    token_ids = input_ids[0].tolist()
    tokens = []
    for idx, tid in enumerate(token_ids):
        # 跳过特殊 token（如 <s>, </s>）可以根据需要保留，这里全部转换
        token_str = tokenizer.decode([tid], skip_special_tokens=False).strip()
        # 如果解码结果为空或仅为空格，尝试使用 convert_ids_to_tokens 作为后备
        if not token_str:
            token_str = tokenizer.convert_ids_to_tokens(tid)
            if isinstance(token_str, bytes):
                token_str = token_str.decode('utf-8', errors='replace')
        tokens.append(token_str)
    
    nll_list = [float('nan')] * len(tokens)
    for i in range(1, len(tokens)):
        nll_list[i] = token_nll[0, i-1].item()
    return tokens, nll_list