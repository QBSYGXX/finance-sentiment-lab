"""TextCNN adapted from 649453932/Chinese-Text-Classification-Pytorch.

Copyright (c) 2019 huwenxing, MIT. See third_party/LICENSE.TextCNN.
Changes: smaller dimensions, tensor input, padding-window masking, train-only vocabulary.
"""
from collections import Counter
import torch
from torch import nn
from torch.nn import functional as F
from common import MAX_LENGTH


# 模块：字符词表；核心业务：是。
# 词表仅来自训练集；0 表示补齐，1 表示未见字符。
def build_vocab(texts):
    counts = Counter("".join(texts))
    chars = sorted(char for char, count in counts.items() if count >= 2)
    return {char: i + 2 for i, char in enumerate(chars)}


def encode(texts, vocab):
    rows = [[vocab.get(char, 1) for char in text[:MAX_LENGTH]] for text in texts]
    return torch.tensor([row + [0] * (MAX_LENGTH - len(row)) for row in rows], dtype=torch.long)


# 模块：轻量卷积分类；核心业务：是。
# 不同卷积窗口提取局部短语特征；屏蔽完全位于补齐区的窗口。
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=96, filters=64, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.kernels = (2, 3, 4)
        self.convs = nn.ModuleList([nn.Conv2d(1, filters, (k, embedding_dim)) for k in self.kernels])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(filters * len(self.kernels), 3)

    def forward(self, tokens):
        embedded = self.embedding(tokens).unsqueeze(1)
        lengths = tokens.ne(0).sum(1)
        pooled = []
        for kernel, conv in zip(self.kernels, self.convs):
            features = F.relu(conv(embedded)).squeeze(3)
            valid_windows = (lengths - kernel + 1).clamp(min=1)
            positions = torch.arange(features.shape[2], device=tokens.device)
            invalid = positions.unsqueeze(0) >= valid_windows.unsqueeze(1)
            features = features.masked_fill(invalid.unsqueeze(1), torch.finfo(features.dtype).min)
            pooled.append(features.amax(dim=2))
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))

