import torch
import torch.nn as nn
import torch.nn.functional as F
from src.transformers.attention import SelfAttention
from src.transformers.embeddings import Embeddings
import math


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, heads, mask, ff_hidden_mult=4, dropout=0.0):
        super().__init__()
        self.attention = SelfAttention(embedding_dim, heads=heads, mask=mask)
        self.mask = mask
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_hidden_mult * embedding_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * embedding_dim, embedding_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attented = self.attention(x)
        x = self.norm1(attented + x)
        x = self.dropout(x)
        feedforward = self.ff(x)
        x = self.norm2(feedforward + x)
        x = self.dropout(x)
        return x


class Encoder(nn.Module):
    """
    Transformer encoder for classifying sequences
    """

    def __init__(
        self,
        emb,
        heads,
        depth,
        seq_length,
        vocab_size,
        num_classes,
        max_pool=True,
        dropout=0.0,
    ):
        super().__init__()
        self.embed_dim = emb
        self.num_tokens = vocab_size
        self.max_pool = max_pool
        self.embeddings = Embeddings(
            embed_dim=emb, vocab_size=vocab_size, max_seq_len=seq_length
        )
        tblocks = []
        for i in range(depth):
            tblocks.append(
                EncoderLayer(
                    embedding_dim=emb,
                    heads=heads,
                    mask=False,
                    dropout=dropout,
                )
            )

        self.tblocks = nn.Sequential(*tblocks)
        self.dense = nn.Linear(emb, num_classes)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embeddings(x) * math.sqrt(self.embed_dim)
        x = self.do(x)
        x = self.tblocks(x)
        x = (
            x.max(dim=1)[0] if self.max_pool else x.mean(dim=1)
        )
        logits = self.dense(x)
        prob = F.log_softmax(logits, dim=1)
        return prob


if __name__ == "__main__":
    model = Encoder(emb=128, heads=8, depth=6, seq_length=128, vocab_size=20000, num_classes=2)
    x = torch.randint(0, 20000, (32, 128))
    print(model(x).shape)
