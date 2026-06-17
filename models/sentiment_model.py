import torch.nn as nn

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=64):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.fc = nn.Linear(emb_dim, 1)

    def forward(self, x):
        # x : [batch, seq_len]
        embedded = self.embedding(x)        # [batch, seq_len, emb_dim]
        pooled   = embedded.mean(dim=1)     # [batch, emb_dim] moyenne
        out      = self.fc(pooled)          # [batch, 1]
        return out