import torch
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=50):
        self.labels = labels
        self.max_len = max_len
        self.vocab = vocab
        self.data = [self.encode(t) for t in texts]

    def encode(self, text):
        tokens = text.lower().split()[:self.max_len]
        ids = [self.vocab.get(w, 1) for w in tokens]  # 1 = <UNK>
        # Padding
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], torch.tensor(self.labels[idx], dtype=torch.float)