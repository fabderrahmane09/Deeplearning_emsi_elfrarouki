import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.datasets import fetch_openml
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from models.sentiment_model import SentimentModel
from utils.dataset import SentimentDataset

# ── 1. Données IMDb via datasets ──────────────────────────────
from datasets import load_dataset

print("Chargement du dataset IMDb...")
raw = load_dataset("imdb")

train_texts  = raw["train"]["text"]
train_labels = raw["train"]["label"]
test_texts   = raw["test"]["text"]
test_labels  = raw["test"]["label"]

# ── 2. Vocabulaire ────────────────────────────────────────────
print("Construction du vocabulaire...")
counter = Counter()
for text in train_texts:
    counter.update(text.lower().split())

# Garder les 10000 mots les plus fréquents
vocab = {word: idx+2 for idx, (word, _) in enumerate(counter.most_common(10000))}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1
vocab_size = len(vocab) + 2

# ── 3. Datasets & DataLoaders ─────────────────────────────────
train_dataset = SentimentDataset(train_texts, train_labels, vocab)
test_dataset  = SentimentDataset(test_texts,  test_labels,  vocab)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

# ── 4. Modèle ─────────────────────────────────────────────────
model     = SentimentModel(vocab_size=vocab_size, emb_dim=64)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ── 5. Entraînement ───────────────────────────────────────────
losses = []
for epoch in range(5):
    model.train()
    running_loss = 0.0
    for texts, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(texts).squeeze(1)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    losses.append(running_loss)
    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

# ── 6. Courbe de loss ─────────────────────────────────────────
plt.plot(losses)
plt.title('Courbe de Loss - Sentiment IMDb')
plt.xlabel('Époque')
plt.ylabel('Loss')
plt.savefig('loss_sentiment.png')
plt.show()

# ── 7. Évaluation ─────────────────────────────────────────────
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for texts, labels in test_loader:
        outputs = model(texts).squeeze(1)
        preds   = (torch.sigmoid(outputs) > 0.5).long()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.long().tolist())

print(classification_report(all_labels, all_preds,
      target_names=["Négatif", "Positif"]))

# ── 8. Fonction de prédiction ─────────────────────────────────
def predict(text):
    model.eval()
    tokens = text.lower().split()[:50]
    ids    = [vocab.get(w, 1) for w in tokens]
    ids   += [0] * (50 - len(ids))
    tensor = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        output = torch.sigmoid(model(tensor)).item()
    label = "Positif 😊" if output > 0.5 else "Négatif 😞"
    print(f"\nTexte : {text}")
    print(f"Résultat : {label} ({output:.2f})")

# Test rapide
predict("This movie was absolutely amazing!")
predict("Terrible film, waste of time.")

# ── 9. Sauvegarde ─────────────────────────────────────────────
torch.save(model.state_dict(), 'sentiment_model.pth')
print("\nModèle sauvegardé !")