import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from models.model import FashionCNN
from utils.train import train_model
from utils.test import test_model

# Labels Fashion-MNIST
class_names = ['T-shirt', 'Pantalon', 'Pull', 'Robe', 'Manteau',
               'Sandale', 'Chemise', 'Sneaker', 'Sac', 'Bottine']

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Datasets
train_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform
)

# DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=64, shuffle=False)

# Modèle
model     = FashionCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraînement
losses = train_model(model, train_loader, criterion, optimizer, epochs=5)

# Courbe de loss
plt.plot(losses)
plt.title('Courbe de Loss - Fashion-MNIST CNN')
plt.xlabel('Époque')
plt.ylabel('Loss')
plt.savefig('loss_curve_fashion.png')
plt.show()

# Évaluation
test_model(model, test_loader)

# Images mal classées
model.eval()
wrong_images, wrong_preds, wrong_labels = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        for i in range(len(labels)):
            if predicted[i] != labels[i]:
                wrong_images.append(images[i])
                wrong_preds.append(predicted[i].item())
                wrong_labels.append(labels[i].item())

num_to_show = min(10, len(wrong_images))
if num_to_show > 0:
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for idx, ax in enumerate(axes.flat):
        if idx < num_to_show:
            ax.imshow(wrong_images[idx].squeeze(), cmap='gray')
            ax.set_title(f'Prédit: {class_names[wrong_preds[idx]]}\nRéel: {class_names[wrong_labels[idx]]}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('erreurs_fashion.png')
    plt.show()
else:
    print("Aucune erreur de classification trouvée à afficher !")

# Sauvegarde
torch.save(model.state_dict(), 'fashion_cnn.pth')
print("Modèle sauvegardé !")