import torch

def train_model(model, train_loader, criterion, optimizer, epochs=5):
    """
    Entraîne un modèle PyTorch pour un nombre spécifié d'époques.
    
    Args:
        model: Le modèle de réseau de neurones à entraîner.
        train_loader: DataLoader contenant les données d'entraînement.
        criterion: La fonction de perte (loss).
        optimizer: L'optimiseur (ex: Adam, SGD).
        epochs: Le nombre d'époques d'entraînement.
        
    Returns:
        losses: Une liste contenant la loss cumulée de chaque époque.
    """
    losses = []
    for epoch in range(epochs):
        model.train()  # Mode entraînement (active le dropout, batchnorm, etc.)
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        losses.append(running_loss)
        print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")
    
    return losses