import torch

def test_model(model, loader):
    """
    Évalue les performances du modèle sur un DataLoader de test.
    
    Args:
        model: Le modèle à évaluer.
        loader: Le DataLoader contenant les données de test.
        
    Returns:
        accuracy: Le pourcentage de prédictions correctes.
    """
    model.eval()  # Mode évaluation (désactive le dropout, batchnorm, etc.)
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy