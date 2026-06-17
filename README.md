# Projet Deep Learning (PyTorch) - EMSI

Ce dépôt contient le projet de Deep Learning réalisé dans le cadre du cursus d'ingénieur à l'**École Marocaine des Sciences de l'Ingénieur (EMSI)**. Il met en œuvre deux modèles d'apprentissage profond pour résoudre des tâches de vision par ordinateur (classification d'images) et de traitement du langage naturel (NLP - analyse de sentiments).

---

## 📌 Présentation et Objectifs

Le projet est divisé en deux applications distinctes :
1. **Classification d'images (Fashion-MNIST)** : Entraînement d'un réseau de neurones convolutif (CNN) pour classifier des images d'articles de mode en 10 classes distinctes (vêtements, chaussures, sacs, etc.).
2. **Analyse de sentiments (IMDb Movie Reviews)** : Entraînement d'un modèle d'embedding avec pooling moyen pour classifier des critiques de films comme positives ou négatives.

---

## 📊 Jeux de Données (Datasets)

*   **Fashion-MNIST** : 70 000 images de $28 \times 28$ pixels en niveaux de gris (60 000 pour l'entraînement et 10 000 pour le test) représentant 10 catégories de vêtements et d'accessoires.
*   **IMDb Movie Reviews** : 50 000 critiques de films hautement polarisées (25 000 pour l'entraînement et 25 000 pour le test) pour de la classification binaire (Positif / Négatif).

---

## 🛠️ Technologies Utilisées

*   **Langage** : Python 3
*   **Framework de Deep Learning** : PyTorch (Tensors, Autograd, NN module)
*   **Traitement de Données NLP** : Hugging Face `datasets`
*   **Évaluation et Métriques** : Scikit-learn
*   **Visualisation graphique** : Matplotlib

---

## 🧠 Architecture des Modèles

### 1. Modèle de Vision : `FashionCNN`
Localisé dans [`models/model.py`](models/model.py) :
*   **Couche de Convolution 1** : 1 canal d'entrée, 32 filtres de taille $3 \times 3$, padding = 1.
*   **Activation & Pooling** : Activation ReLU + MaxPool2d ($2 \times 2$, stride = 2).
*   **Couche de Convolution 2** : 32 canaux d'entrée, 64 filtres de taille $3 \times 3$, padding = 1.
*   **Activation & Pooling** : Activation ReLU + MaxPool2d ($2 \times 2$, stride = 2).
*   **Flattening** : Aplatissement du tenseur en vecteur de taille 3136.
*   **Couche Dense 1** : Couche linéaire de $3136 \to 128$ neurones avec activation ReLU.
*   **Régularisation** : Dropout à 25%.
*   **Couche Dense 2 (Sortie)** : Couche linéaire de $128 \to 10$ neurones pour les scores de classe.

### 2. Modèle NLP : `SentimentModel`
Localisé dans [`models/sentiment_model.py`](models/sentiment_model.py) :
*   **Couche d'Embedding** : Taille du vocabulaire (10 002) $\times$ 64 dimensions.
*   **Global Average Pooling** : Calcul de la moyenne temporelle des embeddings pour obtenir un vecteur représentatif.
*   **Couche Linéaire (Sortie)** : $64 \to 1$ neurone produisant la log-probabilité du sentiment.

---

## 📈 Résultats Obtenus

### Vision (Fashion-MNIST)
*   **Accuracy sur l'ensemble de test** : **91.13%**
*   **Visualisation des erreurs** : Sauvegardée sous `erreurs_fashion.png` pour analyser les confusions du modèle (ex: Pull vs Manteau, Sandale vs Sneaker).

### NLP (IMDb)
*   **Accuracy sur l'ensemble de test** : **78%**
*   **Rapport de performance** :
    ```
              precision    recall  f1-score   support

    Négatif       0.77      0.79      0.78     12500
    Positif       0.78      0.77      0.77     12500
    ```

Les courbes d'apprentissage de chaque entraînement sont disponibles dans les fichiers `loss_curve_fashion.png` et `loss_sentiment.png`.

---

## 🚀 Instructions d'Installation et d'Exécution

### 1. Installation des dépendances
Clonez le dépôt, puis installez les paquets requis à l'aide de :
```bash
pip install -r requirements.txt
```

### 2. Exécution du modèle de vision (Fashion-MNIST)
Pour lancer l'entraînement, l'évaluation et la visualisation des erreurs de Fashion-MNIST, exécutez :
```bash
python main.py
```

### 3. Exécution du modèle NLP (IMDb)
Pour charger les données IMDb, entraîner le modèle, afficher le rapport de classification et tester des exemples de prédictions, exécutez :
```bash
python main_sentiment.py
```

---

## ✍️ Auteur

**Abderrahmane Elfarouki**  
Élève Ingénieur à l'École Marocaine des Sciences de l'Ingénieur (EMSI).
