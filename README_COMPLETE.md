# 🐱 Cat Classifier - Régression Logistique

Application web de classification d'images utilisant la régression logistique vue comme un réseau de neurones.

## 📖 Description

Ce projet démontre l'implémentation complète d'un système de machine learning pour classifier les images comme contenant un chat ou non-chat. L'application combine un algorithme de régression logistique avec une interface web Flask pour créer une démo interactive.

## 🎯 Objectifs Pédagogiques

- Comprendre les fondamentaux de la régression logistique
- Implémenter un réseau de neurones à une couche
- Créer une API REST avec Flask
- Développer une interface web interactive
- Apprendre le déploiement d'applications ML

## 🏗️ Architecture

```
Frontend (HTML/Bootstrap) ←→ Flask API ←→ Modèle ML (NumPy)
```

### Composants principaux

1. **🧠 Modèle Machine Learning**
   - Régression logistique avec approche réseau de neurones
   - Entraînement sur dataset Cats vs Non-cats
   - Prétraitement d'images 64×64 pixels

2. **🔧 API Flask**
   - Endpoints REST pour les prédictions
   - Gestion des uploads d'images
   - Interface web responsive

3. **🌐 Interface Utilisateur**
   - Upload drag-and-drop d'images
   - Visualisation des prédictions en temps réel
   - Page de démo interactive

## 🚀 Démarrage Rapide

### Prérequis

- Python 3.7+
- pip (gestionnaire de paquets Python)

### Installation

1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/leadylearn/logistic-regression-neural-network.git
   cd logistic-regression-neural-network
   ```

2. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

3. **Lancer l'application**
   ```bash
   python app.py
   ```

4. **Accéder à l'application**
   Ouvrez votre navigateur sur [http://localhost:5000](http://localhost:5000)

## 📁 Structure du Projet

```
logistic-regression-neural-network/
├── app.py                          # Application Flask principale
├── requirements.txt                 # Dépendances Python
├── README.md                       # Documentation
├── templates/                      # Templates HTML
│   ├── index.html                  # Page d'accueil
│   ├── demo.html                   # Page de démo
│   └── docs.html                   # Documentation technique
├── Logistic Regression as a Neural Network/
│   ├── Logistic.ipynb              # Notebook Jupyter du TP
│   ├── lr_utils.py                 # Fonctions utilitaires
│   ├── datasets/                   # Données d'entraînement
│   │   ├── train_catvnoncat.h5
│   │   └── test_catvnoncat.h5
│   └── images/                     # Images de test
│       ├── my_image.jpg
│       ├── image1.png
│       └── ...
└── images/                         # Images pour l'interface web
    └── (copiées depuis le dossier TP)
```

## 🔌 API Reference

### Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Page d'accueil avec interface de prédiction |
| POST | `/predict` | Classifier une image |
| GET | `/demo` | Page de démonstration interactive |
| GET | `/docs` | Documentation technique complète |
| GET | `/api/info` | Informations sur l'API |

### Exemple d'utilisation

```bash
# Avec curl
curl -X POST -F "image=@chat.jpg" http://localhost:5000/predict

# Réponse attendue
{
  "success": true,
  "prediction": "cat",
  "confidence": 0.85,
  "image": "base64_encoded_image"
}
```

## 📊 Performance

### Métriques du modèle

- **Accuracy (entraînement)**: ~99%
- **Accuracy (test)**: ~70%
- **Dataset**: 209 images train, 50 images test
- **Temps de prédiction**: <100ms
- **Taille du modèle**: ~100KB

### Limitations

- 🔶 Overfitting (accuracy train >> accuracy test)
- 🖼️ Résolution limitée (uniquement 64×64 pixels)
- 🎯 Classification binaire (chat/non-chat uniquement)
- 📊 Dataset relativement petit

## 🧮 Théorie Mathématique

### Régression Logistique

Le modèle utilise la régression logistique vue comme un réseau de neurones :

#### Forward Propagation
```
z = w^T x + b
a = σ(z) = 1 / (1 + e^(-z))
```

#### Fonction de coût
```
J = -(1/m) Σ[y^(i) log(a^(i)) + (1-y^(i)) log(1-a^(i))]
```

#### Gradient Descent
```
w = w - α ∂J/∂w
b = b - α ∂J/∂b
```

## 🛠️ Technologies Utilisées

### Backend
- **Python 3.7+**: Langage principal
- **Flask**: Framework web
- **NumPy**: Calculs numériques
- **SciPy**: Fonctions scientifiques
- **PIL/Pillow**: Traitement d'images

### Frontend
- **HTML5**: Structure des pages
- **Bootstrap 5**: Framework CSS responsive
- **JavaScript**: Interactivité côté client

### Machine Learning
- **NumPy**: Opérations matricielles
- **Matplotlib**: Visualisations
- **h5py**: Gestion des datasets

## 🎮 Fonctionnalités

### Interface Principale
- 📤 Upload d'images (drag-and-drop)
- 🖼️ Prévisualisation des images
- 🤖 Prédiction en temps réel
- 📊 Affichage de la confiance

### Page de Démo
- 🎯 Tests avec images exemples
- 📈 Statistiques des prédictions
- 🔄 Comparaison multiple
- 📊 Visualisation des résultats

### Documentation
- 📖 Guide complet
- 🔧 Référence API
- 💡 Exemples de code
- 🚀 Instructions de déploiement

## 🧪 Tests et Validation

### Tests manuels
1. Upload d'images de chats
2. Upload d'images sans chats
3. Tests avec différents formats
4. Validation des réponses API

### Tests automatisés
```bash
# Tester l'API
python -c "
import requests
response = requests.get('http://localhost:5000/api/info')
print('API Status:', response.status_code)
print('Response:', response.json())
"
```

## 🚀 Déploiement

### Développement
```bash
python app.py
```

### Production (recommandé)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker (optionnel)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## 🔧 Dépannage

### Problèmes courants

**ModuleNotFoundError**
```bash
pip install -r requirements.txt
```

**Port déjà utilisé**
```bash
# Changer le port dans app.py
app.run(port=5001)
```

**Modèle non chargé**
```bash
# Vérifier la présence des datasets
ls "Logistic Regression as a Neural Network/datasets/"
```

## 🤝 Contribution

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amélioration`)
3. Commit les changements (`git commit -am 'Ajout nouvelle fonctionnalité'`)
4. Push vers la branche (`git push origin feature/amélioration`)
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est développé dans un cadre pédagogique. 

## 👨‍🏫 Auteur

Développé dans le cadre d'un TP de Machine Learning sur la régression logistique.

## 🔗 Liens Utiles

- [Dépôt GitHub](https://github.com/leadylearn/logistic-regression-neural-network)
- [Documentation en ligne](http://localhost:5000/docs)
- [Démo interactive](http://localhost:5000/demo)

---

**Note**: Ce projet est une démonstration éducative des concepts de machine learning et n'est pas destiné à un usage en production.
