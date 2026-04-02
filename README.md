# Cat Classifier - Régression Logistique

## Description

Ce projet implémente une application web de classification d'images utilisant la régression logistique vue comme un réseau de neurones. L'objectif est de classifier les images comme contenant un chat ou non-chat de manière automatique.

## Objectifs

- Implémenter un algorithme de régression logistique avec approche réseau de neurones
- Créer une interface web interactive pour les prédictions
- Développer une API REST pour l'intégration
- Comprendre les fondamentaux du machine learning

## Structure du projet

```
logistic-regression-neural-network/
├── app.py                          # Application Flask principale
├── requirements.txt                 # Dépendances Python
├── README.md                       # Documentation du projet
├── .gitignore                      # Fichiers ignorés par Git
├── templates/                      # Templates HTML
│   ├── index.html                  # Page d'accueil moderne
│   ├── demo.html                   # Galerie de démo interactive
│   └── docs.html                   # Documentation technique
├── static/
│   └── images/                    # Images pour la démo
│       ├── my_image.jpg
│       ├── image1.png
│       ├── image2.png
│       └── ...
└── Logistic Regression as a Neural Network/
    ├── Logistic.ipynb              # Notebook Jupyter du TP
    ├── lr_utils.py                 # Fonctions utilitaires
    └── datasets/                   # Données d'entraînement
        ├── train_catvnoncat.h5
        └── test_catvnoncat.h5
```

## Installation

### Prérequis

- Python 3.7+
- pip (gestionnaire de paquets Python)

### Étapes d'installation

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

## Utilisation

### Interface web

1. **Page d'accueil** : Upload d'images pour prédiction
2. **Page de démo** : Tests avec images exemples
3. **Documentation** : Guide technique complet

### API REST

#### Endpoint principal
```bash
curl -X POST -F "image=@votre_image.jpg" http://localhost:5000/predict
```

#### Réponse attendue
```json
{
  "success": true,
  "prediction": "cat",
  "confidence": 0.85,
  "image": "base64_encoded_image"
}
```

## Performances

### Métriques du modèle

- **Accuracy (entraînement)**: ~99%
- **Accuracy (test)**: ~70%
- **Dataset**: 209 images train, 50 images test
- **Temps de prédiction**: <100ms
- **Taille du modèle**: ~100KB

### Caractéristiques techniques

- **Input**: Images 64×64 pixels RGB
- **Output**: Classification binaire (chat/non-chat)
- **Algorithme**: Régression logistique
- **Framework**: NumPy + Flask

## 🔧 Technologies utilisées

### Backend
- **Python 3.7+** : Langage principal
- **Flask** : Framework web
- **NumPy** : Calculs numériques
- **SciPy** : Fonctions scientifiques
- **PIL/Pillow** : Traitement d'images

### Frontend
- **HTML5** : Structure des pages
- **Bootstrap 5** : Framework CSS responsive
- **JavaScript** : Interactivité côté client

### Machine Learning
- **NumPy** : Opérations matricielles
- **Matplotlib** : Visualisations
- **h5py** : Gestion des datasets

## 🌐 Endpoints de l'API

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Page d'accueil avec interface de prédiction |
| POST | `/predict` | Classifier une image |
| GET | `/demo` | Page de démonstration interactive |
| GET | `/docs` | Documentation technique complète |
| GET | `/api/info` | Informations sur l'API |

## 📖 Documentation complète

Pour une documentation technique détaillée, consultez :
- **[Documentation en ligne](http://localhost:5000/docs)** une fois l'application lancée
- **[Notebook Jupyter](Logistic%20Regression%20as%20a%20Neural%20Network/Logistic.ipynb)** pour les détails d'implémentation

## 🧪 Tests

### Tests manuels

1. Upload d'images de chats
2. Upload d'images sans chats  
3. Tests avec différents formats (JPG, PNG)
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

## Déploiement

### Développement
```bash
python app.py
```

### Production (recommandé)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Dépannage

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

## Licence

Ce projet est développé dans un cadre pédagogique.

## Auteur

Développé dans le cadre d'un TP de Machine Learning sur la régression logistique.

## Liens utiles

- **[Dépôt GitHub](https://github.com/leadylearn/logistic-regression-neural-network)**
- **[Démo interactive](https://github.com/leadylearn/logistic-regression-neural-network#d%C3%A9mo-en-direct)** 
- **[Documentation technique](https://github.com/leadylearn/logistic-regression-neural-network#documentation)**

### Lancement rapide

Après avoir cloné le projet :

```bash
# Lancer l'application
python app.py

# Accéder aux interfaces :
# Page d'accueil : http://localhost:5000
# Démo interactive : http://localhost:5000/demo  
# Documentation : http://localhost:5000/docs
# API info : http://localhost:5000/api/info
```

---

**Note** : Ce projet est une démonstration éducative des concepts de machine learning.
