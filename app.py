from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import sys
import os

# Ajouter le chemin du TP
sys.path.append('Logistic Regression as a Neural Network')

# Importer les fonctions du TP
from lr_utils import load_dataset
from Logistic import *

app = Flask(__name__)

# Charger le modèle entraîné (vous devrez l'entraîner d'abord)
model_data = None

def load_model():
    """Charger le modèle entraîné"""
    global model_data
    try:
        # Charger les données
        train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
        
        # Prétraiter les données
        train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
        test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
        train_set_x = train_set_x_flatten / 255.
        test_set_x = test_set_x_flatten / 255.
        
        # Entraîner le modèle
        d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=False)
        
        model_data = {
            'w': d['w'],
            'b': d['b'],
            'classes': classes,
            'num_px': train_set_x_orig.shape[1]
        }
        
        print("✅ Modèle chargé avec succès!")
        return True
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        return False

@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    """API pour prédire si une image contient un chat"""
    try:
        # Récupérer l'image uploadée
        if 'image' not in request.files:
            return jsonify({'error': 'Aucune image fournie'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Nom de fichier vide'}), 400
        
        # Lire et traiter l'image
        img = Image.open(file.stream)
        img = img.convert('RGB')  # S'assurer que c'est en RGB
        
        # Redimensionner à 64x64
        img_resized = img.resize((64, 64))
        
        # Convertir en numpy array et normaliser
        img_array = np.array(img_resized) / 255.
        
        # Aplatir l'image pour la prédiction
        img_flatten = img_array.reshape((1, 64*64*3)).T
        
        # Faire la prédiction
        if model_data is None:
            return jsonify({'error': 'Modèle non chargé'}), 500
        
        prediction = predict(model_data['w'], model_data['b'], img_flatten)
        prediction_prob = sigmoid(np.dot(model_data['w'].T, img_flatten) + model_data['b'])[0][0]
        
        # Déterminer la classe
        predicted_class = int(np.squeeze(prediction))
        class_name = model_data['classes'][predicted_class].decode("utf-8")
        
        # Convertir l'image en base64 pour l'affichage
        img_buffer = io.BytesIO()
        img_resized.save(img_buffer, format='PNG')
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        return jsonify({
            'prediction': class_name,
            'confidence': float(prediction_prob),
            'image': img_str,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/demo')
def demo():
    """Page de démonstration avec images exemples"""
    return render_template('demo.html')

@app.route('/docs')
def documentation():
    """Page de documentation"""
    return render_template('docs.html')

@app.route('/api/info')
def api_info():
    """Informations sur l'API"""
    return jsonify({
        'name': 'Cat Classifier API',
        'version': '1.0.0',
        'description': 'API pour classifier les images comme chat ou non-chat',
        'model': 'Régression Logistique',
        'accuracy': '~70% sur jeu de test',
        'input_size': '64x64 pixels RGB',
        'endpoints': {
            '/': 'Page d\'accueil',
            '/predict': 'POST - Prédire une image',
            '/demo': 'Page de démonstration',
            '/docs': 'Documentation',
            '/api/info': 'Informations API'
        }
    })

if __name__ == '__main__':
    print("🚀 Démarrage de l'application Flask...")
    
    # Charger le modèle
    if load_model():
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Impossible de démarrer l'application sans modèle entraîné")
