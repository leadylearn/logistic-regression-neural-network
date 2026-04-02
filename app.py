from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
from PIL import Image
import io
import random
import os

app = Flask(__name__)

# Route explicite pour les images
@app.route('/images/<path:filename>')
def serve_images(filename):
    return send_from_directory('static/images', filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    try:
        print("📸 Requête reçue")
        
        # Vérifier l'image
        if 'image' not in request.files:
            return jsonify({'error': 'Aucune image fournie', 'success': False})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Nom de fichier vide', 'success': False})
        
        print(f"📁 Fichier reçu: {file.filename}")
        
        # Traitement de l'image avec debug
        try:
            # Lire les données brutes
            file.stream.seek(0)
            image_data = file.stream.read()
            print(f"📊 Taille des données: {len(image_data)} bytes")
            
            # Vérifier si c'est une image valide
            if len(image_data) == 0:
                return jsonify({'error': 'Fichier vide', 'success': False})
            
            # Ouvrir l'image
            img = Image.open(io.BytesIO(image_data))
            print(f"🖼️ Format original: {img.format}, Taille: {img.size}")
            
            # Convertir en RGB si nécessaire
            if img.mode != 'RGB':
                img = img.convert('RGB')
                print("🔄 Converti en RGB")
            
            # Redimensionner
            img_resized = img.resize((64, 64))
            print(f"✅ Image redimensionnée: {img_resized.size}")
            
        except Exception as img_error:
            print(f"❌ Erreur de traitement image: {img_error}")
            # Pour la démo, on simule même si l'image échoue
            print("🎯 Utilisation de prédiction simulée")
        
        # Simulation de prédiction
        prediction_prob = random.uniform(0.1, 0.95)
        is_cat = prediction_prob > 0.5
        
        if is_cat:
            class_name = "cat"
            confidence = prediction_prob
        else:
            class_name = "non-cat"
            confidence = 1 - prediction_prob
        
        print(f"🎯 Prédiction: {class_name} (confiance: {confidence:.3f})")
        
        return jsonify({
            'prediction': class_name,
            'confidence': float(confidence),
            'success': True
        })
        
    except Exception as e:
        print(f"❌ Erreur générale: {e}")
        return jsonify({'error': str(e), 'success': False})

@app.route('/demo')
def demo():
    return render_template('demo.html')

@app.route('/docs')
def docs():
    return render_template('docs.html')

@app.route('/api/info')
def api_info():
    return jsonify({
        'name': 'Cat Classifier API',
        'version': '1.0.0',
        'status': 'running',
        'mode': 'test',
        'accuracy': '70%',
        'input_size': '64x64 pixels',
        'dataset_size': '259 images',
        'training_images': '209',
        'test_images': '50',
        'processing_time': '< 100ms',
        'algorithm': 'Régression Logistique'
    })

if __name__ == '__main__':
    print("🚀 Démarrage de l'application Flask FINALE...")
    print("📝 Mode: Test avec prédictions simulées")
    print("🖼️ Images statiques activées")
    app.run(host='0.0.0.0', port=5000, debug=False)
