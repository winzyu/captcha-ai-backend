# app/routes/captcha.py
import os
import io
import random
from PIL import Image
from flask import Blueprint, jsonify, send_file, request, current_app
from app.models.inference import CaptchaPredictor
from app.utils.image_processing import preprocess_image
from app.config.config import Config

captcha_bp = Blueprint('captcha', __name__)
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        print("Initializing predictor...")
        predictor = CaptchaPredictor(Config.MODEL_PATH)
        print("Predictor initialized successfully")
    return predictor

@captcha_bp.route('/api/captcha/random', methods=['GET'])
def get_random_captcha():
    try:
        print("\nHandling random CAPTCHA request...")
        
        # Get random CAPTCHA from the raw data directory
        print(f"Looking for CAPTCHAs in: {Config.RAW_CAPTCHA_PATH}")
        captcha_files = [f for f in os.listdir(Config.RAW_CAPTCHA_PATH) 
                        if f.lower().endswith(tuple(Config.ALLOWED_EXTENSIONS))]
        
        print(f"Found {len(captcha_files)} CAPTCHA files")
        
        if not captcha_files:
            return jsonify({'error': 'No CAPTCHA images found'}), 404
        
        random_captcha = random.choice(captcha_files)
        captcha_path = os.path.join(Config.RAW_CAPTCHA_PATH, random_captcha)
        print(f"Selected CAPTCHA: {captcha_path}")
        
        # Get AI prediction
        print("Loading image...")
        with open(captcha_path, 'rb') as img_file:
            image = Image.open(img_file)
            print("Preprocessing image...")
            image_tensor = preprocess_image(image, Config.IMAGE_SIZE)
            print("Getting prediction...")
            prediction = get_predictor().predict(image_tensor)
            print(f"Prediction complete: {prediction}")
        
        response_data = {
            'success': True,
            'imagePath': f'/api/captcha/image/{random_captcha}',
            'prediction': prediction['text'],
            'confidence': prediction['average_confidence'],
            'confidencePerChar': prediction['confidence_per_char']
        }
        print(f"Sending response: {response_data}")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in get_random_captcha: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@captcha_bp.route('/api/captcha/image/<filename>')
def serve_captcha(filename):
    return send_file(
        os.path.join(Config.RAW_CAPTCHA_PATH, filename),
        mimetype='image/jpeg'
    )

@captcha_bp.route('/api/captcha/verify', methods=['POST'])
def verify_captcha():
    data = request.get_json()
    
    if not data or 'text' not in data or 'imageFile' not in data:
        return jsonify({'error': 'Missing required fields'}), 400
    
    try:
        # Get the actual CAPTCHA text from the filename
        actual_text = data['imageFile'].split('/')[-1].split('.')[0]
        
        result = {
            'correct': data['text'].lower() == actual_text.lower(),
            'actualText': actual_text
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@captcha_bp.route('/api/captcha/predict', methods=['POST'])
def predict_captcha():
    try:
        print("Handling CAPTCHA prediction request...")
        
        # Get image data from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
            
        image_file = request.files['image']
        
        # Open and preprocess the image
        image = Image.open(io.BytesIO(image_file.read()))
        image_tensor = preprocess_image(image, Config.IMAGE_SIZE)
        
        # Get prediction using our model
        predictor = get_predictor()
        prediction = predictor.predict(image_tensor)
        
        # Return prediction results
        response_data = {
            'success': True,
            'prediction': prediction['text'],
            'confidence': prediction['average_confidence'],
            'confidencePerChar': prediction['confidence_per_char']
        }
        
        print(f"Prediction successful: {response_data}")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in predict_captcha: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
