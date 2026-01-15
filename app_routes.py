"""
Additional routes for the Pechay Detection System
This file contains API endpoints and additional routes
"""
from flask import Blueprint, request, jsonify, session, send_file
from services.model_service import ModelService
from services.database_service import DatabaseService
from services.embedding_service import EmbeddingGenerator, SimilaritySearch
from utils.image_processor import ImageProcessor
from utils.disease_info import get_treatment_recommendations, get_condition_from_disease
from config.database import supabase, STORAGE_BUCKETS
import numpy as np
import cv2
import os
from datetime import datetime
import uuid
from werkzeug.utils import secure_filename

# Create blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Initialize services (will be initialized in main app)
model_service = None
embedding_generator = None
similarity_search = None
db_service = DatabaseService()
image_processor = ImageProcessor()

def init_services(model_service_instance, embedding_gen, similarity_search_instance):
    """Initialize services"""
    global model_service, embedding_generator, similarity_search
    model_service = model_service_instance
    embedding_generator = embedding_gen
    similarity_search = similarity_search_instance


@api_bp.route('/predict', methods=['POST'])
def predict():
    """Prediction API endpoint"""
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    method = request.form.get('method', 'hybrid')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read image
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Validate image
        is_valid, validation_msg, validation_details = image_processor.validate_image_quality(image)
        if not is_valid:
            return jsonify({'error': validation_msg, 'validation': validation_details}), 400
        
        # Predict
        if not model_service or not model_service.models_loaded:
            return jsonify({'error': 'Models not loaded'}), 500
        
        prediction = model_service.predict(image, method=method)
        
        if 'error' in prediction:
            return jsonify({'error': prediction['error']}), 500
        
        disease_name = prediction.get('class', 'Unknown')
        confidence = prediction.get('confidence', 0.0)
        condition = get_condition_from_disease(disease_name)
        
        # Get treatment recommendations
        recommendations = get_treatment_recommendations(disease_name, confidence)
        
        result = {
            'condition': condition,
            'disease_name': disease_name if condition == 'Diseased' else None,
            'confidence': float(confidence),
            'method': method,
            'recommendations': recommendations,
            'validation': validation_details
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/upload_smart', methods=['POST'])
def upload_smart():
    """Smart upload workflow with validation and detection"""
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    if 'leafImage' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['leafImage']
    user_id = session.get('user_id')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read and validate image
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Validate
        is_valid, validation_msg, validation_details = image_processor.validate_image_quality(image)
        if not is_valid:
            return jsonify({'error': validation_msg}), 400
        
        # Generate filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        
        # Save to uploads directory
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        local_path = os.path.join(upload_dir, unique_filename)
        cv2.imwrite(local_path, image)
        
        # Upload to Supabase Storage
        storage_url = db_service.upload_to_storage(
            STORAGE_BUCKETS['uploads'],
            'detections',
            file_bytes
        )
        
        # Run detection
        prediction = model_service.predict(image, method='hybrid') if model_service and model_service.models_loaded else None
        
        if prediction and 'error' not in prediction:
            disease_name = prediction.get('class', 'Unknown')
            confidence = prediction.get('confidence', 0.0)
            condition = get_condition_from_disease(disease_name)
            
            # Save detection result
            detection_result = {
                'condition': condition,
                'disease_name': disease_name if condition == 'Diseased' else None,
                'confidence': float(confidence),
                'method': 'hybrid'
            }
            
            db_service.save_detection(user_id, storage_url or local_path, detection_result)
        
        return jsonify({
            'success': True,
            'message': 'Image uploaded and processed',
            'image_path': storage_url or local_path,
            'prediction': prediction
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/upload_image_immediate', methods=['POST'])
def upload_image_immediate():
    """Immediate image upload without processing"""
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    user_id = session.get('user_id')
    
    try:
        file_bytes = file.read()
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        
        # Upload to storage
        storage_url = db_service.upload_to_storage(
            STORAGE_BUCKETS['uploads'],
            'immediate',
            file_bytes
        )
        
        return jsonify({
            'success': True,
            'url': storage_url,
            'filename': unique_filename
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500







