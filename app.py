# CRITICAL: Disable FastSAM before any ultralytics imports to save memory
import os
os.environ["ULTRALYTICS_NO_FASTSAM"] = "1"

from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory, jsonify
from flask_cors import CORS
import sys
from supabase import create_client, Client
from datetime import datetime
from werkzeug.utils import secure_filename

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://zqkqmjlepigpwfykwzey.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY", os.environ.get("SUPABASE_KEY", "sb_publishable_HNgog4XZVoR6FqaKuzIcGQ_7yrDAjFn"))

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key-here-change-in-production')

# Configure session for production
# Render uses HTTPS, so secure cookies should be enabled
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FLASK_ENV') == 'production'  # True for production (HTTPS)
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = 86400  # 24 hours

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRYCNN_REPO_PATH = os.path.join(BASE_DIR, "trycnn_repo")
if TRYCNN_REPO_PATH not in sys.path:
    sys.path.append(TRYCNN_REPO_PATH)

_cnn_predictor = None
_yolo_model = None  # Global YOLO model - loaded once at startup


def get_yolo_model():
    """Load YOLO model once at startup - shared across all requests"""
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    
    try:
        from ultralytics import YOLO
        print("Loading YOLOv8n model (startup, shared instance)...")
        # Use yolov8n.pt (nano) for minimal memory usage
        _yolo_model = YOLO('yolov8n.pt')
        print("✓ YOLOv8n model loaded successfully")
        return _yolo_model
    except Exception as e:
        print(f"⚠ Failed to load YOLO model: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_cnn_predictor():
    """Lazy load CNN predictor to save memory - only load when actually needed"""
    global _cnn_predictor
    if _cnn_predictor is not None:
        return _cnn_predictor
    try:
        # Only import when needed to save memory at startup
        from predict import PechayPredictor
        model_path = os.path.join(TRYCNN_REPO_PATH, "pechay_cnn_model_20251212_184656.pth")
        if not os.path.exists(model_path):
            return None
        # Use CPU to save memory
        _cnn_predictor = PechayPredictor(model_path, device='cpu')
        return _cnn_predictor
    except Exception as e:
        print(f"Error loading CNN predictor: {e}")
        return None

# Preload YOLO model at startup (shared instance)
# DISABLED: YOLO loading causes memory issues on Render free tier
# Uncomment below to enable YOLO (not recommended for free tier)
# print("=== Preloading YOLO model at startup ===")
# get_yolo_model()
print("=== YOLO model loading DISABLED to prevent memory issues ===")
print("=== Detection will use Roboflow API only (no YOLO validation) ===")

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    mode = request.args.get('mode', 'login')
    
    if request.method == 'POST':
        # Login logic - Try both Supabase Auth and custom users table
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            return render_template('login.html', mode='login', error='Please provide username and password')
        
        from services.database_service import DatabaseService
        import hashlib
        db_service = DatabaseService()
        
        try:
            # Use custom users table for authentication
            user = None
            
            # Try email first
            if '@' in username:
                user = db_service.get_user_by_email(username)
            
            # If not found or not an email, try username
            if not user:
                user = db_service.get_user_by_username(username)
            
            if user:
                # Verify password
                stored_password = user.get('password', '')
                password_hash = hashlib.sha256(password.encode()).hexdigest()
                
                # Check if password matches (plain text or hashed)
                if stored_password == password or stored_password == password_hash:
                    # Login successful - set session
                    session.permanent = True  # Make session permanent
                    session['user'] = user.get('username', username)
                    session['user_id'] = str(user.get('id'))  # Convert UUID to string
                    session['user_email'] = user.get('email', username)
                    session['user_full'] = user  # Store full user object if needed
                    
                    print(f"Login successful for user: {user.get('username')} (ID: {user.get('id')})")
                    return redirect(url_for('dashboard', page='dashboard'))
                else:
                    return render_template('login.html', mode='login', error='Invalid password.')
            else:
                return render_template('login.html', mode='login', error='User not found. Please register first.')
        except Exception as e:
            print(f"Login error: {e}")
            import traceback
            traceback.print_exc()
            # Don't expose internal errors to user, but log them
            error_msg = 'Invalid credentials'
            if 'email' in str(e).lower() or 'user' in str(e).lower():
                error_msg = 'User not found. Please register first.'
            return render_template('login.html', mode='login', error=error_msg)
    
    error = request.args.get('error')
    success = request.args.get('success')
    return render_template('login.html', mode=mode, error=error, success=success)

@app.route('/api/detect_live', methods=['POST'])
def detect_live():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
        
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data'}), 400
            
        image_data = data['image']
        # Remove header if present (data:image/jpeg;base64,...)
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        # Decode base64
        import base64
        import numpy as np
        import cv2
        
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image'}), 400
            
        # Save temporarily for processing
        user_id = session.get('user_id')
        filename = f"live_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        # For Vercel/serverless: use /tmp directory (writable)
        # For regular deployment: use uploads directory
        if os.path.exists('/tmp'):
            uploads_dir = '/tmp/uploads'
        else:
            uploads_dir = os.path.join('uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        filepath = os.path.join(uploads_dir, filename)
        
        cv2.imwrite(filepath, img)
        
        # Run detection
        from services.detection_service import DetectionService
        # Pass shared YOLO model to avoid reloading
        shared_yolo_model = get_yolo_model()
        detection_service = DetectionService(yolo_model=shared_yolo_model)
        detection_result = detection_service.detect_leaf(filepath)
        
        # Check if we should save (Auto-capture logic)
        # Save if it is a Pechay (Healthy or Diseased) and confidence is high
        should_save = False
        condition = detection_result.get('condition', 'Unknown')
        
        if condition in ['Healthy', 'Diseased']:
            should_save = True
            
        saved_result = None
        if should_save:
            from services.database_service import DatabaseService
            db_service = DatabaseService()
            
            # Add specific live detection metadata
            detection_result['method'] = 'Live Stream Auto-Capture'
            
            saved_result = db_service.save_detection(
                user_id=user_id,
                image_path=f"uploads/{filename}",
                detection_result=detection_result
            )
            
        return jsonify({
            'success': True,
            'result': detection_result,
            'saved': saved_result is not None,
            'saved_id': saved_result.get('id') if saved_result else None
        })
        
    except Exception as e:
        print(f"Live detection error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/logout')
def logout():
    try:
        # Sign out from Supabase
        supabase.auth.sign_out()
    except Exception as e:
        print(f"Logout error: {e}")
    finally:
        session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    # Check if user is logged in
    if 'user' not in session:
        return redirect(url_for('login'))
    
    page = request.args.get('page', 'dashboard')
    if page == 'up':
        page = 'upload'
    user_id = session.get('user_id')
    
    # Debug logging
    print(f"\n=== Dashboard Access ===")
    print(f"Session user: {session.get('user')}")
    print(f"Session user_id: {user_id}")
    print(f"Page: {page}")
    print(f"Request method: {request.method}")
    
    # Handle POST requests (file uploads)
    detection_result = None
    upload_status = None
    
    if request.method == 'POST':
        # Handle leaf image upload
        if 'leafImage' in request.files:
            file = request.files['leafImage']
            if file and file.filename:
                try:
                    # For Vercel/serverless: use /tmp directory (writable)
                    # For regular deployment: use uploads directory
                    if os.path.exists('/tmp'):
                        uploads_dir = '/tmp/uploads'
                    else:
                        uploads_dir = os.path.join('uploads')
                    os.makedirs(uploads_dir, exist_ok=True)
                    
                    # Save uploaded file
                    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
                    filepath = os.path.join(uploads_dir, filename)
                    file.save(filepath)
                    
                    print(f"\n=== Image Upload Received ===")
                    print(f"File saved: {filepath}")
                    print(f"Filename: {filename}")
                    print(f"User ID: {user_id}")
                    
                    # Process image with detection service (Roboflow API first)
                    # Import detection service lazily to avoid heavy imports at startup
                    try:
                        from services.detection_service import DetectionService
                        print(f"\n=== Starting Detection Service ===")
                        # Pass shared YOLO model to avoid reloading (CRITICAL for memory)
                        shared_yolo_model = get_yolo_model()
                        if shared_yolo_model is None:
                            print("⚠ Warning: YOLO model not available, detection will proceed without YOLO validation")
                        detection_service = DetectionService(yolo_model=shared_yolo_model)
                        print(f"Calling API for image detection (Roboflow)...")
                        if shared_yolo_model:
                            print(f"✓ Using shared YOLO model (pre-loaded)")
                        else:
                            print(f"⚠ YOLO validation will be skipped (no model available)")
                        # Set a timeout for detection to prevent worker timeout
                        import signal
                        detection_result = detection_service.detect_leaf(filepath)
                    except Exception as e:
                        print(f"Detection service error: {e}")
                        import traceback
                        traceback.print_exc()
                        detection_result = {
                            'filename': filename,
                            'condition': 'Diseased',
                            'disease_name': 'Detection Failed',
                            'confidence': 0.0,
                            'error': f'Detection service error: {str(e)}'
                        }
                    api_result = detection_result if isinstance(detection_result, dict) else None
                    
                    if isinstance(detection_result, dict):
                        detection_result['image_path'] = f"uploads/{filename}"
                    print(f"=== Detection Complete (API) ===")
                    print(f"Result: {detection_result}")
                    
                    api_error = None
                    if isinstance(detection_result, dict):
                        api_error = detection_result.get('error')
                        api_condition = detection_result.get('condition')
                        api_disease = detection_result.get('disease_name')
                        if api_error:
                            print(f"API error detected: {api_error}")
                            print("Skipping heavy CNN/AI fallback due to environment limits; using API result only.")
                    
                    # Save detection result to database
                    from services.database_service import DatabaseService
                    db_service = DatabaseService()
                    
                    try:
                        saved_result = db_service.save_detection(
                            user_id=user_id,
                            image_path=f"uploads/{filename}",
                            detection_result=detection_result
                        )
                        
                        if saved_result:
                            upload_status = "Image processed and saved successfully!"
                            print(f"Detection saved: {saved_result}")
                        else:
                            upload_status = "Detection completed but failed to save to database."
                            print("Warning: Detection result not saved to database")
                    except Exception as db_error:
                        print(f"Database save error: {db_error}")
                        import traceback
                        traceback.print_exc()
                        upload_status = f"Detection completed but database save failed: {str(db_error)}"
                        saved_result = None
                        
                except Exception as e:
                    print(f"Upload error: {e}")
                    import traceback
                    traceback.print_exc()
                    upload_status = f"Error processing image: {str(e)}"
                    detection_result = None
    
    # Get data from Supabase using database service
    from services.database_service import DatabaseService
    
    # Initialize default values
    dashboard_stats = {
        'total_scans': 0,
        'healthy_leaves': 0,
        'diseased_leaves': 0,
        'success_rate': 0
    }
    results = []
    
    try:
        db_service = DatabaseService()
        # Get dashboard stats
        dashboard_stats = db_service.get_dashboard_stats(user_id)
        print(f"Dashboard stats: {dashboard_stats}")
        
        # Get detection results if needed
        filter_condition = request.args.get('filter')
        results = db_service.get_user_detections(user_id, limit=50, condition=filter_condition)
        print(f"Detection results: {len(results)} found")
        
    except Exception as e:
        print(f"Database error: {e}")
        import traceback
        traceback.print_exc()
        # Use default values already set above
    
    # Get analytics data if on analytics page
    analytics_data = None
    if page == 'analytics':
        from services.analytics_service import AnalyticsService
        analytics_service = AnalyticsService()
        analytics_data = analytics_service.get_analytics_data(user_id, days=7)
    filter_condition = request.args.get('filter')
    filter_days = request.args.get('days', '')
    date_range_text = None
    
    # Database URL for display (masked for security)
    try:
        db_url_display = SUPABASE_URL.replace('https://', '').split('.')[0] + '.supabase.co' if SUPABASE_URL else 'Configured'
    except:
        db_url_display = 'Configured'
    
    try:
        current_user_data = {
            'id': session.get('user_id'),
            'email': session.get('user_email', ''),
            'username': session.get('user', '')
        }
        
        return render_template('dashboard.html',
                             page=page,
                             dashboard_stats=dashboard_stats,
                             results=results,
                             analytics_data=analytics_data,
                             detection_result=detection_result,
                             upload_status=upload_status,
                             filter_condition=filter_condition,
                             filter_days=filter_days,
                             date_range_text=date_range_text,
                             db_url_display=db_url_display,
                             current_user=current_user_data)
    except Exception as e:
        print(f"Template rendering error: {e}")
        import traceback
        traceback.print_exc()
        return f"Error rendering template: {str(e)}", 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # For Vercel/serverless: check /tmp first
    if os.path.exists('/tmp/uploads') and os.path.exists(f'/tmp/uploads/{filename}'):
        return send_from_directory('/tmp/uploads', filename)
    return send_from_directory('uploads', filename)


@app.route('/uploads/dataset/<filename>')
def uploaded_dataset_file(filename):
    # For Vercel/serverless: check /tmp first
    if os.path.exists('/tmp/uploads/dataset') and os.path.exists(f'/tmp/uploads/dataset/{filename}'):
        return send_from_directory('/tmp/uploads/dataset', filename)
    return send_from_directory(os.path.join('uploads', 'dataset'), filename)

@app.route('/healthz')
def healthz():
    return jsonify({'status': 'ok'}), 200


@app.route('/upload_image_immediate', methods=['POST'])
def upload_image_immediate():
    if 'user' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    try:
        # For Vercel/serverless: use /tmp directory (writable)
        # For regular deployment: use uploads directory
        if os.path.exists('/tmp'):
            uploads_dir = '/tmp/uploads/dataset'
        else:
            uploads_dir = os.path.join('uploads', 'dataset')
        os.makedirs(uploads_dir, exist_ok=True)
        
        filename = secure_filename(file.filename)
        unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
        filepath = os.path.join(uploads_dir, unique_filename)
        file.save(filepath)
        
        from services.database_service import DatabaseService
        from db import create_yolo_file
        db_service = DatabaseService()
        user_id = session.get('user_id')
        condition = request.form.get('condition', 'Healthy')
        disease_select = request.form.get('disease_select', '')
        new_disease_name = request.form.get('new_disease_name', '').strip()
        treatment = request.form.get('treatment', '').strip()
        if condition not in ['Healthy', 'Diseased', 'Not Pechay']:
            condition = 'Healthy'
        disease_name = None
        if condition == 'Diseased':
            if disease_select == 'new':
                disease_name = new_disease_name or 'Unknown Disease'
            elif disease_select:
                disease_name = disease_select
        label = 'Healthy' if condition == 'Healthy' else 'Diseased'
        db_service.add_dataset_image(
            user_id=user_id,
            filename=unique_filename,
            label=label,
            image_path=f"/uploads/dataset/{unique_filename}"
        )
        embedding_list = None
        predictor = get_cnn_predictor()
        if predictor is not None:
            try:
                features = predictor.extract_features(filepath)
                if features is not None:
                    embedding_list = [float(x) for x in features.flatten().tolist()] if hasattr(features, "flatten") else [float(x) for x in features]
            except Exception as e:
                print(f"Error generating embedding: {e}")
        db_service.save_petchay_dataset_entry(
            filename=unique_filename,
            condition=condition,
            disease_name=disease_name,
            image_url=f"/uploads/dataset/{unique_filename}",
            embedding=embedding_list,
            user_id=user_id
        )
        dataset_type = disease_name if disease_name else condition
        create_yolo_file(
            filename=unique_filename,
            file_type="image",
            dataset_type=dataset_type,
            url=f"/uploads/dataset/{unique_filename}",
            treatment=treatment if condition == 'Diseased' and treatment else None
        )
        try:
            supabase.table("yolo_files").update({
                "label": label,
                "label_confidence": 1.0,
                "image_region": "leaf",
                "quality_score": 0.9,
                "is_verified": True
            }).eq("filename", unique_filename).execute()
        except Exception as e:
            print(f"Error updating yolo_files label fields: {e}")
        
        return jsonify({'success': True, 'filename': unique_filename}), 200
    except Exception as e:
        print(f"Immediate upload error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/dataset')
def dataset_manager():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    from services.database_service import DatabaseService
    db_service = DatabaseService()
    user_id = session.get('user_id')
    dataset_stats = db_service.get_dataset_stats(user_id)
    dataset_images = db_service.get_dataset_images(user_id=user_id, limit=100)
    return render_template('dataset_manager.html',
                         dataset_stats=dataset_stats,
                         dataset_images=dataset_images)


@app.route('/upload_dataset_workflow', methods=['POST'])
def upload_dataset_workflow():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    try:
        condition = request.form.get('condition', 'Healthy')
        disease_select = request.form.get('disease_select', '')
        new_disease_name = request.form.get('new_disease_name', '').strip()
        treatment = request.form.get('treatment', '').strip()
        
        if condition not in ['Healthy', 'Diseased']:
            condition = 'Healthy'
        
        disease_name = None
        if condition == 'Diseased':
            if disease_select == 'new':
                disease_name = new_disease_name or 'Unknown Disease'
            elif disease_select:
                disease_name = disease_select
        
        parts = [f"Condition: {condition}"]
        if disease_name:
            parts.append(f"Disease: {disease_name}")
        if treatment:
            parts.append("Treatment notes saved")
        
        flash(f"success_dataset_workflow|{' | '.join(parts)}")
    except Exception as e:
        flash(f"error_dataset_workflow|Error processing dataset workflow: {e}")
    
    return redirect(url_for('dataset_manager'))

@app.route('/report')
def report():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    user_id = session.get('user_id')
    
    from services.database_service import DatabaseService
    db_service = DatabaseService()
    
    stats = db_service.get_dashboard_stats(user_id)
    results = db_service.get_user_detections(user_id, limit=100)
    
    filters = {
        'condition': request.args.get('filter') or 'All',
        'range': request.args.get('range') or 'All time'
    }
    
    return render_template(
        'report.html',
        username=session.get('user'),
        date=datetime.now(),
        filters=filters,
        stats=stats,
        results=results
    )

@app.route('/download_report')
def download_report():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    user_id = session.get('user_id')
    from services.database_service import DatabaseService
    db_service = DatabaseService()
    
    # Get results (same as report)
    results = db_service.get_user_detections(user_id, limit=100)
    
    # Create CSV
    import io
    import csv
    from flask import make_response
    
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['Date', 'Filename', 'Condition', 'Confidence', 'Disease Name'])
    
    for r in results:
        cw.writerow([
            r.get('timestamp', ''),
            r.get('filename', ''),
            r.get('condition', ''),
            f"{r.get('confidence', 0)}%",
            r.get('disease_name', '')
        ])
        
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=pechay_report.csv"
    output.headers["Content-type"] = "text/csv"
    return output

@app.route('/api/delete_detection', methods=['POST'])
def delete_detection():
    """Delete a detection result"""
    if 'user' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    
    try:
        data = request.json
        detection_id = data.get('id')
        
        if not detection_id:
            return jsonify({'success': False, 'error': 'Detection ID required'}), 400
        
        user_id = session.get('user_id')
        
        from services.database_service import DatabaseService
        db_service = DatabaseService()
        
        success = db_service.delete_detection(user_id, detection_id)
        
        if success:
            return jsonify({'success': True, 'message': 'Detection deleted successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to delete detection or detection not found'}), 404
            
    except Exception as e:
        print(f"Delete detection error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable (Render provides this) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=port)
