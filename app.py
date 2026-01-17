from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory, jsonify
import os
import sys
from supabase import create_client, Client
from datetime import datetime
from werkzeug.utils import secure_filename

SUPABASE_URL = "https://zqkqmjlepigpwfykwzey.supabase.co"
SUPABASE_KEY = "sb_publishable_HNgog4XZVoR6FqaKuzIcGQ_7yrDAjFn"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key-here-change-in-production')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRYCNN_REPO_PATH = os.path.join(BASE_DIR, "trycnn_repo")
if TRYCNN_REPO_PATH not in sys.path:
    sys.path.append(TRYCNN_REPO_PATH)

_cnn_predictor = None


def get_cnn_predictor():
    global _cnn_predictor
    if _cnn_predictor is not None:
        return _cnn_predictor
    try:
        from predict import PechayPredictor
        model_path = os.path.join(TRYCNN_REPO_PATH, "pechay_cnn_model_20251212_184656.pth")
        if not os.path.exists(model_path):
            return None
        _cnn_predictor = PechayPredictor(model_path, device='cpu')
        return _cnn_predictor
    except Exception as e:
        print(f"Error loading CNN predictor: {e}")
        return None

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    mode = request.args.get('mode', 'login')
    
    if request.method == 'POST':
        if request.form.get('action') == 'register':
            # Registration logic with Supabase
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')
            
            if password != confirm_password:
                return render_template('login.html', mode='register', error='Passwords do not match')
            
            try:
                from services.database_service import DatabaseService
                import hashlib
                db_service = DatabaseService()
                
                # Check if user already exists
                existing_user = db_service.get_user_by_email(email)
                if existing_user:
                    return render_template('login.html', mode='register', error='Email already registered')
                
                existing_user = db_service.get_user_by_username(username)
                if existing_user:
                    return render_template('login.html', mode='register', error='Username already taken')
                
                # Hash password (using SHA256 for now - use bcrypt in production!)
                password_hash = hashlib.sha256(password.encode()).hexdigest()
                
                # Save to custom users table
                user_result = db_service.create_user(username, email, password_hash)
                
                if user_result:
                    flash('Registration successful! Please login.')
                    return redirect(url_for('login', mode='login'))
                else:
                    return render_template('login.html', mode='register', error='Registration failed. Please try again.')
            except Exception as e:
                print(f"Registration error: {e}")
                import traceback
                traceback.print_exc()
                return render_template('login.html', mode='register', error=f'Registration error: {str(e)}')
        else:
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
                    # Ensure uploads directory exists
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
                    from services.detection_service import DetectionService
                    print(f"\n=== Starting Detection Service ===")
                    detection_service = DetectionService()
                    print(f"Calling API for image detection (Roboflow)...")
                    detection_result = detection_service.detect_leaf(filepath)
                    
                    # Ensure image_path is set for dashboard display
                    if isinstance(detection_result, dict):
                        detection_result['image_path'] = f"uploads/{filename}"
                    print(f"=== Detection Complete (API) ===")
                    print(f"Result: {detection_result}")
                    
                    # If API returned an error, fall back to local CNN/AI pipeline
                    use_cnn_fallback = False
                    if not isinstance(detection_result, dict):
                        use_cnn_fallback = True
                    else:
                        api_error = detection_result.get('error')
                        if api_error:
                            print(f"API error detected: {api_error}")
                            use_cnn_fallback = True
                    
                    if use_cnn_fallback:
                        try:
                            print("\n=== Falling back to local CNN/AI detection ===")
                            from trycnn_repo.app import detect_leaf_condition
                            cnn_result = detect_leaf_condition(filepath)
                            print(f"=== CNN/AI Fallback Result ===")
                            print(cnn_result)
                            
                            if isinstance(cnn_result, dict):
                                detection_result = {
                                    'filename': filename,
                                    'condition': cnn_result.get('condition', 'Unknown'),
                                    'disease_name': cnn_result.get('disease_name'),
                                    'confidence': cnn_result.get('confidence', 0.0),
                                    'probabilities': cnn_result.get('all_probabilities', {}),
                                    'recommendations': cnn_result.get('recommendations', {}),
                                    'treatment': cnn_result.get('treatment', ''),
                                    'error': None,
                                    'validation_details': cnn_result.get('validation_reason'),
                                    'is_fallback': True,
                                    'method': 'CNN/AI Fallback',
                                    'image_path': f"uploads/{filename}"
                                }
                        except Exception as e:
                            print(f"CNN/AI fallback error: {e}")
                    
                    # Save detection result to database
                    from services.database_service import DatabaseService
                    db_service = DatabaseService()
                    
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
                        
                except Exception as e:
                    print(f"Upload error: {e}")
                    import traceback
                    traceback.print_exc()
                    upload_status = f"Error processing image: {str(e)}"
                    detection_result = None
    
    # Get data from Supabase using database service
    from services.database_service import DatabaseService
    db_service = DatabaseService()
    
    try:
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
        dashboard_stats = {
            'total_scans': 0,
            'healthy_leaves': 0,
            'diseased_leaves': 0,
            'success_rate': 0
        }
        results = []
    
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
                         current_user={'id': session.get('user_id'), 'email': session.get('user_email'), 'username': session.get('user')})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)


@app.route('/uploads/dataset/<filename>')
def uploaded_dataset_file(filename):
    return send_from_directory(os.path.join('uploads', 'dataset'), filename)


@app.route('/upload_image_immediate', methods=['POST'])
def upload_image_immediate():
    if 'user' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    try:
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
        if condition not in ['Healthy', 'Diseased']:
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
        
        flash(' | '.join(parts))
    except Exception as e:
        flash(f"Error processing dataset workflow: {e}")
    
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

if __name__ == '__main__':
    # Get port from environment variable (Render provides this) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=port)
