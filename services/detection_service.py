"""
Detection service for leaf image analysis using Roboflow API
"""
import os
import cv2
import numpy as np
import requests
import base64
from typing import Dict, Optional, Tuple
from utils.disease_info import get_treatment_recommendations, get_condition_from_disease
from utils.image_processor import ImageProcessor

# Use HTTP requests directly instead of inference_sdk package
ROBOFLOW_AVAILABLE = True  # Always available - using HTTP requests

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. Install with: pip install ultralytics")

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("Warning: face_recognition not available. Install with: pip install face-recognition")

class DetectionService:
    """Service for detecting leaf diseases using Roboflow API"""
    
    def __init__(self):
        self.api_url = "https://serverless.roboflow.com"
        self.api_key = "B7RL7B0aD47us70qgevM"
        self.workspace_name = "bael"
        self.workflow_id = "custom-workflow-3"
    
    def detect_leaf(self, image_path: str) -> Dict:
        """
        Detect leaf condition from image using Roboflow API workflow
        """
        filename = image_path.split('/')[-1] if '/' in image_path else image_path.split('\\')[-1] if '\\' in image_path else image_path
        
        try:
            # Check if image file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Load image for validation
            image_np = ImageProcessor.load_image(image_path)
            if image_np is None:
                return self._get_default_result(filename, error="Could not load image for validation.")
            
            # Perform image validation (optional - don't block API call for non-critical issues)
            try:
                is_valid, validation_msg, validation_details = ImageProcessor.validate_image_quality(image_np)
                if not is_valid:
                    # Only block if it's a critical validation error (image too small, completely invalid)
                    # Allow API call for other validation warnings
                    critical_errors = ["too small", "invalid image", "could not read"]
                    is_critical = any(critical in validation_msg.lower() for critical in critical_errors)
                    
                    if is_critical:
                        print(f"Critical validation error: {validation_msg}")
                        return self._get_default_result(filename, error=f"Image validation failed: {validation_msg}", validation_details=validation_details)
                    else:
                        print(f"Validation warning (non-critical): {validation_msg} - proceeding to API")
            except Exception as e:
                print(f"Validation error (non-critical, proceeding): {e}")
                # Continue to API call even if validation has errors
            
            # Use Roboflow API via HTTP requests
            if ROBOFLOW_AVAILABLE:
                try:
                    print(f"\n=== Calling Roboflow API ===")
                    print(f"Image path: {image_path}")
                    print(f"Image exists: {os.path.exists(image_path)}")
                    print(f"Workspace: {self.workspace_name}")
                    print(f"Workflow ID: {self.workflow_id}")
                    
                    # Read image file and encode to base64
                    if not os.path.exists(image_path):
                        raise FileNotFoundError(f"Image not found: {image_path}")
                    
                    with open(image_path, "rb") as img_file:
                        img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
                    
                    # Try different API base URLs (from trycnn_repo format)
                    base_urls = [
                        "https://api.roboflow.com",  # Standard API domain
                        "https://detect.roboflow.com",  # Detection API domain
                        "https://serverless.roboflow.com",  # Serverless workflows domain
                    ]
                    
                    # Construct URL using format from trycnn_repo: /infer/workflows/{workspace}/{workflow}
                    workflow_url = None
                    for base_url in base_urls:
                        # Format: base_url/infer/workflows/workspace/workflow
                        test_url = f"{base_url}/infer/workflows/{self.workspace_name}/{self.workflow_id}"
                        workflow_url = test_url
                        print(f"Trying API endpoint: {workflow_url}")
                        
                        # Payload format from trycnn_repo
                        payload = {
                            "inputs": {
                                "image": {"type": "base64", "value": img_base64}
                            },
                            "api_key": self.api_key
                        }
                        
                        try:
                            response = requests.post(
                                workflow_url,
                                json=payload,
                                headers={"Content-Type": "application/json"},
                                timeout=30
                            )
                            
                            print(f"API Response Status: {response.status_code}")
                            
                            if response.status_code == 200:
                                result = response.json()
                                print(f"API Response (first 1000 chars): {str(result)[:1000]}")
                                
                                # Parse Roboflow API results
                                detection_result = self._parse_roboflow_result(result, filename)
                                detection_result['method'] = f'Roboflow API ({base_url})'
                                detection_result['is_fallback'] = False
                                
                                print(f"=== Parsed Detection Result ===")
                                print(f"Condition: {detection_result.get('condition')}")
                                print(f"Disease: {detection_result.get('disease_name')}")
                                print(f"Confidence: {detection_result.get('confidence')}%")
                                
                                return detection_result
                            elif response.status_code != 404:  # 404 means endpoint doesn't exist
                                # Try next URL if this one returns error (but not 404)
                                error_text = response.text[:200]
                                print(f"API Error ({response.status_code}): {error_text}")
                                if base_url == base_urls[-1]:  # Last URL, return error
                                    return self._get_default_result(
                                        filename,
                                        error=f"API returned {response.status_code}: {error_text}"
                                    )
                                continue  # Try next base URL
                            else:
                                # 404 - endpoint doesn't exist on this domain, try next
                                print(f"Endpoint not found on {base_url}, trying next...")
                                continue
                                
                        except requests.exceptions.RequestException as req_error:
                            print(f"Request error for {base_url}: {str(req_error)[:200]}")
                            if base_url == base_urls[-1]:  # Last URL
                                return self._get_default_result(
                                    filename,
                                    error=f"API request failed: {str(req_error)[:200]}"
                                )
                            continue  # Try next base URL
                    
                    # If we get here, all URLs failed
                    return self._get_default_result(
                        filename,
                        error="All Roboflow API endpoints failed. Please check configuration."
                    )
                    
                except Exception as e:
                    print(f"=== Roboflow API Error ===")
                    print(f"Error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return self._get_default_result(filename, error=f"API error: {str(e)}")
            
            # Fallback: return default/error result
            return self._get_default_result(filename, error="API not available or failed")
            
        except Exception as e:
            print(f"Detection error: {e}")
            import traceback
            traceback.print_exc()
            return self._get_default_result(filename, error=str(e))
    
    def _parse_roboflow_result(self, result: Dict, filename: str) -> Dict:
        """Parse Roboflow API workflow result - handles multiple response formats"""
        try:
            import json
            print(f"\n=== Parsing Roboflow Result ===")
            print(f"Result type: {type(result)}")
            print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            print(f"Full result (first 2000 chars): {str(result)[:2000]}")
            
            # Handle different possible response structures
            predictions = None
            prediction = None
            class_name = None
            confidence = 0.0
            probabilities = {}
            
            # Try different possible response formats
            if isinstance(result, dict):
                # Format 1: Direct predictions array
                if 'predictions' in result:
                    predictions = result['predictions']
                # Format 2: Nested in workflow output
                elif 'output' in result:
                    output = result['output']
                    if isinstance(output, dict) and 'predictions' in output:
                        predictions = output['predictions']
                    elif isinstance(output, list):
                        predictions = output
                # Format 3: Direct prediction object
                elif 'class' in result or 'predicted_class' in result:
                    prediction = result
                # Format 4: Workflow steps output
                elif 'steps' in result:
                    # Try to find prediction in workflow steps
                    for step in result.get('steps', []):
                        if 'output' in step:
                            step_output = step['output']
                            if isinstance(step_output, dict) and 'predictions' in step_output:
                                predictions = step_output['predictions']
                                break
                            elif isinstance(step_output, list):
                                predictions = step_output
                                break
                # Format 5: Check if result itself is a list
                elif isinstance(result, list):
                    predictions = result
            
            # Extract prediction data
            if predictions:
                if isinstance(predictions, list) and len(predictions) > 0:
                    # Get highest confidence prediction
                    prediction = max(predictions, key=lambda p: float(
                        p.get('confidence', p.get('score', p.get('confidence_score', 0.0)))
                    ))
                    # Build probabilities dict from all predictions
                    for pred in predictions:
                        pred_class = pred.get('class', pred.get('predicted_class', pred.get('label', 'Unknown')))
                        pred_conf_raw = pred.get('confidence', pred.get('score', pred.get('confidence_score', 0.0)))
                        pred_conf = float(pred_conf_raw)
                        
                        # Handle percentage format
                        if pred_conf > 1.0:
                            pred_conf = pred_conf / 100.0
                        
                        if pred_class and pred_conf > 0:
                            probabilities[pred_class] = pred_conf
                elif isinstance(predictions, dict):
                    prediction = predictions
            
            # Extract class and confidence from prediction
            if prediction:
                # Try multiple possible field names for class
                class_name = (
                    prediction.get('class') or 
                    prediction.get('predicted_class') or 
                    prediction.get('label') or 
                    prediction.get('name') or
                    prediction.get('top_class') or
                    'Unknown'
                )
                
                # Try multiple possible field names for confidence
                confidence_raw = (
                    prediction.get('confidence') or 
                    prediction.get('score') or 
                    prediction.get('confidence_score') or 
                    prediction.get('probability') or
                    prediction.get('prob') or
                    0.0
                )
                confidence = float(confidence_raw)
                
                # Handle confidence values that might already be in percentage (0-100) format
                if confidence > 1.0:
                    # Assume it's already a percentage, convert to decimal
                    confidence = confidence / 100.0
                    print(f"Converted confidence from percentage: {confidence_raw}% -> {confidence}")
            
            # If still no prediction found, try to extract from result directly
            if not class_name or class_name == 'Unknown':
                class_name = (
                    result.get('class') or 
                    result.get('predicted_class') or 
                    result.get('label') or 
                    result.get('top_class') or
                    'Unknown'
                )
                confidence_raw = (
                    result.get('confidence') or 
                    result.get('score') or 
                    result.get('confidence_score') or 
                    0.0
                )
                confidence = float(confidence_raw)
                
                # Handle confidence values that might already be in percentage (0-100) format
                if confidence > 1.0:
                    confidence = confidence / 100.0
                    print(f"Converted confidence from percentage: {confidence_raw}% -> {confidence}")
            
            print(f"Extracted class_name: {class_name}")
            print(f"Extracted confidence: {confidence}")
            print(f"Extracted probabilities: {probabilities}")
            
            if class_name == 'Unknown' or confidence == 0.0:
                print("Warning: Could not extract valid prediction from API response")
                return self._get_default_result(filename, error="Could not parse API response - invalid format")
            
            # Normalize class name
            class_name = self._normalize_class_name(class_name)
            
            # Determine condition
            condition = get_condition_from_disease(class_name)
            
            # Get recommendations
            recommendations = get_treatment_recommendations(class_name, confidence)
            
            # If no probabilities extracted, create from single prediction
            if not probabilities:
                probabilities[class_name] = confidence
            
            print(f"Final parsed result:")
            print(f"  Condition: {condition}")
            print(f"  Disease: {class_name}")
            print(f"  Confidence: {confidence * 100:.1f}%")
            
            return {
                'filename': filename,
                'condition': condition,
                'disease_name': class_name.replace('-Pechay', '') if condition == 'Healthy' else class_name,
                'confidence': round(confidence * 100, 1) if confidence <= 1.0 else round(confidence, 1),
                'probabilities': probabilities,
                'recommendations': recommendations,
                'treatment': recommendations.get('action', '')
            }
            
        except Exception as e:
            print(f"Error parsing Roboflow result: {e}")
            import traceback
            traceback.print_exc()
            return self._get_default_result(filename, error=f"Parse error: {str(e)}")
    
    def _normalize_class_name(self, class_name: str) -> str:
        """Normalize class name from API to match disease_info format"""
        class_name = str(class_name).strip()
        
        # Map common variations
        name_mapping = {
            'healthy': 'Healthy-Pechay',
            'healthy-pechay': 'Healthy-Pechay',
            'alternaria': 'Alternaria',
            'blackrot': 'Blackrot',
            'black-rot': 'Blackrot',
            'leaf-spot': 'Leaf Spot'
        }
        
        normalized = name_mapping.get(class_name.lower(), class_name)
        return normalized
    
    def _validate_pechay_image(self, image_path: str) -> Dict:
        """
        Validate if image contains a pechay leaf
        Filters out: faces, non-green colors, round objects, walls, other objects
        """
        validation_errors = []
        reasons = []
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'is_valid': False,
                    'message': 'Invalid image file',
                    'reasons': ['Could not read image file']
                }
            
            # 1. Check for faces
            if FACE_RECOGNITION_AVAILABLE:
                try:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_image)
                    if len(face_locations) > 0:
                        validation_errors.append('face_detected')
                        reasons.append(f'Image contains {len(face_locations)} face(s) - not a pechay leaf')
                except Exception as e:
                    print(f"Face detection error: {e}")
            
            # 2. Check for green color (pechay leaves are green)
            green_validation = self._check_green_color(image)
            if not green_validation['is_green']:
                validation_errors.append('no_green_color')
                reasons.append(f"Image lacks green color ({green_validation['green_percentage']:.1f}% green detected, need at least 10%)")
            
            # 3. Check for round objects (using contour detection)
            round_objects = self._detect_round_objects(image)
            if round_objects['has_round_objects'] and round_objects['round_score'] > 0.7:
                validation_errors.append('round_objects')
                reasons.append(f"Image contains round objects (score: {round_objects['round_score']:.2f}) - likely not a leaf")
            
            # 4. Check for walls/backgrounds (using edge detection and texture analysis)
            wall_detection = self._detect_walls_backgrounds(image)
            if wall_detection['is_wall']:
                validation_errors.append('wall_background')
                reasons.append("Image appears to be a wall or background - not a pechay leaf")
            
            # 5. Use YOLO to detect non-leaf objects
            if YOLO_AVAILABLE:
                yolo_validation = self._validate_with_yolo(image_path)
                if not yolo_validation['is_valid']:
                    validation_errors.extend(yolo_validation.get('errors', []))
                    reasons.extend(yolo_validation.get('reasons', []))
            
            # If any validation errors, image is invalid
            if validation_errors:
                return {
                    'is_valid': False,
                    'message': 'Image does not appear to be a pechay leaf',
                    'reasons': reasons,
                    'errors': validation_errors
                }
            
            return {
                'is_valid': True,
                'message': 'Image validated as pechay leaf'
            }
            
        except Exception as e:
            print(f"Validation error: {e}")
            import traceback
            traceback.print_exc()
            # If validation fails, allow processing (fail open)
            return {
                'is_valid': True,
                'message': 'Validation check failed, proceeding with detection'
            }
    
    def _check_green_color(self, image: np.ndarray) -> Dict:
        """Check if image contains green colors (pechay leaves are green)"""
        try:
            # Convert to HSV color space
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define green color range (HSV)
            lower_green = np.array([35, 50, 50])  # Lower bound for green
            upper_green = np.array([85, 255, 255])  # Upper bound for green/yellow-green
            
            # Create mask for green colors
            mask = cv2.inRange(hsv, lower_green, upper_green)
            green_percentage = (np.sum(mask > 0) / mask.size) * 100
            
            return {
                'is_green': green_percentage >= 10,  # At least 10% green
                'green_percentage': green_percentage
            }
        except Exception as e:
            print(f"Green color check error: {e}")
            return {'is_green': True, 'green_percentage': 50}  # Fail open
    
    def _detect_round_objects(self, image: np.ndarray) -> Dict:
        """Detect round objects in image (leaves are typically not perfectly round)"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            round_score = 0.0
            has_round_objects = False
            
            for contour in contours:
                if cv2.contourArea(contour) < 100:  # Ignore small contours
                    continue
                
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                
                area = cv2.contourArea(contour)
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Perfect circle has circularity = 1.0
                if circularity > 0.8:  # Very round
                    round_score = max(round_score, circularity)
                    has_round_objects = True
            
            return {
                'has_round_objects': has_round_objects,
                'round_score': round_score
            }
        except Exception as e:
            print(f"Round object detection error: {e}")
            return {'has_round_objects': False, 'round_score': 0.0}
    
    def _detect_walls_backgrounds(self, image: np.ndarray) -> Dict:
        """Detect walls or plain backgrounds (not leaves)"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate gradient magnitude (walls have low variation)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            std_dev = np.std(gradient_magnitude)
            mean_gradient = np.mean(gradient_magnitude)
            
            # Walls/backgrounds have very low variation and low gradient
            is_wall = std_dev < 10 and mean_gradient < 20
            
            # Check for uniform color (walls are often uniform)
            color_std = np.std(image.reshape(-1, 3), axis=0)
            is_uniform = np.mean(color_std) < 15
            
            return {
                'is_wall': is_wall or is_uniform,
                'std_dev': std_dev,
                'mean_gradient': mean_gradient
            }
        except Exception as e:
            print(f"Wall detection error: {e}")
            return {'is_wall': False}
    
    def _validate_with_yolo(self, image_path: str) -> Dict:
        """Use YOLO to detect non-leaf objects (persons, cars, etc.)"""
        errors = []
        reasons = []
        
        try:
            # Load YOLOv8n (nano) model for general object detection
            model = YOLO('yolov8n.pt')  # Pre-trained COCO model
            
            # Run inference
            results = model(image_path)
            
            # COCO class names that are NOT leaves
            non_leaf_classes = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush'
            ]
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = model.names[cls]
                        
                        # If confidence is high and it's a non-leaf object, flag it
                        if conf > 0.5 and class_name.lower() in [c.lower() for c in non_leaf_classes]:
                            errors.append(f'yolo_{class_name}')
                            reasons.append(f'YOLO detected {class_name} (confidence: {conf:.2f}) - not a pechay leaf')
            
            return {
                'is_valid': len(errors) == 0,
                'errors': errors,
                'reasons': reasons
            }
            
        except Exception as e:
            print(f"YOLO validation error: {e}")
            # Fail open - if YOLO fails, allow processing
            return {
                'is_valid': True,
                'errors': [],
                'reasons': []
            }
    
    def _get_default_result(self, filename: str, error: Optional[str] = None, validation_details: Optional[Dict] = None) -> Dict:
        """Get default result when API fails - returns error result, not random data"""
        from utils.disease_info import get_treatment_recommendations, get_condition_from_disease
        
        # Return an error result instead of random data
        # This ensures users know when the API fails
        return {
            'filename': filename,
            'condition': 'Unknown',
            'disease_name': 'Detection Failed',
            'confidence': 0.0,
            'probabilities': {},
            'recommendations': {
                'title': 'Detection Error',
                'tips': ['Please try uploading the image again', 'Check your internet connection', 'Ensure the image is a valid pechay leaf photo'],
                'action': f'Error: {error or "API unavailable or failed"}',
                'urgency': 'high'
            },
            'treatment': f'Detection service unavailable. {error or "Please try again later."}',
            'error': error,
            'validation_details': validation_details,
            'is_fallback': True  # Flag to indicate this is not a real API result
        }

