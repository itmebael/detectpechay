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

# Try to use Roboflow inference package, fallback to HTTP requests
try:
    from inference_sdk import InferenceHTTPClient
    INFERENCE_SDK_AVAILABLE = True
    print("✓ Roboflow inference_sdk available")
except ImportError:
    INFERENCE_SDK_AVAILABLE = False
    print("⚠ inference_sdk not available, will use HTTP requests")

# Use HTTP requests directly as fallback
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
        # Use serverless.roboflow.com for workflows
        self.api_url = "https://serverless.roboflow.com"
        self.api_key = "B7RL7B0aD47us70qgevM"
        self.workspace_name = "bael"
        self.workflow_id = "custom-workflow-3"  # Can be changed to "custom-workflow-2" if needed
        
        # Initialize inference client if available
        self.inference_client = None
        if INFERENCE_SDK_AVAILABLE:
            try:
                self.inference_client = InferenceHTTPClient(
                    api_url="https://detect.roboflow.com",
                    api_key=self.api_key
                )
                print("✓ InferenceHTTPClient initialized")
            except Exception as e:
                print(f"⚠ Failed to initialize InferenceHTTPClient: {e}")
                self.inference_client = None
    
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
            
            pechay_validation = self._validate_pechay_image(image_path)
            if not pechay_validation.get('is_valid'):
                reasons = pechay_validation.get('reasons') or []
                if not reasons:
                    reasons = ['Image does not appear to be a pechay leaf']
                return {
                    'filename': filename,
                    'condition': 'Not Pechay',
                    'disease_name': 'Not Pechay Leaf',
                    'confidence': 100.0,
                    'probabilities': {'Not Pechay Leaf': 1.0},
                    'recommendations': {
                        'title': 'Not a Pechay leaf image',
                        'tips': reasons,
                        'action': 'Please upload a clear image of a single pechay leaf on plain background.',
                        'urgency': 'medium'
                    },
                    'treatment': 'Detection skipped because the image is not a pechay leaf.',
                    'error': None,
                    'validation_details': pechay_validation,
                    'is_fallback': True,
                    'method': 'Validation/YOLO'
                }
            
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
                    
                    # Try different API base URLs - prioritize serverless.roboflow.com for workflows
                    base_urls = [
                        "https://serverless.roboflow.com",  # Serverless workflows domain (primary)
                        "https://api.roboflow.com",  # Standard API domain (fallback)
                        "https://detect.roboflow.com",  # Detection API domain (fallback)
                    ]
                    
                    # Construct URL - try different endpoint formats for workflows
                    workflow_url = None
                    for base_url in base_urls:
                        # Try different endpoint formats
                        endpoint_formats = [
                            f"{base_url}/workflows/{self.workspace_name}/{self.workflow_id}",  # Direct workflow endpoint
                            f"{base_url}/infer/workflows/{self.workspace_name}/{self.workflow_id}",  # Inference workflow format
                            f"{base_url}/workflow/{self.workspace_name}/{self.workflow_id}",  # Alternative format
                        ]
                        
                        for test_url in endpoint_formats:
                            workflow_url = test_url
                            print(f"Trying API endpoint: {workflow_url}")
                            
                            # Try different payload formats
                            payload_formats = [
                                # Format 1: Workflow input format
                                {
                                    "inputs": {
                                        "image": {"type": "base64", "value": img_base64}
                                    },
                                    "api_key": self.api_key
                                },
                                # Format 2: Direct image format
                                {
                                    "image": img_base64,
                                    "api_key": self.api_key
                                },
                                # Format 3: With api_key in query
                                {
                                    "inputs": {
                                        "image": {"type": "base64", "value": img_base64}
                                    }
                                }
                            ]
                            
                            response = None
                            for payload_idx, payload in enumerate(payload_formats):
                                if payload_idx > 0:
                                    print(f"  Trying payload format {payload_idx + 1}...")
                                
                                # If api_key not in payload, add to URL
                                url_with_key = f"{test_url}?api_key={self.api_key}" if "api_key" not in payload else test_url
                                
                                try:
                                    response = requests.post(
                                        url_with_key,
                                        json=payload if "api_key" in payload else payload,
                                        headers={"Content-Type": "application/json"},
                                        params={"api_key": self.api_key} if "api_key" not in payload else None,
                                        timeout=30
                                    )
                                    
                                    print(f"  API Response Status: {response.status_code}")
                                    
                                    if response.status_code == 200:
                                        try:
                                            result = response.json()
                                        except Exception as json_error:
                                            print(f"⚠️ Failed to parse JSON response: {json_error}")
                                            print(f"Raw response text (first 500 chars): {response.text[:500]}")
                                            # Try next payload format
                                            if payload_idx < len(payload_formats) - 1:
                                                continue
                                            break
                                        
                                        print(f"\n=== API Response Structure ===")
                                        print(f"Response type: {type(result)}")
                                        if isinstance(result, dict):
                                            print(f"Response keys: {list(result.keys())}")
                                        print(f"Response preview (first 2000 chars): {str(result)[:2000]}")
                                        
                                        # Parse Roboflow API results
                                        detection_result = self._parse_roboflow_result(result, filename)
                                        
                                        # Check if parsing was successful (has valid data, not just error)
                                        if detection_result.get('error') and not detection_result.get('disease_name'):
                                            print(f"⚠️ Parsing returned error: {detection_result.get('error')}")
                                            # Still return it so user sees the error message
                                        
                                        detection_result['method'] = f'Roboflow API ({base_url})'
                                        detection_result['is_fallback'] = False
                                        
                                        print(f"\n=== Parsed Detection Result ===")
                                        print(f"Condition: {detection_result.get('condition')}")
                                        print(f"Disease: {detection_result.get('disease_name')}")
                                        print(f"Confidence: {detection_result.get('confidence')}%")
                                        if detection_result.get('error'):
                                            print(f"Error: {detection_result.get('error')}")
                                        
                                        return detection_result
                                    
                                    elif response.status_code != 404:  # Non-404 error, try next payload format
                                        error_text = response.text[:200]
                                        print(f"  API Error ({response.status_code}): {error_text}")
                                        if payload_idx == len(payload_formats) - 1:
                                            # Last payload format, try next endpoint
                                            break
                                        continue  # Try next payload format
                                    else:
                                        # 404 - endpoint doesn't exist, try next endpoint format
                                        print(f"  Endpoint not found (404), trying next format...")
                                        break  # Break payload loop, try next endpoint
                                
                                except requests.exceptions.RequestException as req_error:
                                    print(f"  Request error: {str(req_error)[:200]}")
                                    if payload_idx == len(payload_formats) - 1:
                                        break  # Try next endpoint format
                                    continue  # Try next payload format
                            
                            # If we got here, this endpoint format didn't work, try next
                            if response and response.status_code == 200:
                                break  # Success, exit endpoint loop
                        
                        # If successful, exit base_url loop
                        if response and response.status_code == 200:
                            break
                    
                    # If we get here and no successful response, return error
                    if not response or (hasattr(response, 'status_code') and response.status_code != 200):
                        return self._get_default_result(
                            filename,
                            error="All Roboflow API endpoint formats failed. Please check configuration and API response logs."
                        )
                                
                except requests.exceptions.RequestException as req_error:
                    print(f"Request error: {str(req_error)[:200]}")
                    return self._get_default_result(
                        filename,
                        error=f"API request failed: {str(req_error)[:200]}"
                    )
                    
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
            
            # Check for error responses first
            if isinstance(result, dict):
                if 'error' in result or 'message' in result or 'status' in result:
                    error_msg = result.get('error') or result.get('message') or result.get('detail')
                    status = result.get('status')
                    if status and status != 200:
                        print(f"API returned error status: {status}, message: {error_msg}")
                        return self._get_default_result(filename, error=f"API error (status {status}): {error_msg}")
                    elif error_msg:
                        print(f"API returned error message: {error_msg}")
                        return self._get_default_result(filename, error=f"API error: {error_msg}")
            
            # Try different possible response formats
            if isinstance(result, dict):
                # Format 1: outputs array (Roboflow workflow format)
                if 'outputs' in result:
                    outputs = result['outputs']
                    print(f"  Found 'outputs' key with {len(outputs) if isinstance(outputs, list) else 'non-list'} items")
                    if isinstance(outputs, list) and len(outputs) > 0:
                        # Get first output item
                        first_output = outputs[0]
                        print(f"  First output type: {type(first_output)}, keys: {list(first_output.keys()) if isinstance(first_output, dict) else 'not a dict'}")
                        if isinstance(first_output, dict):
                            # Check for predictions inside the output
                            if 'predictions' in first_output:
                                pred_data = first_output['predictions']
                                print(f"  Found 'predictions' in output: type={type(pred_data)}")
                                
                                # Handle nested predictions structure: outputs[0].predictions.predictions[]
                                if isinstance(pred_data, dict):
                                    print(f"  pred_data is dict, keys: {list(pred_data.keys())}")
                                    if 'predictions' in pred_data:
                                        predictions = pred_data['predictions']
                                        print(f"  ✓ Extracted nested predictions list: {len(predictions) if isinstance(predictions, list) else 'not a list'} items")
                                        if isinstance(predictions, list) and len(predictions) > 0:
                                            print(f"  First prediction: {predictions[0]}")
                                    elif 'image' in pred_data:
                                        # If predictions dict only has 'image', check if there's a predictions list elsewhere
                                        print(f"  pred_data has 'image' key, checking for predictions list...")
                                        # Try to find predictions in other keys
                                        for key, value in pred_data.items():
                                            if key != 'image' and isinstance(value, list):
                                                predictions = value
                                                print(f"  Found predictions list in key '{key}': {len(predictions)} items")
                                                break
                                    else:
                                        # Single prediction object
                                        predictions = [pred_data]
                                        print(f"  Wrapped single prediction dict in list")
                                elif isinstance(pred_data, list):
                                    predictions = pred_data
                                    print(f"  Using predictions as direct list: {len(predictions)} items")
                                else:
                                    predictions = [pred_data]
                                    print(f"  Wrapped single prediction in list")
                            elif 'prediction' in first_output:
                                prediction = first_output['prediction']
                                print(f"  Found single 'prediction' key")
                            elif 'class' in first_output or 'predicted_class' in first_output:
                                prediction = first_output
                                print(f"  Found class directly in first_output")
                # Format 2: Direct predictions array
                elif 'predictions' in result:
                    predictions = result['predictions']
                # Format 3: Nested in workflow output
                elif 'output' in result:
                    output = result['output']
                    if isinstance(output, dict) and 'predictions' in output:
                        predictions = output['predictions']
                    elif isinstance(output, list):
                        predictions = output
                # Format 4: Direct prediction object
                elif 'class' in result or 'predicted_class' in result:
                    prediction = result
                # Format 5: Workflow steps output
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
                # Format 6: Check if result itself is a list
                elif isinstance(result, list):
                    predictions = result
            
            # Extract prediction data
            print(f"\n=== Extracting Prediction Data ===")
            print(f"predictions variable: type={type(predictions)}, is_list={isinstance(predictions, list)}, len={len(predictions) if isinstance(predictions, (list, dict)) else 'N/A'}")
            
            if predictions:
                if isinstance(predictions, list) and len(predictions) > 0:
                    print(f"  ✓ predictions is a list with {len(predictions)} items")
                    print(f"  First prediction item: {predictions[0]}")
                    # Get highest confidence prediction
                    try:
                        prediction = max(predictions, key=lambda p: float(
                            p.get('confidence', p.get('score', p.get('confidence_score', 0.0)))
                        ))
                        print(f"  ✓ Selected prediction (highest confidence): {prediction}")
                    except Exception as e:
                        print(f"  ⚠️ Error selecting max prediction: {e}, using first item")
                        prediction = predictions[0]
                    
                    # Build probabilities dict from all predictions
                    for pred in predictions:
                        pred_class = pred.get('class', pred.get('predicted_class', pred.get('label', 'Unknown')))
                        pred_conf_raw = pred.get('confidence', pred.get('score', pred.get('confidence_score', 0.0)))
                        try:
                            pred_conf = float(pred_conf_raw)
                            
                            # Handle percentage format
                            if pred_conf > 1.0:
                                pred_conf = pred_conf / 100.0
                            
                            if pred_class and pred_conf > 0:
                                probabilities[pred_class] = pred_conf
                                print(f"  ✓ Added to probabilities: {pred_class} = {pred_conf}")
                        except Exception as e:
                            print(f"  ⚠️ Error processing prediction: {e}")
                elif isinstance(predictions, dict):
                    prediction = predictions
                    print(f"  ✓ predictions is a dict, using as single prediction: {prediction}")
            else:
                print(f"  ⚠️ predictions is None or empty - attempting direct extraction...")
                # Try to extract directly from result as last resort
                if isinstance(result, dict) and 'outputs' in result:
                    print(f"  Attempting direct extraction from outputs structure...")
                    try:
                        outputs = result['outputs']
                        if isinstance(outputs, list) and len(outputs) > 0:
                            first_output = outputs[0]
                            if isinstance(first_output, dict) and 'predictions' in first_output:
                                pred_data = first_output['predictions']
                                if isinstance(pred_data, dict) and 'predictions' in pred_data:
                                    predictions = pred_data['predictions']
                                    if isinstance(predictions, list) and len(predictions) > 0:
                                        print(f"  ✓ Found predictions list: {len(predictions)} items")
                                        # Get highest confidence prediction
                                        prediction = max(predictions, key=lambda p: float(
                                            p.get('confidence', p.get('score', p.get('confidence_score', 0.0)))
                                        ))
                                        print(f"  ✓ Direct extraction successful: {prediction}")
                                        # Also set predictions for probabilities
                                        for pred in predictions:
                                            pred_class = pred.get('class', pred.get('predicted_class', pred.get('label', 'Unknown')))
                                            pred_conf_raw = pred.get('confidence', pred.get('score', pred.get('confidence_score', 0.0)))
                                            try:
                                                pred_conf = float(pred_conf_raw)
                                                if pred_conf > 1.0:
                                                    pred_conf = pred_conf / 100.0
                                                if pred_class and pred_conf > 0:
                                                    probabilities[pred_class] = pred_conf
                                            except:
                                                pass
                    except Exception as e:
                        print(f"  ⚠️ Direct extraction failed: {e}")
                        import traceback
                        traceback.print_exc()
            
            # Extract class and confidence from prediction
            print(f"\n=== Extracting Class and Confidence ===")
            print(f"prediction variable: type={type(prediction)}, value={prediction}")
            
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
                print(f"  Initial class_name extraction: '{class_name}'")
                
                # Clean up class_name - strip whitespace and ensure it's a string
                if class_name and isinstance(class_name, str):
                    class_name = class_name.strip()
                    print(f"  ✓ Cleaned class_name: '{class_name}'")
                elif not class_name or class_name == 'Unknown':
                    # Last resort: try to extract from string representation
                    if isinstance(prediction, dict):
                        pred_str = str(prediction)
                        # Try to find class in the string
                        import re
                        class_match = re.search(r'"class":\s*"([^"]+)"', pred_str)
                        if class_match:
                            class_name = class_match.group(1).strip()
                            print(f"  ✓ Extracted class_name from string: '{class_name}'")
                
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
                print(f"  ✓ Extracted confidence_raw: {confidence_raw}, confidence: {confidence}")
                
                # Handle confidence values that might already be in percentage (0-100) format
                if confidence > 1.0:
                    # Assume it's already a percentage, convert to decimal
                    confidence = confidence / 100.0
                    print(f"  ✓ Converted confidence from percentage: {confidence_raw}% -> {confidence}")
                else:
                    print(f"  ✓ Confidence is already in 0-1 format: {confidence}")
            
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
                if isinstance(predictions, list) and len(predictions) == 0:
                    print("No predictions returned by API; treating as Healthy result")
                    class_name = 'Healthy-Pechay'
                    confidence = 0.9
                else:
                    print("⚠️ Warning: Could not extract valid prediction from API response, trying additional formats...")
                    if isinstance(result, dict):
                        for key, value in result.items():
                            if isinstance(value, str) and value.strip() and len(value) < 100:
                                if not value.startswith('http') and value.lower() not in ['true', 'false', 'null']:
                                    disease_keywords = ['healthy', 'diseased', 'alternaria', 'blackrot', 'leaf spot', 'pechay']
                                    if any(keyword in value.lower() for keyword in disease_keywords):
                                        class_name = value
                                        confidence = 0.5
                                        print(f"⚠️ Found potential class name in field '{key}': {class_name}")
                                        break
                    
                    if class_name == 'Unknown' or confidence == 0.0:
                        print(f"\n❌ ERROR: Could not parse API response - invalid format")
                        print(f"Result type: {type(result)}")
                        
                        error_details = []
                        if isinstance(result, dict):
                            error_details.append(f"Response keys: {list(result.keys())}")
                            for key, value in list(result.items())[:5]:
                                if isinstance(value, (str, int, float, bool)):
                                    error_details.append(f"{key}: {value}")
                                elif isinstance(value, (list, dict)):
                                    error_details.append(f"{key}: {type(value).__name__} with {len(value)} items")
                        elif isinstance(result, list):
                            error_details.append(f"Response is a list with {len(result)} items")
                            if len(result) > 0:
                                error_details.append(f"First item type: {type(result[0])}")
                        else:
                            error_details.append(f"Response type: {type(result)}")
                        
                        try:
                            import json
                            result_str = json.dumps(result, indent=2, default=str)
                            print(f"Full response (first 5000 chars):\n{result_str[:5000]}")
                            error_details.append(f"Response preview: {result_str[:500]}")
                        except:
                            result_str = str(result)
                            print(f"Full response (first 5000 chars): {result_str[:5000]}")
                            error_details.append(f"Response preview: {result_str[:500]}")
                        
                        error_msg = "Could not parse API response. " + " | ".join(error_details)
                        
                        return self._get_default_result(filename, error=error_msg)
            
            # Clean and normalize class name
            if class_name and isinstance(class_name, str):
                # Remove any JSON-like artifacts or extra text
                import re
                # Extract just the disease name if it's mixed with other text
                # Look for patterns like "class": "Alternaria" or just "Alternaria"
                class_match = re.search(r'"class":\s*"([^"]+)"', class_name)
                if class_match:
                    class_name = class_match.group(1).strip()
                else:
                    # Remove common prefixes/suffixes
                    class_name = re.sub(r'^.*?(?:class|name|disease)[":\s]*', '', class_name, flags=re.IGNORECASE)
                    class_name = re.sub(r'["\s]*confidence.*$', '', class_name, flags=re.IGNORECASE)
                    class_name = class_name.strip()
            
            # Normalize class name
            class_name = self._normalize_class_name(class_name) if class_name else 'Unknown'
            
            # Determine condition
            condition = get_condition_from_disease(class_name)
            
            # Get recommendations (treatment)
            recommendations = get_treatment_recommendations(class_name, confidence)
            
            # If no probabilities extracted, create from single prediction
            if not probabilities:
                probabilities[class_name] = confidence
            
            # Clean disease name for display (remove -Pechay suffix for Healthy)
            display_disease_name = class_name.replace('-Pechay', '') if condition == 'Healthy' else class_name
            
            print(f"Final parsed result:")
            print(f"  Condition: {condition}")
            print(f"  Disease: {display_disease_name}")
            print(f"  Confidence: {confidence * 100:.1f}%")
            print(f"  Treatment: {recommendations.get('action', '')[:100]}...")
            
            return {
                'filename': filename,
                'condition': condition,
                'disease_name': display_disease_name,
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
            
            if FACE_RECOGNITION_AVAILABLE:
                try:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_image)
                    if len(face_locations) > 0:
                        validation_errors.append('face_detected')
                        reasons.append(f'Image contains {len(face_locations)} face(s) - not a pechay leaf')
                except Exception as e:
                    print(f"Face detection error: {e}")
            skin_validation = self._detect_skin_like(image)
            if skin_validation.get('has_skin'):
                validation_errors.append('face_like')
                reasons.append(f"Image appears to contain human skin ({skin_validation.get('skin_percentage', 0.0):.1f}% of pixels) - not a pechay leaf")
            
            # 2. Check for green color (pechay leaves are green)
            green_validation = self._check_green_color(image)
            if not green_validation['is_green']:
                validation_errors.append('no_green_color')
                reasons.append(f"Image lacks green color ({green_validation['green_percentage']:.1f}% green detected, need at least 10%)")
            
            # 3. Check for round objects (using contour detection)
            # Relaxed: Don't block on round objects as leaves can be roundish
            round_objects = self._detect_round_objects(image)
            # if round_objects['has_round_objects'] and round_objects['round_score'] > 0.7:
            #     validation_errors.append('round_objects')
            #     reasons.append(f"Image contains round objects (score: {round_objects['round_score']:.2f}) - likely not a leaf")
            
            # 4. Check for walls/backgrounds (using edge detection and texture analysis)
            wall_detection = self._detect_walls_backgrounds(image)
            if wall_detection['is_wall']:
                if green_validation['green_percentage'] < 10:
                    validation_errors.append('wall_background')
                    reasons.append("Image appears to be a wall or background - not a pechay leaf")
            
            # Relaxed: Leaf shape check is too strict for zoomed-in images
            # leaf_shape = self._check_leaf_shape(image)
            # if not leaf_shape['is_leaf_like']:
            #    validation_errors.append('shape_invalid')
            #    reasons.append("Image shape does not match typical pechay leaf")
            
            if validation_errors:
                critical_errors = []
                skin_percentage = skin_validation.get('skin_percentage', 0.0)
                green_pct = green_validation.get('green_percentage', 0.0)
                for err in validation_errors:
                    if err == 'face_detected':
                        critical_errors.append(err)
                    elif err in ['no_green_color', 'wall_background']:
                        critical_errors.append(err)
                    elif err.startswith('yolo_'):
                        critical_errors.append(err)
                if critical_errors:
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
                'is_green': green_percentage >= 10,
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

    def _check_leaf_shape(self, image: np.ndarray) -> Dict:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            is_leaf_like = False
            max_area = 0.0
            best_aspect = 0.0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 800:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if w == 0 or h == 0:
                    continue
                aspect = max(w, h) / min(w, h)
                if 1.1 <= aspect <= 5.0:
                    is_leaf_like = True
                    if area > max_area:
                        max_area = area
                        best_aspect = aspect
            return {
                'is_leaf_like': is_leaf_like,
                'max_area': max_area,
                'aspect_ratio': best_aspect
            }
        except Exception as e:
            print(f"Leaf-shape detection error: {e}")
            return {
                'is_leaf_like': True,
                'max_area': 0.0,
                'aspect_ratio': 0.0
            }
    
    def _detect_skin_like(self, image: np.ndarray) -> Dict:
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower1 = np.array([0, 30, 50])
            upper1 = np.array([20, 255, 255])
            lower2 = np.array([160, 30, 50])
            upper2 = np.array([180, 255, 255])
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask_hsv = cv2.bitwise_or(mask1, mask2)
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            lower_ycrcb = np.array([0, 133, 77])
            upper_ycrcb = np.array([255, 173, 127])
            mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
            mask = cv2.bitwise_or(mask_hsv, mask_ycrcb)
            skin_percentage = (np.sum(mask > 0) / mask.size) * 100
            has_skin = skin_percentage >= 20
            return {
                'has_skin': has_skin,
                'skin_percentage': skin_percentage
            }
        except Exception as e:
            print(f"Skin-like detection error: {e}")
            return {'has_skin': False, 'skin_percentage': 0.0}
    
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
            
            color_std = np.std(image.reshape(-1, 3), axis=0)
            mean_color_std = np.mean(color_std)
            is_wall = std_dev < 8 and mean_gradient < 15 and mean_color_std < 10
            
            return {
                'is_wall': is_wall,
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
                'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
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
        
        # If API fails, do not show "Detection Failed". Treat as Not Pechay Leaf for clarity.
        return {
            'filename': filename,
            'condition': 'Diseased',
            'disease_name': 'Not Pechay Leaf',
            'confidence': 0.0,
            'probabilities': {},
            'recommendations': {
                'title': 'Not a Pechay leaf image',
                'tips': [
                    'Upload a clear image of a single pechay leaf',
                    'Avoid faces and non-leaf objects',
                    'Use plain background for better detection'
                ],
                'action': 'Please upload a proper pechay leaf image.',
                'urgency': 'medium'
            },
            'treatment': 'Detection skipped due to invalid image or API error.',
            'error': error,
            'validation_details': validation_details,
            'is_fallback': True,
            'method': 'Validation/Error'
        }
