"""
Image processing and validation utilities
"""
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Optional
import os

# Optional dependencies - don't block if not available
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("Warning: face_recognition not available. Face detection will be skipped.")

YOLO_AVAILABLE = False
YOLO_MODEL = None

class ImageProcessor:
    """Handles image validation and preprocessing"""
    
    @staticmethod
    def validate_color_gate(image: np.ndarray) -> Tuple[bool, str]:
        """
        Validate if image contains green/yellow-green colors (pechay leaf colors)
        Returns: (is_valid, message)
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define green color range (HSV)
        lower_green = np.array([35, 50, 50])  # Lower bound for green
        upper_green = np.array([85, 255, 255])  # Upper bound for green/yellow-green
        
        # Create mask for green colors
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green_percentage = (np.sum(mask > 0) / mask.size) * 100
        
        if green_percentage < 10:  # Less than 10% green
            return False, f"Image contains only {green_percentage:.1f}% green colors. Not a valid pechay leaf."
        
        return True, f"Color validation passed ({green_percentage:.1f}% green)"
    
    @staticmethod
    def detect_digital_image(image_path: str) -> Tuple[bool, str]:
        """
        Detect if image is a real photo or digital/screenshot
        Returns: (is_digital, message)
        """
        image = cv2.imread(image_path)
        if image is None:
            return True, "Invalid image file"
        
        # Check for JPEG compression artifacts (real photos have them)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Real photos have more variation
        std_dev = np.std(gradient_magnitude)
        if std_dev < 10:  # Very low variation suggests digital image
            return True, "Image appears to be digital/screenshot"
        
        return False, "Image appears to be a real photo"
    
    @staticmethod
    def filter_faces_and_persons(image: np.ndarray) -> Tuple[bool, str]:
        """
        Check if image contains faces or persons (non-pechay objects)
        Returns: (has_faces_persons, message)
        """
        try:
            # Detect faces if face_recognition is available
            if FACE_RECOGNITION_AVAILABLE:
                try:
                    # Convert BGR to RGB for face_recognition
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces
                    face_locations = face_recognition.face_locations(rgb_image)
                    if len(face_locations) > 0:
                        return True, f"Image contains {len(face_locations)} face(s). Not a pechay leaf."
                except Exception as e:
                    print(f"Face detection error (skipping): {e}")
                    # Continue to other checks
            
            # Use YOLO for person/object detection if available
            if YOLO_AVAILABLE:
                global YOLO_MODEL
                if YOLO_MODEL is None:
                    try:
                        YOLO_MODEL = YOLO('yolov8n.pt')
                    except Exception as e:
                        print(f"Failed to load YOLO model: {e}")
                
                if YOLO_MODEL:
                    try:
                        results = YOLO_MODEL(image, verbose=False)
                        for r in results:
                            for box in r.boxes:
                                cls = int(box.cls[0])
                                conf = float(box.conf[0])
                                class_name = YOLO_MODEL.names[cls]
                                
                                # Filter out non-leaf objects
                                non_leaf_classes = ['person', 'car', 'bus', 'truck', 'bicycle', 'motorcycle']
                                if conf > 0.5 and class_name.lower() in [c.lower() for c in non_leaf_classes]:
                                    return True, f"Image contains {class_name} (confidence: {conf:.2f}). Not a pechay leaf."
                    except Exception as e:
                        print(f"YOLO detection error: {e}")
            
            # Fallback: check aspect ratio for portrait orientation
            h, w = image.shape[:2]
            aspect_ratio = h / w if w > 0 else 0
            if aspect_ratio > 2.5:
                return True, "Image aspect ratio suggests person/portrait. Not a pechay leaf."
            
            return False, "No faces or persons detected"
        except Exception as e:
            # If face detection fails, assume it's okay (don't block)
            return False, f"Face detection unavailable: {str(e)}"
    
    @staticmethod
    def detect_round_shape(image: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if image contains round/leaf-like shapes
        Returns: (has_round_shapes, roundness_score)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return False, 0.0
        
        # Check for round shapes
        max_roundness = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Skip small contours
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            # Roundness = 4π * area / perimeter²
            roundness = (4 * np.pi * area) / (perimeter ** 2)
            max_roundness = max(max_roundness, roundness)
        
        has_round = max_roundness > 0.5  # Threshold for roundness
        return has_round, max_roundness
    
    @staticmethod
    def validate_image_quality(image: np.ndarray) -> Tuple[bool, str, Dict]:
        """
        Comprehensive image validation
        Returns: (is_valid, message, validation_details)
        Note: Optional checks (face detection, YOLO) are skipped if dependencies are missing
        """
        details = {}
        
        # Check image size (required check)
        try:
            h, w = image.shape[:2]
            if w < 100 or h < 100:
                return False, "Image too small (minimum 100x100)", details
            details['size'] = f"{w}x{h}"
        except Exception as e:
            return False, f"Invalid image format: {str(e)}", details
        
        # Color gate validation (required check)
        try:
            color_valid, color_msg = ImageProcessor.validate_color_gate(image)
            details['color_validation'] = color_msg
            if not color_valid:
                return False, color_msg, details
        except Exception as e:
            print(f"Color validation error (skipping): {e}")
            details['color_validation'] = "Color validation skipped"
        
        # Face/person filtering (optional - skip if dependencies missing)
        try:
            has_faces, face_msg = ImageProcessor.filter_faces_and_persons(image)
            details['face_detection'] = face_msg
            if has_faces:
                # Only block if face was actually detected (not if check was skipped)
                if "unavailable" not in face_msg.lower() and "skipping" not in face_msg.lower():
                    return False, face_msg, details
        except Exception as e:
            print(f"Face detection check error (skipping): {e}")
            details['face_detection'] = "Face detection skipped"
        
        # Round shape detection (optional)
        try:
            has_round, roundness = ImageProcessor.detect_round_shape(image)
            details['roundness_score'] = roundness
            if has_round and roundness > 0.8:  # Very round objects are likely not leaves
                return False, f"Image contains highly round object (score: {roundness:.2f}). Not a pechay leaf.", details
        except Exception as e:
            print(f"Round shape detection error (skipping): {e}")
            details['roundness_score'] = "Round shape detection skipped"
        
        # Wall/background detection (optional)
        try:
            is_wall, wall_msg = ImageProcessor.detect_walls_or_backgrounds(image)
            details['background_detection'] = wall_msg
            if is_wall:
                return False, wall_msg, details
        except Exception as e:
            print(f"Wall detection error (skipping): {e}")
            details['background_detection'] = "Wall detection skipped"
        
        return True, "Image validation passed", details
    
    @staticmethod
    def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Preprocess image for model input
        """
        # Resize
        resized = cv2.resize(image, target_size)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    @staticmethod
    def detect_walls_or_backgrounds(image: np.ndarray) -> Tuple[bool, str]:
        """
        Detect if image is a wall or plain background (not a leaf)
        Returns: (is_wall, message)
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate histogram entropy (walls have low entropy)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_norm = hist / (hist.sum() + 1e-7)
            entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-7))
            
            # Calculate gradient magnitude (walls have low variation)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            std_dev = np.std(gradient_magnitude)
            mean_gradient = np.mean(gradient_magnitude)
            
            # Walls/backgrounds have very low variation, low gradient, and low entropy
            if entropy < 5.0 and std_dev < 10 and mean_gradient < 20:
                return True, "Image appears to be a plain background or wall. Not a pechay leaf."
            
            return False, "Background appears complex enough"
        except Exception as e:
            print(f"Wall detection error: {e}")
            return False, "Wall detection unavailable"
    
    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        """Load image from file path"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                # Try with PIL
                pil_image = Image.open(image_path)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None


