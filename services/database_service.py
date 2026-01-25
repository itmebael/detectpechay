"""
Database service for Supabase operations
"""
from config.database import supabase, TABLES, STORAGE_BUCKETS
from typing import Dict, List, Optional, Any
import uuid
from datetime import datetime
import io
import base64
import os

class DatabaseService:
    """Service for database operations"""
    
    @staticmethod
    def save_detection(user_id: str, image_path: str, detection_result: Dict) -> Optional[Dict]:
        """Save detection result to detection_results table"""
        try:
            condition = detection_result.get('condition', 'Unknown')
            allowed_conditions = ['Healthy', 'Diseased', 'Not Pechay']
            if condition not in allowed_conditions:
                print(f"Warning: Invalid condition '{condition}', defaulting to 'Diseased'")
                condition = 'Diseased'
            
            # Map to schema: id, user_id, image_path, condition, confidence, created_at, disease_name, filename
            detection_data = {
                'condition': condition,
                'confidence': float(detection_result.get('confidence', 0.0)),
                'image_path': image_path,
                'user_id': user_id,
                'created_at': datetime.now().isoformat(),
                'filename': detection_result.get('filename') or os.path.basename(image_path)
            }

            disease_name = detection_result.get('disease_name')
            if disease_name:
                disease_name_str = str(disease_name).strip()
                if disease_name_str and disease_name_str != 'Unknown':
                    detection_data['disease_name'] = disease_name_str

            for attempt in range(2):
                print(f"Saving detection to database (attempt {attempt + 1}): {detection_data}")
                try:
                    response = supabase.table(TABLES['detections']).insert(detection_data).execute()
                    
                    if response.data and len(response.data) > 0:
                        print(f"✓ Detection saved successfully: {response.data[0].get('id')}")
                        return response.data[0]
                    else:
                        print("⚠ Warning: Insert succeeded but no data returned")
                        return None
                except Exception as e_inner:
                    print(f"❌ Error on insert attempt {attempt + 1}: {e_inner}")
                    if 'PGRST204' in str(e_inner) and attempt == 0:
                        # If column error, try removing optional fields that might not exist
                        detection_data.pop('disease_name', None)
                        continue
                    import traceback
                    traceback.print_exc()
                    return None
        except Exception as e:
            print(f"❌ Error saving detection: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def get_user_detections(user_id: str, limit: int = 100, condition: Optional[str] = None) -> List[Dict]:
        """Get detection history for user from detection_results table"""
        try:
            # Select all fields including id
            query = supabase.table(TABLES['detections'])\
                .select('id, user_id, image_path, condition, confidence, created_at, disease_name, filename, recommendation')\
                .eq('user_id', user_id)
            
            if condition:
                query = query.eq('condition', condition)
            
            response = query.order('created_at', desc=True)\
                .limit(limit)\
                .execute()
            
            # Format results to match template expectations
            formatted_results = []
            for row in (response.data if response.data else []):
                # Extract filename from image_path if not present
                filename = row.get('filename')
                if not filename and row.get('image_path'):
                    filename = os.path.basename(row.get('image_path'))
                
                formatted_results.append({
                    'id': row.get('id'),  # Add ID for delete functionality
                    'filename': filename or 'unknown.jpg',
                    'condition': row.get('condition', 'Unknown'),
                    'disease_name': row.get('disease_name'),
                    'confidence': float(row.get('confidence', 0.0)),
                    'timestamp': row.get('created_at', ''),
                    'image_path': row.get('image_path', ''),
                    'treatment': row.get('recommendation', '')
                })
            
            return formatted_results
        except Exception as e:
            print(f"❌ Error getting user detections: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    @staticmethod
    def delete_detection(user_id: str, detection_id: str) -> bool:
        """Delete a detection result by ID (only if owned by user)"""
        try:
            # First verify the detection belongs to the user
            check_response = supabase.table(TABLES['detections'])\
                .select('id')\
                .eq('id', detection_id)\
                .eq('user_id', user_id)\
                .execute()
            
            if not check_response.data or len(check_response.data) == 0:
                print(f"Detection {detection_id} not found or doesn't belong to user {user_id}")
                return False
            
            # Delete the detection
            delete_response = supabase.table(TABLES['detections'])\
                .delete()\
                .eq('id', detection_id)\
                .eq('user_id', user_id)\
                .execute()
            
            if delete_response.data:
                print(f"✓ Detection {detection_id} deleted successfully")
                return True
            else:
                print(f"⚠ Warning: Delete succeeded but no data returned")
                return False
        except Exception as e:
            print(f"❌ Error deleting detection: {e}")
            import traceback
            traceback.print_exc()
            return False
        except Exception as e:
            print(f"Error getting detections: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    @staticmethod
    def get_dashboard_stats(user_id: Optional[str] = None) -> Dict:
        """Get dashboard statistics from detection_results table"""
        try:
            if not user_id:
                print("Warning: No user_id provided for dashboard stats")
                return {
                    'total_scans': 0,
                    'healthy_leaves': 0,
                    'diseased_leaves': 0,
                    'success_rate': 0
                }
            
            query = supabase.table(TABLES['detections']).select('condition, user_id')
            query = query.eq('user_id', user_id)
            
            response = query.execute()
            
            data = response.data if response.data else []
            total = len(data)
            healthy = len([d for d in data if d.get('condition') == 'Healthy'])
            diseased = total - healthy
            
            return {
                'total_scans': total,
                'healthy_leaves': healthy,
                'diseased_leaves': diseased,
                'success_rate': round((healthy / total * 100) if total > 0 else 0, 1)
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            import traceback
            traceback.print_exc()
            return {
                'total_scans': 0,
                'healthy_leaves': 0,
                'diseased_leaves': 0,
                'success_rate': 0
            }
    
    @staticmethod
    def get_dataset_stats(user_id: Optional[str] = None) -> Dict:
        """Get dataset statistics from dataset_images table"""
        try:
            query = supabase.table(TABLES['dataset_images']).select('label')
            
            if user_id:
                query = query.eq('user_id', user_id)
            
            response = query.execute()
            
            data = response.data if response.data else []
            healthy_count = len([d for d in data if d.get('label') == 'Healthy'])
            diseased_count = len([d for d in data if d.get('label') == 'Diseased'])
            
            return {
                'total_images': len(data),
                'healthy_count': healthy_count,
                'diseased_count': diseased_count
            }
        except Exception as e:
            print(f"Error getting dataset stats: {e}")
            return {
                'total_images': 0,
                'healthy_count': 0,
                'diseased_count': 0
            }
    
    @staticmethod
    def add_dataset_image(user_id: Optional[str], filename: str, label: str, image_path: str, split: str = 'train', metadata: Optional[Dict] = None) -> Optional[Dict]:
        try:
            data = {
                'filename': filename,
                'label': label,
                'image_path': image_path,
                'split': split,
                'metadata': metadata or {}
            }
            
            if user_id:
                data['user_id'] = user_id
            
            response = supabase.table(TABLES['dataset_images']).insert(data).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            print(f"Error adding dataset image: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def get_dataset_images(user_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
        try:
            query = supabase.table(TABLES['dataset_images']).select('*')
            if user_id:
                query = query.eq('user_id', user_id)
            
            response = query.order('created_at', desc=True).limit(limit).execute()
            rows = response.data if response.data else []
            
            images: List[Dict[str, Any]] = []
            for row in rows:
                images.append({
                    'filename': row.get('filename', ''),
                    'label': row.get('label', ''),
                    'image_url': row.get('image_path', ''),
                    'created_at': row.get('created_at') or row.get('timestamp', '')
                })
            
            return images
        except Exception as e:
            print(f"Error getting dataset images: {e}")
            import traceback
            traceback.print_exc()
            return []

    @staticmethod
    def save_petchay_dataset_entry(
        filename: str,
        condition: str,
        disease_name: Optional[str],
        image_url: str,
        embedding: Optional[List[float]],
        user_id: Optional[str] = None
    ) -> Optional[Dict]:
        try:
            data: Dict[str, Any] = {
                'filename': filename,
                'condition': condition,
                'image_url': image_url,
                'embedding': embedding
            }
            
            if disease_name:
                data['disease_name'] = disease_name
            if user_id:
                data['user_id'] = user_id
            
            response = supabase.table(TABLES['petchay_dataset']).insert(data).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            print(f"Error saving petchay_dataset entry: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def create_user(username: str, email: str, password: str) -> Optional[Dict]:
        """Create a new user in the users table"""
        try:
            user_data = {
                'username': username,
                'email': email,
                'password': password
            }
            
            response = supabase.table(TABLES['users']).insert(user_data).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            print(f"Error creating user: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def get_user_by_email(email: str) -> Optional[Dict]:
        """Get user by email"""
        try:
            response = supabase.table(TABLES['users'])\
                .select('*')\
                .eq('email', email)\
                .execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            print(f"Error getting user by email: {e}")
            return None
    
    @staticmethod
    def get_user_by_username(username: str) -> Optional[Dict]:
        """Get user by username"""
        try:
            response = supabase.table(TABLES['users'])\
                .select('*')\
                .eq('username', username)\
                .execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            print(f"Error getting user by username: {e}")
            return None
