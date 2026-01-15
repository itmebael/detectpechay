"""
Database service for Supabase operations
"""
from config.database import supabase, TABLES, STORAGE_BUCKETS
from typing import Dict, List, Optional, Any
import uuid
from datetime import datetime
import io
import base64

class DatabaseService:
    """Service for database operations"""
    
    @staticmethod
    def save_detection(user_id: str, image_path: str, detection_result: Dict) -> Optional[Dict]:
        """Save detection result to detection_results table"""
        try:
            detection_data = {
                'filename': detection_result.get('filename', 'unknown.jpg'),
                'condition': detection_result.get('condition', 'Unknown'),
                'confidence': float(detection_result.get('confidence', 0.0)),
                'image_path': image_path,
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'all_probabilities': detection_result.get('probabilities', {}),
                'recommendations': detection_result.get('recommendations', {})
            }
            
            # Add disease_name if available
            if detection_result.get('disease_name'):
                detection_data['disease_name'] = detection_result.get('disease_name')
            
            response = supabase.table(TABLES['detections']).insert(detection_data).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            print(f"Error saving detection: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def get_user_detections(user_id: str, limit: int = 100, condition: Optional[str] = None) -> List[Dict]:
        """Get detection history for user from detection_results table"""
        try:
            query = supabase.table(TABLES['detections'])\
                .select('*')\
                .eq('user_id', user_id)
            
            if condition:
                query = query.eq('condition', condition)
            
            response = query.order('timestamp', desc=True)\
                .limit(limit)\
                .execute()
            
            # Format results to match template expectations
            formatted_results = []
            for row in (response.data if response.data else []):
                formatted_results.append({
                    'filename': row.get('filename', 'unknown.jpg'),
                    'condition': row.get('condition', 'Unknown'),
                    'disease_name': row.get('disease_name'),
                    'confidence': float(row.get('confidence', 0.0)),
                    'timestamp': row.get('timestamp', row.get('created_at', '')),
                    'image_path': row.get('image_path', ''),
                    'recommendations': row.get('recommendations', {}),
                    'treatment': row.get('recommendations', {}).get('action', '') if isinstance(row.get('recommendations'), dict) else ''
                })
            
            return formatted_results
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
