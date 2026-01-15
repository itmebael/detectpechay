"""
Analytics service for generating dashboard analytics data
"""
from config.database import supabase, TABLES
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

class AnalyticsService:
    """Service for generating analytics data"""
    
    @staticmethod
    def get_analytics_data(user_id: Optional[str] = None, days: int = 7) -> Optional[Dict]:
        """Generate analytics data for the dashboard"""
        try:
            query = supabase.table(TABLES['detections']).select('*')
            
            if user_id:
                query = query.eq('user_id', user_id)
            
            # Get last N days of data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            response = query.order('timestamp', desc=True).execute()
            
            if not response.data:
                # Return empty/default data structure instead of None
                end_date = datetime.now()
                daily_labels = []
                daily_values = []
                for i in range(7):
                    date = (end_date - timedelta(days=i)).strftime('%Y-%m-%d')
                    daily_labels.insert(0, date)
                    daily_values.insert(0, 0)
                
                return {
                    'daily_labels': daily_labels,
                    'daily_values': daily_values,
                    'healthy_count': 0,
                    'diseased_count': 0,
                    'confidence_labels': ['0-20', '21-40', '41-60', '61-80', '81-100'],
                    'confidence_values': [0, 0, 0, 0, 0]
                }
            
            # Filter by date range
            data = []
            for row in response.data:
                timestamp_str = row.get('timestamp') or row.get('created_at')
                if timestamp_str:
                    try:
                        # Parse timestamp
                        if isinstance(timestamp_str, str):
                            # Remove timezone info for parsing
                            timestamp_str = timestamp_str.split('+')[0].split('.')[0]
                            row_date = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S')
                        else:
                            row_date = timestamp_str
                        
                        if row_date >= start_date:
                            data.append(row)
                    except:
                        # Include all data if date parsing fails
                        data.append(row)
                else:
                    data.append(row)
            
            if not data:
                # Return empty/default data structure instead of None
                end_date = datetime.now()
                daily_labels = []
                daily_values = []
                for i in range(7):
                    date = (end_date - timedelta(days=i)).strftime('%Y-%m-%d')
                    daily_labels.insert(0, date)
                    daily_values.insert(0, 0)
                
                return {
                    'daily_labels': daily_labels,
                    'daily_values': daily_values,
                    'healthy_count': 0,
                    'diseased_count': 0,
                    'confidence_labels': ['0-20', '21-40', '41-60', '61-80', '81-100'],
                    'confidence_values': [0, 0, 0, 0, 0]
                }
            
            # Generate daily activity data (last 7 days)
            daily_data = defaultdict(int)
            for row in data:
                timestamp_str = row.get('timestamp') or row.get('created_at')
                if timestamp_str:
                    try:
                        if isinstance(timestamp_str, str):
                            timestamp_str = timestamp_str.split('+')[0].split('.')[0]
                            row_date = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S')
                        else:
                            row_date = timestamp_str
                        date_key = row_date.strftime('%Y-%m-%d')
                        daily_data[date_key] += 1
                    except:
                        pass
            
            # Get last 7 days
            daily_labels = []
            daily_values = []
            for i in range(7):
                date = (end_date - timedelta(days=i)).strftime('%Y-%m-%d')
                daily_labels.insert(0, date)
                daily_values.insert(0, daily_data.get(date, 0))
            
            # Health distribution
            healthy_count = len([r for r in data if r.get('condition') == 'Healthy'])
            diseased_count = len([r for r in data if r.get('condition') == 'Diseased'])
            
            # Confidence level distribution
            confidence_ranges = {
                '0-20': 0,
                '21-40': 0,
                '41-60': 0,
                '61-80': 0,
                '81-100': 0
            }
            
            for row in data:
                confidence = float(row.get('confidence', 0))
                if confidence <= 20:
                    confidence_ranges['0-20'] += 1
                elif confidence <= 40:
                    confidence_ranges['21-40'] += 1
                elif confidence <= 60:
                    confidence_ranges['41-60'] += 1
                elif confidence <= 80:
                    confidence_ranges['61-80'] += 1
                else:
                    confidence_ranges['81-100'] += 1
            
            return {
                'daily_labels': daily_labels,
                'daily_values': daily_values,
                'healthy_count': healthy_count,
                'diseased_count': diseased_count,
                'confidence_labels': list(confidence_ranges.keys()),
                'confidence_values': list(confidence_ranges.values())
            }
        except Exception as e:
            print(f"Error generating analytics data: {e}")
            import traceback
            traceback.print_exc()
            # Return empty/default data structure instead of None
            end_date = datetime.now()
            daily_labels = []
            daily_values = []
            for i in range(7):
                date = (end_date - timedelta(days=i)).strftime('%Y-%m-%d')
                daily_labels.insert(0, date)
                daily_values.insert(0, 0)
            
            return {
                'daily_labels': daily_labels,
                'daily_values': daily_values,
                'healthy_count': 0,
                'diseased_count': 0,
                'confidence_labels': ['0-20', '21-40', '41-60', '61-80', '81-100'],
                'confidence_values': [0, 0, 0, 0, 0]
            }
