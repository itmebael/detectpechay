"""
Database configuration and Supabase setup
"""
from supabase import create_client, Client
import os

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://zqkqmjlepigpwfykwzey.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "sb_publishable_HNgog4XZVoR6FqaKuzIcGQ_7yrDAjFn")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Table names (matching your Supabase schema)
TABLES = {
    'users': 'users',
    'detections': 'detection_results',
    'dataset_images': 'dataset_images',
    'petchay_dataset': 'petchay_dataset',
    'file_uploads': 'file_uploads',
    'yolo_files': 'yolo_files'
}

# Storage bucket names
STORAGE_BUCKETS = {
    'uploads': 'user-uploads',
    'dataset': 'dataset-images',
    'detections': 'detection-results'
}

