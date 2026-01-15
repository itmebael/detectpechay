"""
Utility to load YOLOv9 dataset configuration
"""
import yaml
import os
from typing import Dict, List, Optional

def load_yolo_dataset_config(data_yaml_path: str) -> Dict:
    """
    Load YOLOv9 dataset configuration from data.yaml
    Returns: Dictionary with dataset configuration
    """
    try:
        with open(data_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update paths to use current directory
        dataset_dir = os.path.dirname(data_yaml_path)
        
        # Update paths if they're absolute or relative
        if 'train' in config:
            train_path = config['train']
            if not os.path.isabs(train_path):
                # Make path relative to dataset directory
                config['train'] = os.path.join(dataset_dir, 'train', 'images')
        if 'val' in config:
            val_path = config['val']
            if not os.path.isabs(val_path):
                config['val'] = os.path.join(dataset_dir, 'valid', 'images')
        if 'test' in config:
            test_path = config['test']
            if not os.path.isabs(test_path):
                config['test'] = os.path.join(dataset_dir, 'test', 'images')
        
        return config
    except Exception as e:
        print(f"Error loading dataset config: {e}")
        return {
            'nc': 3,
            'names': ['Alternaria', 'Blackrot', 'Healthy-Pechay']
        }

def get_class_names(data_yaml_path: Optional[str] = None) -> List[str]:
    """Get class names from dataset config"""
    if data_yaml_path and os.path.exists(data_yaml_path):
        config = load_yolo_dataset_config(data_yaml_path)
        return config.get('names', ['Alternaria', 'Blackrot', 'Healthy-Pechay'])
    return ['Alternaria', 'Blackrot', 'Healthy-Pechay']







