"""
Disease information and treatment recommendations
"""
from typing import Dict, List, Optional

DISEASE_INFO = {
    'Alternaria': {
        'name': 'Alternaria Leaf Spot',
        'description': 'Fungal disease causing dark spots on leaves',
        'symptoms': [
            'Dark brown to black circular spots',
            'Concentric rings on lesions',
            'Yellowing around spots',
            'Leaf curling and wilting'
        ],
        'treatment': """1. Remove and destroy infected leaves
2. Apply fungicide (Mancozeb or Copper-based)
3. Improve air circulation
4. Avoid overhead watering
5. Use resistant varieties if available
6. Practice crop rotation""",
        'urgency': 'medium',
        'severity': 'moderate'
    },
    'Blackrot': {
        'name': 'Black Rot',
        'description': 'Bacterial disease causing blackening of leaves',
        'symptoms': [
            'V-shaped yellow lesions',
            'Blackening of leaf veins',
            'Leaf drop',
            'Stunted growth'
        ],
        'treatment': """1. Remove affected plants immediately
2. Apply copper-based bactericide
3. Avoid overhead irrigation
4. Use disease-free seeds
5. Practice crop rotation (3-4 years)
6. Maintain proper spacing between plants
7. Avoid working in fields when plants are wet""",
        'urgency': 'high',
        'severity': 'severe'
    },
    'Healthy-Pechay': {
        'name': 'Healthy Pechay',
        'description': 'Plant is in good health',
        'symptoms': [],
        'treatment': """1. Continue current care practices
2. Regular monitoring for early detection
3. Maintain proper watering schedule
4. Ensure adequate sunlight
5. Apply balanced fertilizer as needed""",
        'urgency': 'low',
        'severity': 'none'
    },
    'Leaf Spot': {
        'name': 'Leaf Spot Disease',
        'description': 'Various fungal or bacterial leaf spot diseases',
        'symptoms': [
            'Small circular or irregular spots',
            'Brown or black lesions',
            'Yellow halos around spots',
            'Premature leaf drop'
        ],
        'treatment': """1. Remove infected leaves
2. Apply appropriate fungicide/bactericide
3. Improve drainage
4. Reduce leaf wetness
5. Use resistant varieties""",
        'urgency': 'medium',
        'severity': 'moderate'
    }
}

def get_disease_info(disease_name: str) -> Optional[Dict]:
    """Get information about a specific disease"""
    # Normalize disease name
    disease_key = disease_name.replace(' ', '').replace('-', '')
    for key, info in DISEASE_INFO.items():
        if key.lower() == disease_key.lower() or info['name'].lower() == disease_name.lower():
            return info
    return None

def get_treatment_recommendations(disease_name: str, confidence: float) -> Dict:
    """Get treatment recommendations based on disease and confidence"""
    disease_info = get_disease_info(disease_name)
    
    if not disease_info:
        return {
            'title': 'Unknown Disease',
            'tips': ['Consult with agricultural expert', 'Monitor plant closely', 'Take photo for expert review'],
            'action': 'Please consult with a plant pathologist for proper diagnosis.',
            'urgency': 'medium'
        }
    
    # Determine urgency based on disease and confidence
    urgency = disease_info['urgency']
    if confidence < 0.7:
        urgency = 'high'  # Low confidence means need expert review
    
    recommendations = {
        'title': f"{disease_info['name']} - Treatment Recommendations",
        'tips': disease_info['symptoms'][:3] if disease_info['symptoms'] else ['Monitor plant closely'],
        'action': disease_info['treatment'],
        'urgency': urgency
    }
    
    return recommendations

def get_condition_from_disease(disease_name: str) -> str:
    """Determine condition (Healthy/Diseased) from disease name"""
    if 'healthy' in disease_name.lower() or disease_name.lower() == 'healthy-pechay':
        return 'Healthy'
    return 'Diseased'







