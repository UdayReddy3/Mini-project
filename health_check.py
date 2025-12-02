"""
🏥 Health Check Diagnostic Script
Validates all components of the Plant Disease Detection System
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

print("\n" + "="*70)
print("🏥 PLANT DISEASE DETECTION - HEALTH CHECK")
print("="*70 + "\n")

def check_section(title):
    print(f"\n{'='*70}")
    print(f"🔍 {title}")
    print(f"{'='*70}")

def success(msg):
    print(f"✅ {msg}")

def warning(msg):
    print(f"⚠️  {msg}")

def error(msg):
    print(f"❌ {msg}")

def info(msg):
    print(f"ℹ️  {msg}")

# ============================================================================
# 1. CHECK DIRECTORY STRUCTURE
# ============================================================================
check_section("DIRECTORY STRUCTURE")

required_dirs = [
    'data',
    'data/PlantVillage',
    'data/PlantVillage/raw',
    'data/PlantVillage/raw/color',
    'models',
    'uploads',
    'diagnostics',
    'precautions_extra'
]

for dir_path in required_dirs:
    full_path = os.path.join('.', dir_path)
    if os.path.isdir(full_path):
        success(f"Directory exists: {dir_path}")
    else:
        warning(f"Directory missing: {dir_path}")

# ============================================================================
# 2. CHECK REQUIRED FILES
# ============================================================================
check_section("REQUIRED FILES")

required_files = {
    'models/plant_disease_model.h5': 'Base trained model',
    'models/class_names.json': 'Disease class names mapping',
    'app.py': 'Streamlit web application',
    'predict.py': 'Prediction module',
    'precautions.py': 'Disease precautions database',
    'auth.py': 'Authentication module',
    'db.py': 'Database module',
    'language.py': 'Multi-language support',
    'requirements.txt': 'Python dependencies'
}

for file_path, description in required_files.items():
    if os.path.isfile(file_path):
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        if file_size > 100:  # Model files are large
            success(f"{file_path} ({file_size:.1f} MB) - {description}")
        else:
            success(f"{file_path} - {description}")
    else:
        error(f"{file_path} NOT FOUND - {description}")

# ============================================================================
# 3. CHECK DISEASE DATA
# ============================================================================
check_section("DISEASE DATA")

try:
    with open('models/class_names.json', 'r') as f:
        class_names = json.load(f)
    
    success(f"Loaded {len(class_names)} disease classes")
    
    # Show sample classes
    info("Sample disease classes:")
    for i, cls in enumerate(class_names[:5]):
        print(f"  {i+1}. {cls}")
    if len(class_names) > 5:
        print(f"  ... and {len(class_names) - 5} more")
    
    # Count Paddy classes
    paddy_classes = [c for c in class_names if c.startswith('Paddy_')]
    if paddy_classes:
        success(f"Found {len(paddy_classes)} Paddy/Rice disease classes")
    else:
        warning("No Paddy/Rice classes found (PlantVillage only model)")

except Exception as e:
    error(f"Failed to load class names: {e}")

# ============================================================================
# 4. CHECK PRECAUTIONS DATABASE
# ============================================================================
check_section("PRECAUTIONS DATABASE")

try:
    from precautions import DISEASE_PRECAUTIONS, precaution_generator
    
    num_diseases = len(DISEASE_PRECAUTIONS)
    success(f"Precautions database contains {num_diseases} disease entries")
    
    # Check data completeness
    complete_count = 0
    incomplete_diseases = []
    
    for disease, data in DISEASE_PRECAUTIONS.items():
        required_keys = [
            'disease_name', 'severity', 'description', 'symptoms',
            'precautions', 'chemical_treatment', 'natural_treatment',
            'time_to_recovery', 'yield_impact', 'cost_effectiveness'
        ]
        
        if all(key in data for key in required_keys):
            complete_count += 1
        else:
            missing_keys = [k for k in required_keys if k not in data]
            incomplete_diseases.append((disease, missing_keys))
    
    success(f"{complete_count}/{num_diseases} diseases have complete data")
    
    if incomplete_diseases:
        warning(f"Found {len(incomplete_diseases)} diseases with incomplete data:")
        for disease, missing_keys in incomplete_diseases[:3]:
            print(f"  - {disease}: missing {', '.join(missing_keys)}")

except Exception as e:
    error(f"Failed to load precautions: {e}")

# ============================================================================
# 5. CHECK PYTHON DEPENDENCIES
# ============================================================================
check_section("PYTHON DEPENDENCIES")

dependencies = {
    'tensorflow': 'Deep Learning Framework',
    'keras': 'Keras API',
    'streamlit': 'Web Framework',
    'numpy': 'Numerical Computing',
    'pandas': 'Data Processing',
    'opencv-python': 'Image Processing (cv2)',
    'pillow': 'Image Processing (PIL)',
    'plotly': 'Interactive Visualization',
    'scikit-learn': 'Machine Learning Utils',
    'sqlite3': 'Database (built-in)'
}

for package, description in dependencies.items():
    try:
        if package == 'opencv-python':
            __import__('cv2')
        elif package == 'pillow':
            __import__('PIL')
        elif package == 'scikit-learn':
            __import__('sklearn')
        elif package == 'sqlite3':
            __import__('sqlite3')
        else:
            __import__(package)
        success(f"{package} - {description}")
    except ImportError:
        error(f"{package} NOT INSTALLED - {description}")

# ============================================================================
# 6. TEST MODEL LOADING
# ============================================================================
check_section("MODEL LOADING TEST")

try:
    from tensorflow import keras
    
    model_path = 'models/plant_disease_model.h5'
    
    if os.path.isfile(model_path):
        try:
            model = keras.models.load_model(model_path)
            success(f"Model loaded successfully from {model_path}")
            
            info(f"Model Architecture:")
            print(f"  - Input Shape: {model.input_shape}")
            print(f"  - Output Shape: {model.output_shape}")
            print(f"  - Total Layers: {len(model.layers)}")
            print(f"  - Total Parameters: {model.count_params():,}")
            
        except Exception as e:
            error(f"Failed to load model: {e}")
    else:
        error(f"Model file not found: {model_path}")

except Exception as e:
    error(f"TensorFlow not available: {e}")

# ============================================================================
# 7. TEST PREDICTION MODULE
# ============================================================================
check_section("PREDICTION MODULE TEST")

try:
    from predict import DiseasePredictor
    
    try:
        predictor = DiseasePredictor(
            model_path='models/plant_disease_model.h5',
            class_names_path='models/class_names.json'
        )
        success("DiseasePredictor initialized successfully")
        success(f"Loaded {len(predictor.class_names)} disease classes in predictor")
        
    except Exception as e:
        error(f"Failed to initialize predictor: {e}")

except Exception as e:
    error(f"Prediction module error: {e}")

# ============================================================================
# 8. TEST DATABASE
# ============================================================================
check_section("DATABASE TEST")

try:
    from db import init_database, hash_password
    
    try:
        init_database()
        success("Database initialized/verified")
        
        # Test password hashing
        test_password = "test123"
        hashed = hash_password(test_password)
        success(f"Password hashing works (hash length: {len(hashed)})")
        
    except Exception as e:
        error(f"Database error: {e}")

except Exception as e:
    error(f"DB module error: {e}")

# ============================================================================
# 9. CHECK DATA VOLUME
# ============================================================================
check_section("DATA VOLUME ANALYSIS")

try:
    data_dir = 'data/PlantVillage/raw/color'
    if os.path.isdir(data_dir):
        total_images = 0
        class_counts = {}
        
        for class_name in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):
                image_files = [f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
                count = len(image_files)
                class_counts[class_name] = count
                total_images += count
        
        success(f"Total images in dataset: {total_images:,}")
        success(f"Total disease classes: {len(class_counts)}")
        
        if class_counts:
            min_count = min(class_counts.values())
            max_count = max(class_counts.values())
            avg_count = total_images / len(class_counts)
            
            info(f"Image distribution:")
            print(f"  - Minimum per class: {min_count}")
            print(f"  - Maximum per class: {max_count}")
            print(f"  - Average per class: {avg_count:.0f}")
            
            # Check for imbalance
            if max_count > 2 * min_count:
                warning("Dataset is imbalanced - some classes have significantly fewer images")
    else:
        warning(f"Data directory not found: {data_dir}")

except Exception as e:
    error(f"Data analysis error: {e}")

# ============================================================================
# 10. SYSTEM SUMMARY
# ============================================================================
check_section("SYSTEM SUMMARY")

print("""
📊 RECOMMENDED NEXT STEPS:

1. **If ALL checks passed (✅):**
   - Run: streamlit run app.py
   - Open: http://localhost:8501
   - Start making predictions!

2. **If dependency checks failed (❌):**
   - Run: pip install -r requirements.txt
   - Retry health check

3. **If model/data checks failed (❌):**
   - Ensure PlantVillage dataset is in data/PlantVillage/raw/color/
   - Run: python model_fast.py (to train model)
   - Retry health check

4. **If precautions are incomplete (⚠️):**
   - Run: python enrich_precautions_gemini.py (if API configured)
   - Or manually add missing disease data in precautions.py

5. **For Paddy rice diseases:**
   - Download Paddy dataset from Kaggle
   - Place in: data/PlantVillage/raw/color/Paddy_*/ folders
   - Run: python fine_tune.py
   - Run: python fast_finetune_paddy.py

""")

print("="*70)
print(f"✅ Health Check Complete - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70 + "\n")
