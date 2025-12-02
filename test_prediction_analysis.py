"""
Deep Analysis of Prediction Pipeline
Tests the entire flow: Model -> Prediction -> Precautions -> Display
"""

import os
import json
import numpy as np
from PIL import Image
from predict import DiseasePredictor
from precautions import precaution_generator, DISEASE_PRECAUTIONS
import random

print("=" * 80)
print("🔬 DEEP PREDICTION ANALYSIS TEST")
print("=" * 80)

# 1. Load predictor
print("\n[1] Loading Model and Classes...")
try:
    predictor = DiseasePredictor(
        model_path='models/plant_disease_model_finetuned.h5',
        class_names_path='models/class_names.json'
    )
    print(f"✓ Model loaded successfully")
    print(f"✓ Total classes: {len(predictor.class_names)}")
    print(f"✓ Model input shape: {predictor.model.input_shape}")
    print(f"✓ Model output shape: {predictor.model.output_shape}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit(1)

# 2. Check precautions database
print("\n[2] Checking Precautions Database...")
print(f"✓ Total precaution entries: {len(DISEASE_PRECAUTIONS)}")
print(f"✓ Model classes: {len(predictor.class_names)}")

# 3. Cross-check mapping
print("\n[3] Checking Class-to-Precautions Mapping...")
mapped_count = 0
missing_names = []

for class_name in predictor.class_names:
    if class_name in DISEASE_PRECAUTIONS:
        data = DISEASE_PRECAUTIONS[class_name]
        disease_name = data.get('disease_name', 'N/A')
        if not disease_name or disease_name == 'Unknown':
            missing_names.append(class_name)
        mapped_count += 1

print(f"✓ Classes with precautions: {mapped_count}/{len(predictor.class_names)}")
print(f"✓ Classes without disease_name: {len(missing_names)}")
if missing_names:
    print(f"  Examples: {missing_names[:5]}")

# 4. Test prediction with a random image
print("\n[4] Finding Test Images...")
test_image_path = None
data_dir = 'data/PlantVillage/raw/color'

if os.path.exists(data_dir):
    # Find a random image file
    for root, dirs, files in os.walk(data_dir):
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if image_files:
            test_image_path = os.path.join(root, random.choice(image_files))
            break

if test_image_path:
    print(f"✓ Test image found: {test_image_path}")
    
    print("\n[5] Running Prediction...")
    try:
        result = predictor.predict_disease(test_image_path)
        
        print(f"\n📊 PREDICTION RESULTS:")
        print(f"  Predicted Class: {result['disease_class']}")
        print(f"  Confidence: {result['confidence']}%")
        print(f"  Timestamp: {result['timestamp']}")
        
        print(f"\n  Top 3 Predictions:")
        for i, (disease, conf) in enumerate(result['top_3_predictions'], 1):
            print(f"    {i}. {disease}: {conf:.2f}%")
        
        # 6. Check precautions for predicted disease
        print("\n[6] Checking Precautions for Predicted Disease...")
        predicted_disease = result['disease_class']
        
        if predicted_disease in DISEASE_PRECAUTIONS:
            precautions = DISEASE_PRECAUTIONS[predicted_disease]
            print(f"✓ Precautions found for: {predicted_disease}")
            print(f"\n  Disease Name: {precautions.get('disease_name', 'N/A')}")
            print(f"  Severity: {precautions.get('severity', 'N/A')}")
            print(f"  Description: {precautions.get('description', 'N/A')[:100]}...")
            
            symptoms = precautions.get('symptoms', [])
            print(f"  Symptoms ({len(symptoms)}):")
            for symptom in symptoms[:3]:
                print(f"    - {symptom}")
            
            treatments = precautions.get('chemical_treatment', [])
            print(f"  Chemical Treatments ({len(treatments)}):")
            for treatment in treatments[:2]:
                print(f"    - {treatment}")
            
            natural_treatments = precautions.get('natural_treatment', [])
            print(f"  Natural Treatments ({len(natural_treatments)}):")
            for treatment in natural_treatments[:2]:
                print(f"    - {treatment}")
        else:
            print(f"✗ No precautions found for: {predicted_disease}")
        
        # 7. Test precaution_generator methods
        print("\n[7] Testing Precaution Generator Methods...")
        
        immediate_actions = precaution_generator.get_immediate_actions(predicted_disease)
        print(f"✓ Immediate Actions ({len(immediate_actions)}):")
        for action in immediate_actions:
            print(f"  - {action}")
        
        severity, icon = precaution_generator.get_severity_level(predicted_disease)
        print(f"✓ Severity: {severity} {icon}")
        
        precautions_data = precaution_generator.get_precautions(predicted_disease)
        print(f"✓ Full precautions data keys: {list(precautions_data.keys())}")
        
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("✗ No test images found in data directory")

# 8. Summary
print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE")
print("=" * 80)
print("\nSUMMARY:")
print(f"  - Model: {len(predictor.class_names)} classes")
print(f"  - Precautions: {len(DISEASE_PRECAUTIONS)} entries")
print(f"  - Mapping: {mapped_count} classes have precautions")
print(f"  - Missing disease_name: {len(missing_names)} classes")
print("\nISSUES TO CHECK:")
if missing_names:
    print(f"  ⚠️  {len(missing_names)} classes don't have disease_name set")
    print(f"      These will show 'Unknown' in the dashboard")
print("\n")
