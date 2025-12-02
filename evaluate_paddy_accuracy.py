"""
Evaluate Paddy top-1 accuracy per class using the current model.
Saves results to `diagnostics/paddy_eval.json` and prints a summary.
"""
import os
import json
import random
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

MODEL_PATH = 'models/plant_disease_model_finetuned.h5'
CLASS_NAMES_PATH = 'models/class_names.json'
DATA_DIR = 'data/PlantVillage/raw/color'
OUTPUT_DIR = 'diagnostics'
SAMPLES_PER_CLASS = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

print('Loading model...')
model = keras.models.load_model(MODEL_PATH)
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)

paddy_classes = [c for c in class_names if c.startswith('Paddy_')]
print(f'Found {len(paddy_classes)} Paddy classes: {paddy_classes}')

results = {}

for cls in paddy_classes:
    cls_dir = os.path.join(DATA_DIR, cls)
    if not os.path.isdir(cls_dir):
        print(f'Warning: directory missing for {cls}')
        continue
    imgs = [os.path.join(cls_dir, p) for p in os.listdir(cls_dir) if p.lower().endswith(('.jpg','.jpeg','.png'))]
    if not imgs:
        print(f'No images found for {cls}')
        continue
    sample_imgs = random.sample(imgs, min(SAMPLES_PER_CLASS, len(imgs)))
    correct = 0
    total = 0
    details = []
    for img_path in sample_imgs:
        img = load_img(img_path, target_size=(192,192))
        arr = img_to_array(img) / 255.0
        inp = np.expand_dims(arr, axis=0)
        preds = model.predict(inp)[0]
        top_idx = preds.argmax()
        pred_class = class_names[top_idx]
        is_correct = (pred_class == cls)
        total += 1
        if is_correct:
            correct += 1
        details.append({'image': img_path, 'predicted': pred_class, 'confidence': float(preds[top_idx]), 'correct': is_correct})
    acc = correct / total if total>0 else None
    results[cls] = {'samples': total, 'correct': correct, 'accuracy': acc, 'details': details}
    print(f"{cls}: {correct}/{total} = {acc:.3f}")

out_path = os.path.join(OUTPUT_DIR, 'paddy_eval.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

print('\nSaved evaluation to', out_path)
print('Done')
