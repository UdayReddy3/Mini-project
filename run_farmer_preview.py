import os
import glob
from predict import DiseasePredictor
from precautions import precaution_generator

# Local helper functions (match app.py behavior)
import re

def _extract_dosage_amount(text: str):
    if not text:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*(g/L|kg/ha|ppm|mg/L|%|g/ha)", text, flags=re.I)
    if not m:
        return None
    low = float(m.group(1))
    high = float(m.group(2)) if m.group(2) else None
    unit = m.group(3).lower()
    return (low, high, unit)

def _convert_to_per_ha(low, high, unit):
    if unit in ('g/l', 'mg/l', 'ppm'):
        spray_vol = 500.0
        if unit == 'mg/l' or unit == 'ppm':
            factor = 0.001
        else:
            factor = 1.0
        low_gha = low * factor * spray_vol
        if high:
            high_gha = high * factor * spray_vol
            return f"~{low_gha:.0f}–{high_gha:.0f} g/ha (assuming {int(spray_vol)} L/ha spray volume)"
        return f"~{low_gha:.0f} g/ha (assuming {int(spray_vol)} L/ha spray volume)"
    elif unit == 'g/ha':
        if high:
            return f"{low:.0f}–{high:.0f} g/ha"
        return f"{low:.0f} g/ha"
    elif unit == 'kg/ha':
        if high:
            return f"{low:.2f}–{high:.2f} kg/ha"
        return f"{low:.2f} kg/ha"
    elif unit == '%':
        return "Use label % concentration; follow spray instructions per product"
    else:
        return None


def build_farmer_summary(precautions_data, result):
    lines = []
    lines.append(f"Crop Image: {result.get('original_filename', 'uploaded_image')}")
    lines.append(f"Predicted: {result.get('disease_class', 'Unknown').replace('_', ' ')} ({result.get('confidence',0):.1f}% confidence)")
    lines.append("")
    lines.append("Immediate Actions:")
    for i, a in enumerate(precaution_generator.get_immediate_actions(result.get('disease_class')), 1):
        lines.append(f"{i}. {a}")
    lines.append("")
    lines.append("Simple Treatments:")
    chems = precautions_data.get('chemical_treatment', []) or []
    naturals = precautions_data.get('natural_treatment', []) or []
    if chems:
        lines.append("Chemical options:")
        for t in chems:
            lines.append(f"- {t}")
            parsed = _extract_dosage_amount(t)
            if parsed:
                per_ha = _convert_to_per_ha(*parsed)
                if per_ha:
                    lines.append(f"  -> Approx: {per_ha}")
    if naturals:
        lines.append("Natural/Organic options:")
        for t in naturals:
            lines.append(f"- {t}")
    lines.append("")
    lines.append(f"Fertilizer recommendation: {precautions_data.get('fertilizer_recommendation', 'Follow soil test')}")
    lines.append(f"Estimated recovery time: {precautions_data.get('time_to_recovery', 'Varies')}")
    return "\n".join(lines)


# Find a sample test image
base = os.path.join('data', 'PlantVillage', 'raw', 'color')
pattern = os.path.join(base, '**', '*.JPG')
files = glob.glob(pattern, recursive=True)
if not files:
    pattern = os.path.join(base, '**', '*.jpg')
    files = glob.glob(pattern, recursive=True)

if not files:
    print('No test images found under data/PlantVillage/raw/color')
    exit(1)

sample = files[0]
print('Using test image:', sample)

# Load predictor (small local loader)
model_path = 'models/plant_disease_model_finetuned.h5' if os.path.exists('models/plant_disease_model_finetuned.h5') else 'models/plant_disease_model.h5'
pred = DiseasePredictor(model_path=model_path, class_names_path='models/class_names.json')
res = pred.predict_disease(sample)
res['original_filename'] = os.path.basename(sample)

prec = precaution_generator.get_smart_precautions(res['disease_class'])
summary = build_farmer_summary(prec, res)
print('\n----- FARMER MODE PREVIEW -----\n')
print(summary)
print('\n----- END PREVIEW -----\n')
