"""
Generate manual, curated pesticide/fungicide/insecticide and fertilizer recommendations
for all classes in `models/class_names.json` and save to
`precautions_extra/precautions_enriched_manual.json`.

This provides immediate, editable recommendations without external APIs.
"""
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
CLASS_PATH = BASE / 'models' / 'class_names.json'
OUT_PATH = BASE / 'precautions_extra' / 'precautions_enriched_manual.json'

with open(CLASS_PATH, 'r', encoding='utf-8') as f:
    classes = json.load(f)

# Helper templates for common diseases
FUNGICIDES = [
    {"product": "Mancozeb", "active": "Mancozeb", "note": "Protectant fungicide; rotate with other classes."},
    {"product": "Copper oxychloride", "active": "Copper", "note": "Broad-spectrum bactericide/fungicide; follow label."},
    {"product": "Azoxystrobin", "active": "Azoxystrobin", "note": "Systemic fungicide (strobilurin)."},
    {"product": "Tricyclazole", "active": "Tricyclazole", "note": "Effective against rice blast (use as per local rules)."}
]

BACTERICIDES = [
    {"product": "Copper-based bactericide (e.g., Copper oxychloride)", "active": "Copper", "note": "Use as preventive; follow local regulations."},
    {"product": "Streptomycin (where permitted)", "active": "Streptomycin", "note": "Use only if allowed by local authorities."}
]

INSECTICIDES = [
    {"product": "Neem oil formulations", "active": "Azadirachtin (natural)", "note": "Organic option for many pests."},
    {"product": "Imidacloprid", "active": "Imidacloprid", "note": "Systemic insecticide for sap-sucking pests; observe pollinator safety."},
    {"product": "Emamectin benzoate", "active": "Emamectin", "note": "For lepidopteran pests (follow thresholds)."}
]

VIRAL_CONTROL = [
    {"product": "Vector control (Imidacloprid)", "active": "Imidacloprid", "note": "No cure for viruses; manage vectors and use tolerant varieties."},
    {"product": "Reflective mulch / cultural", "active": "", "note": "Non-chemical method to reduce vector landing."}
]

# Generic fertilizers recommendations
FERTILIZERS = {
    "default": "Balanced NPK following soil test (e.g., DAP as basal; Urea for split N top-dress; MOP for K if needed).",
    "paddy": "DAP at transplanting; split Urea top-dressing; apply Zinc sulphate if Zn deficient.",
    "fruit_tree": "Balanced NPK plus micronutrients per soil test; apply organic manure annually.",
}

manual = {}
for cls in classes:
    entry = {}
    # Basic fertilizer advice
    if cls.startswith('Paddy_'):
        entry['fertilizer_recommendation'] = FERTILIZERS['paddy']
    elif cls.startswith('Tomato') or cls.startswith('Potato') or cls.startswith('Pepper'):
        entry['fertilizer_recommendation'] = FERTILIZERS['default']
    elif cls.startswith('Orange') or cls.startswith('Peach') or cls.startswith('Grape'):
        entry['fertilizer_recommendation'] = FERTILIZERS['fruit_tree']
    else:
        entry['fertilizer_recommendation'] = FERTILIZERS['default']

    # Build recommended products based on disease keywords
    products = []
    lc = cls.lower()
    if 'blast' in lc or 'powdery' in lc or 'leaf_mold' in lc or 'septoria' in lc or 'target_spot' in lc or 'early_blight' in lc or 'late_blight' in lc:
        products.extend(FUNGICIDES[:3])
    if 'bacterial' in lc or 'bacterial_spot' in lc or 'black_rot' in lc:
        products.extend(BACTERICIDES)
    if 'mosaic' in lc or 'virus' in lc or 'tungro' in lc or 'yellow_leaf_curl' in lc:
        products.extend(VIRAL_CONTROL)
    if 'spider' in lc or 'hispa' in lc or 'tungro' in lc:
        products.extend(INSECTICIDES[:2])
    if 'healthy' in lc:
        products.append({"product": "Use preventive cultural practices", "active": "", "note": "Maintain balance, monitor regularly."})

    # If no products identified, add a general advisory
    if not products:
        products.append({"product": "Copper-based fungicide/bactericide", "active": "Copper", "note": "General preventive option; follow label."})

    entry['recommended_products'] = products
    entry['source'] = 'manual_curated'  # mark as manual entries
    manual[cls] = entry

# Save to JSON
OUT_PATH.parent.mkdir(exist_ok=True, parents=True)
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(manual, f, indent=2, ensure_ascii=False)

print('Wrote manual recommendations to', OUT_PATH)
