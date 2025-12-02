import json

# Simulate the merging logic
DISEASE_PRECAUTIONS = {
    "Paddy_bacterial_leaf_blight": {
        "fertilizer_recommendation": "xyz",
        "recommended_products": [],
        "source": "dummy"
    }
}

print("Before merge:")
print(f"  Keys: {list(DISEASE_PRECAUTIONS['Paddy_bacterial_leaf_blight'].keys())}")
print(f"  disease_name: {DISEASE_PRECAUTIONS['Paddy_bacterial_leaf_blight'].get('disease_name')}")

# Load from JSON
with open('precautions_extra/precautions_enriched_all.json') as f:
    json_data = json.load(f)

paddy_data = json_data['Paddy_bacterial_leaf_blight']
print(f"\nJSON data has disease_name: {paddy_data.get('disease_name')}")

# Simulate the merge
if 'Paddy_bacterial_leaf_blight' in DISEASE_PRECAUTIONS:
    print(f"\nKey found in DISEASE_PRECAUTIONS!")
    DISEASE_PRECAUTIONS['Paddy_bacterial_leaf_blight'].update(paddy_data)
    print(f"After update.update():")
    print(f"  Keys: {list(DISEASE_PRECAUTIONS['Paddy_bacterial_leaf_blight'].keys())[:5]}")
    print(f"  disease_name: {DISEASE_PRECAUTIONS['Paddy_bacterial_leaf_blight'].get('disease_name')}")
else:
    print(f"\nKey NOT found in DISEASE_PRECAUTIONS")
