import json

# Load original dict keys (sample from code)
original_keys = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Apple___Apple_scab",
    "Paddy_bacterial_leaf_blight",
]

# Load JSON keys
with open('precautions_extra/precautions_enriched_all.json') as f:
    json_data = json.load(f)
    json_keys = set(json_data.keys())

print("Checking key matching:\n")
for key in original_keys:
    if key in json_keys:
        print(f"✓ {key} - FOUND in JSON")
    else:
        print(f"✗ {key} - NOT found in JSON")

# Check what Paddy keys are in the dict
from precautions import DISEASE_PRECAUTIONS
dict_keys = set(DISEASE_PRECAUTIONS.keys())
paddy_dict = [k for k in dict_keys if 'Paddy' in k or 'paddy' in k]
paddy_json = [k for k in json_keys if 'Paddy' in k or 'paddy' in k]

print(f"\n\nPaddy classes comparison:")
print(f"In original dict: {len(paddy_dict)}")
[print(f"  {k}") for k in paddy_dict[:3]]

print(f"\nIn JSON: {len(paddy_json)}")
[print(f"  {k}") for k in paddy_json[:3]]

# Check if they're the same
if set(paddy_dict) == set(paddy_json):
    print(f"\n✓ Paddy keys match perfectly!")
else:
    missing_in_json = set(paddy_dict) - set(paddy_json)
    missing_in_dict = set(paddy_json) - set(paddy_dict)
    print(f"\n✗ Keys don't match!")
    if missing_in_json:
        print(f"  Missing in JSON: {missing_in_json}")
    if missing_in_dict:
        print(f"  Missing in dict: {missing_in_dict}")
