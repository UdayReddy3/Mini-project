import json
import os

# Simulate the loading with SMART merge logic
merged = {}

files_to_load = ['precautions_complete.json', 'precautions_enriched_manual.json', 'precautions_enriched_all.json']

for fname in files_to_load:
    path = os.path.join('precautions_extra', fname)
    if os.path.isfile(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    print(f"Loading {fname}:")
                    print(f"  - Total entries: {len(data)}")
                    
                    # Check a sample Paddy entry
                    if 'Paddy_bacterial_leaf_blight' in data:
                        sample = data['Paddy_bacterial_leaf_blight']
                        print(f"  - Paddy sample has disease_name: {'disease_name' in sample and bool(sample.get('disease_name'))}")
                    
                    # Smart merge
                    for key, value in data.items():
                        if key not in merged:
                            merged[key] = value
                        else:
                            # Check if new data is better - has disease_name and old doesn't
                            if value.get('disease_name') and not merged[key].get('disease_name'):
                                merged[key] = value
                            # Or if new data is more complete
                            elif len(value) > len(merged[key]):
                                # Merge by updating, not replacing
                                merged[key].update(value)
                    
                    print(f"  - Merged total now: {len(merged)}\n")
        except Exception as e:
            print(f"Error: {e}\n")

# Check final result
print(f"\nFinal merged dict:")
print(f"  Total entries: {len(merged)}")

# Check a Paddy entry
p = merged.get('Paddy_bacterial_leaf_blight')
if p:
    print(f"  Paddy sample:")
    print(f"    - disease_name: {p.get('disease_name')}")
    print(f"    - All keys: {list(p.keys())[:5]}")
