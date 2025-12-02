import sys

# Debug: manually load and merge
from precautions import _load_extra_precautions, _normalize_disease_name, DISEASE_PRECAUTIONS as orig_diseases

print("DEBUG: Checking loading and merging\n")

print(f"1. Original DISEASE_PRECAUTIONS size: {len(orig_diseases)}")
print(f"   Apple___Apple_scab in original: {'Apple___Apple_scab' in orig_diseases}")

if 'Apple___Apple_scab' in orig_diseases:
    print(f"   Original data: {list(orig_diseases['Apple___Apple_scab'].keys())[:3]}")

print(f"\n2. Loading extras...")
extras = _load_extra_precautions()
print(f"   Loaded {len(extras)} extra entries")

if 'Apple___Apple_scab' in extras:
    print(f"   Extra has Apple___Apple_scab: YES")
    print(f"   Extra data keys: {list(extras['Apple___Apple_scab'].keys())}")
else:
    print(f"   Extra has Apple___Apple_scab: NO")
    print(f"   Sample extra keys: {list(extras.keys())[:3]}")

print(f"\n3. Checking normalization...")
test_key = 'Apple___Apple_scab'
print(f"   Normalize('{test_key}'): {_normalize_disease_name(test_key)}")

# Test if merging should work
if 'Apple___Apple_scab' in orig_diseases and 'Apple___Apple_scab' in extras:
    print(f"\n4. Merging would work:")
    print(f"   Both have exact match: YES")
    
    # Try manual merge
    test_merge = orig_diseases['Apple___Apple_scab'].copy()
    print(f"   Before: {list(test_merge.keys())[:3]}")
    test_merge.update(extras['Apple___Apple_scab'])
    print(f"   After: {list(test_merge.keys())[:3]}")
    print(f"   disease_name after merge: {test_merge.get('disease_name')}")
