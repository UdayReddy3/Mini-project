from precautions import DISEASE_PRECAUTIONS

print("Checking DISEASE_PRECAUTIONS merging:")
print(f"\nTotal diseases: {len(DISEASE_PRECAUTIONS)}")

# Check Apple scab
if 'Apple___Apple_scab' in DISEASE_PRECAUTIONS:
    data = DISEASE_PRECAUTIONS['Apple___Apple_scab']
    print(f"\n✓ Apple___Apple_scab found")
    print(f"  - disease_name: {data.get('disease_name')}")
    print(f"  - severity: {data.get('severity')}")
    print(f"  - has_symptoms: {len(data.get('symptoms', [])) > 0}")
    print(f"  - has_precautions: {len(data.get('precautions', [])) > 0}")
else:
    print("\n✗ Apple___Apple_scab NOT found")

# Count how many have disease_name
count_with_name = sum(1 for v in DISEASE_PRECAUTIONS.values() if v.get('disease_name'))
print(f"\nDiseases with disease_name: {count_with_name}/{len(DISEASE_PRECAUTIONS)}")
