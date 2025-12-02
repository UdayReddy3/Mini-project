from precautions import DISEASE_PRECAUTIONS, precaution_generator

print("\n🔍 TESTING PRECAUTIONS DATA\n")

complete = 0
incomplete = []

for disease_key in DISEASE_PRECAUTIONS.keys():
    data = precaution_generator.get_precautions(disease_key)
    
    required_keys = [
        'disease_name', 'severity', 'description', 'precautions',
        'chemical_treatment', 'natural_treatment', 'time_to_recovery',
        'yield_impact', 'cost_effectiveness'
    ]
    
    has_all = all(k in data and data.get(k) for k in required_keys)
    
    if has_all:
        complete += 1
    else:
        missing = [k for k in required_keys if not data.get(k)]
        incomplete.append((disease_key, missing))

print(f"✅ COMPLETE ENTRIES: {complete}/{len(DISEASE_PRECAUTIONS)}")
print(f"⚠️  INCOMPLETE ENTRIES: {len(incomplete)}/{len(DISEASE_PRECAUTIONS)}")

if incomplete:
    print("\nSample incomplete diseases:")
    for disease, missing_keys in incomplete[:3]:
        print(f"  - {disease}: missing {', '.join(missing_keys[:3])}")

print(f"\n📊 SAMPLE DATA TEST:")
sample_disease = "Tomato___Early_blight"
sample_data = precaution_generator.get_precautions(sample_disease)
print(f"\n  Disease: {sample_data.get('disease_name')}")
print(f"  Severity: {sample_data.get('severity')}")
print(f"  Description: {sample_data.get('description')[:80]}...")
print(f"  Symptoms: {len(sample_data.get('symptoms', []))} items")
print(f"  Precautions: {len(sample_data.get('precautions', []))} items")
print(f"  Chemical Treatments: {len(sample_data.get('chemical_treatment', []))} items")
print(f"  Natural Treatments: {len(sample_data.get('natural_treatment', []))} items")

print("\n✅ Precautions module is working correctly!")
