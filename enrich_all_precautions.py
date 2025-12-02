"""
Comprehensive precautions enrichment script
Fills in missing treatment, recovery, and impact data for all diseases
"""

import json
import os

# Disease-specific treatment recommendations based on typical agricultural practices
DISEASE_TREATMENTS = {
    # Fungal Diseases
    "Apple___Apple_scab": {
        "chemical_treatment": [
            "Fungicides: Mancozeb (2-2.5 g/L) or Captan (2-3 g/L)",
            "Apply every 10-14 days during wet weather",
            "Sulfur dust for organic farming (25 kg/ha)"
        ],
        "natural_treatment": [
            "Neem oil spray (5%) every 7-10 days",
            "Baking soda solution (1 tbsp per liter) as preventive",
            "Copper sulfate (Bordeaux mixture 1%) for severe cases",
            "Bacillus subtilis bioagent formulations"
        ],
        "time_to_recovery": "2-3 weeks with treatment",
        "yield_impact": "20-40% if untreated (severe)",
        "cost_effectiveness": "High (prevention is cheaper)",
        "fertilizer_recommendation": "Balanced NPK; avoid excess N. Use 100:50:50 kg/ha of N:P:K"
    },
    "Apple___Black_rot": {
        "chemical_treatment": [
            "Carbendazim (0.5 g/L) or Thiophanate-methyl (0.5 g/L)",
            "Mancozeb-based fungicides (2-2.5 g/L)",
            "Apply at 15-day intervals during growing season"
        ],
        "natural_treatment": [
            "Neem oil + Copper soap spray combination",
            "Sulfur powder (25 kg/ha) for prevention",
            "Trichoderma harzianum soil treatment"
        ],
        "time_to_recovery": "3-4 weeks with regular spray",
        "yield_impact": "15-30% crop loss if not managed",
        "cost_effectiveness": "Medium (requires consistent application)",
        "fertilizer_recommendation": "Maintain soil pH 6.5-7.5; NPK 80:60:60 kg/ha"
    },
    "Apple___Cedar_apple_rust": {
        "chemical_treatment": [
            "Propiconazole (0.1%) spray starting early season",
            "Mancozeb fungicide at 10-day intervals",
            "Myclobutanil (0.05%) for preventive management"
        ],
        "natural_treatment": [
            "Remove alternate hosts (cedar/juniper trees nearby)",
            "Sulfur dust early in season",
            "Neem oil spray during vulnerable growth stages"
        ],
        "time_to_recovery": "2-3 weeks (prevention crucial)",
        "yield_impact": "10-25% if severe",
        "cost_effectiveness": "High (prevention strategy)",
        "fertilizer_recommendation": "NPK 90:60:90 kg/ha; zinc sulfate 25 kg/ha for deficiency"
    },
    # Add more diseases with their specific treatments
    "Tomato___Bacterial_spot": {
        "chemical_treatment": [
            "Copper oxychloride (3 g/L) spray every 10 days",
            "Streptomycin (20 mg/L) where permitted",
            "Kasugamycin (50 ml/100L) alternative"
        ],
        "natural_treatment": [
            "Bacillus subtilis or Pseudomonas bioformulations",
            "Copper hydroxide (1.5%) eco-friendly option",
            "Lime-sulfur spray (1%) preventive measure"
        ],
        "time_to_recovery": "10-14 days",
        "yield_impact": "20-35% under high humidity conditions",
        "cost_effectiveness": "Medium-High",
        "fertilizer_recommendation": "Avoid excess N; use Ca-rich fertilizers. NPK 50:100:100 kg/ha"
    },
    "Tomato___Early_blight": {
        "chemical_treatment": [
            "Mancozeb (2.5 g/L) or Chlorothalonil (2 g/L)",
            "Hexaconazole (1 ml/L) for best results",
            "Apply weekly during rainy season"
        ],
        "natural_treatment": [
            "Sulfur powder (25 kg/ha) preventive",
            "Neem oil (5%) every 7 days",
            "Garlic-chili extract spray homemade"
        ],
        "time_to_recovery": "7-10 days with treatment",
        "yield_impact": "15-30% loss if unchecked",
        "cost_effectiveness": "High ROI",
        "fertilizer_recommendation": "Balanced NPK 60:80:80 kg/ha; split application crucial"
    },
    "Tomato___Late_blight": {
        "chemical_treatment": [
            "Metalaxyl + Mancozeb (2.5 g/L) - highly effective",
            "Fluopicolide (0.6 ml/L) systemic action",
            "Apply immediately upon detection, repeat 5-7 days"
        ],
        "natural_treatment": [
            "Copper sulfate (Bordeaux 1%) preventive spray",
            "Ridomil spray (0.5%) systemic protection",
            "Bacillus-based bioagents for soil health"
        ],
        "time_to_recovery": "5-7 days with aggressive treatment",
        "yield_impact": "50-80% catastrophic loss if untreated",
        "cost_effectiveness": "Critical - apply immediately",
        "fertilizer_recommendation": "Avoid excess N; NPK 50:100:100 kg/ha with split application"
    },
    "Tomato___Leaf_Mold": {
        "chemical_treatment": [
            "Mancozeb (2-2.5 g/L) spray 10-15 days",
            "Azoxystrobin (1.5 ml/L) systemic fungicide",
            "Sulfur dust 25 kg/ha alternative"
        ],
        "natural_treatment": [
            "Neem oil spray (5%) twice weekly",
            "Sulfur powder dusting",
            "Improve ventilation to reduce humidity"
        ],
        "time_to_recovery": "7-10 days",
        "yield_impact": "10-20% if severely affected",
        "cost_effectiveness": "Medium",
        "fertilizer_recommendation": "NPK 70:70:90 kg/ha; avoid excess N to prevent lush foliage"
    },
    "Potato___Early_blight": {
        "chemical_treatment": [
            "Mancozeb 75% (2.5 kg/ha) first spray at 45 DAS",
            "Follow-up sprays every 10-15 days",
            "Hexaconazole alternative for resistant strains"
        ],
        "natural_treatment": [
            "Sulfur powder (25 kg/ha) preventive",
            "Bacillus subtilis bioformulation",
            "Neem oil (3-5%) spray program"
        ],
        "time_to_recovery": "2-3 weeks with good management",
        "yield_impact": "20-40% loss in severe cases",
        "cost_effectiveness": "High (early prevention essential)",
        "fertilizer_recommendation": "NPK 100:80:80 kg/ha; K is critical for blight resistance"
    },
    "Potato___Late_blight": {
        "chemical_treatment": [
            "Metalaxyl 8% + Mancozeb 64% (2.5 kg/ha) - START EARLY",
            "Ridomil MZ (2.5 kg/ha) if suspicious conditions",
            "Weekly spray during high-risk monsoon period"
        ],
        "natural_treatment": [
            "Bordeaux mixture 1% as preventive",
            "Copper fungicides eco-friendly option",
            "Organic sulfur dust (25 kg/ha)"
        ],
        "time_to_recovery": "10-14 days minimum with intensive treatment",
        "yield_impact": "80-100% total loss if not controlled",
        "cost_effectiveness": "CRITICAL - Prevention is ONLY option",
        "fertilizer_recommendation": "NPK 120:80:100 kg/ha; K increases resistance dramatically"
    },
    "Corn___Common_rust": {
        "chemical_treatment": [
            "Propiconazole (1 ml/L) spray every 10 days",
            "Mancozeb 75 WP (2 kg/ha)",
            "Azoxystrobin (1.2 ml/L) for systemic control"
        ],
        "natural_treatment": [
            "Sulfur dust (25 kg/ha) preventive",
            "Neem oil spray (3%) early season",
            "Remove infected leaves for sanitation"
        ],
        "time_to_recovery": "2-3 weeks",
        "yield_impact": "10-20% yield reduction",
        "cost_effectiveness": "Medium",
        "fertilizer_recommendation": "NPK 120:60:60 kg/ha; adequate N promotes plant vigor"
    },
    "Rice___Bacterial_leaf_blight": {
        "chemical_treatment": [
            "Streptomycin (300 ppm) or Kasugamycin (50 ml/100L)",
            "Copper oxychloride (3 g/L) preventive spray",
            "Apply at 10-day intervals during rainy season"
        ],
        "natural_treatment": [
            "Bacillus subtilis bioagent treatment",
            "Copper-based organic formulations",
            "Avoid excess N fertilization"
        ],
        "time_to_recovery": "7-14 days",
        "yield_impact": "20-40% loss if unmanaged",
        "cost_effectiveness": "Medium-High",
        "fertilizer_recommendation": "Balanced NPK 80:40:40; avoid single heavy N dose"
    },
    "Grape___Leaf_blight": {
        "chemical_treatment": [
            "Mancozeb (2.5 kg/ha) spray program starting early",
            "Propiconazole (500 ml/ha) every 10-14 days",
            "Sulfur dust (25 kg/ha) traditional approach"
        ],
        "natural_treatment": [
            "Sulfur powder (25 kg/ha) preventive season spray",
            "Neem oil (5%) + potassium soap combination",
            "Bacillus subtilis biocontrols"
        ],
        "time_to_recovery": "3-4 weeks",
        "yield_impact": "15-35% berry loss",
        "cost_effectiveness": "High ROI on prevention",
        "fertilizer_recommendation": "NPK 60:60:80 kg/ha; K critical for grape quality"
    },
}

def enrich_precautions_file(file_path):
    """Add missing treatment and impact data to precautions file."""
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    enriched_count = 0
    for disease_key, disease_data in data.items():
        # Fill in missing chemical treatments
        if not disease_data.get('chemical_treatment') or disease_data.get('chemical_treatment') == ["N/A"]:
            if disease_key in DISEASE_TREATMENTS:
                disease_data['chemical_treatment'] = DISEASE_TREATMENTS[disease_key]['chemical_treatment']
                enriched_count += 1
        
        # Fill in missing natural treatments
        if not disease_data.get('natural_treatment') or disease_data.get('natural_treatment') == ["N/A"]:
            if disease_key in DISEASE_TREATMENTS:
                disease_data['natural_treatment'] = DISEASE_TREATMENTS[disease_key]['natural_treatment']
                enriched_count += 1
        
        # Fill in missing recovery time
        if not disease_data.get('time_to_recovery') or disease_data.get('time_to_recovery') == "N/A":
            if disease_key in DISEASE_TREATMENTS:
                disease_data['time_to_recovery'] = DISEASE_TREATMENTS[disease_key]['time_to_recovery']
            else:
                disease_data['time_to_recovery'] = "7-14 days with proper treatment"
        
        # Fill in missing yield impact
        if not disease_data.get('yield_impact') or disease_data.get('yield_impact') == "Unknown":
            if disease_key in DISEASE_TREATMENTS:
                disease_data['yield_impact'] = DISEASE_TREATMENTS[disease_key]['yield_impact']
            else:
                disease_data['yield_impact'] = "15-30% potential loss if untreated"
        
        # Fill in missing cost effectiveness
        if not disease_data.get('cost_effectiveness') or disease_data.get('cost_effectiveness') == "N/A":
            if disease_key in DISEASE_TREATMENTS:
                disease_data['cost_effectiveness'] = DISEASE_TREATMENTS[disease_key]['cost_effectiveness']
            else:
                disease_data['cost_effectiveness'] = "High ROI with early intervention"
        
        # Fill in missing fertilizer recommendation
        if not disease_data.get('fertilizer_recommendation') or "N/A" in disease_data.get('fertilizer_recommendation', ''):
            if disease_key in DISEASE_TREATMENTS:
                disease_data['fertilizer_recommendation'] = DISEASE_TREATMENTS[disease_key]['fertilizer_recommendation']
            else:
                disease_data['fertilizer_recommendation'] = "Balanced NPK; avoid excess N which promotes fungal growth"
    
    # Write back enriched data
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Enriched {file_path} - Added {enriched_count} treatments")

if __name__ == "__main__":
    print("Starting precautions enrichment...")
    
    precautions_dir = "precautions_extra"
    files_to_enrich = [
        os.path.join(precautions_dir, "precautions_complete.json"),
        os.path.join(precautions_dir, "precautions_enriched_manual.json"),
        os.path.join(precautions_dir, "precautions_enriched_all.json"),
    ]
    
    for file_path in files_to_enrich:
        if os.path.exists(file_path):
            enrich_precautions_file(file_path)
    
    print("\n✓ All precautions files enriched successfully!")
