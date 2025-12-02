"""
AI-Powered Precautions Generator using Google Gemini
Generates intelligent treatment and prevention recommendations for diseases
"""

import google.generativeai as genai
import json
import os
from pathlib import Path

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'YOUR_API_KEY_HERE')
genai.configure(api_key=GEMINI_API_KEY)

def generate_ai_precautions(disease_name, disease_class, description, symptoms):
    """
    Use Gemini AI to generate comprehensive treatment precautions
    for a disease based on its description and symptoms
    """
    
    prompt = f"""
You are an expert agricultural scientist and plant pathologist. Generate comprehensive and practical treatment recommendations for the following plant disease.

Disease Name: {disease_name}
Disease Class: {disease_class}
Description: {description}
Symptoms: {', '.join(symptoms) if isinstance(symptoms, list) else symptoms}

Please provide ONLY a JSON response in this exact format (no additional text):
{{
    "chemical_treatment": [
        "Specific fungicide/bactericide with dosage and frequency",
        "Alternative chemical treatment",
        "Additional preventive spray"
    ],
    "natural_treatment": [
        "Organic/natural remedy with dosage",
        "Alternative natural treatment",
        "Preventive natural practice"
    ],
    "time_to_recovery": "Expected recovery time (e.g., '7-10 days with treatment')",
    "yield_impact": "Potential yield loss percentage or range (e.g., '15-30% if untreated')",
    "cost_effectiveness": "Cost effectiveness assessment (High/Medium/Low with reasoning)",
    "fertilizer_recommendation": "Specific NPK recommendations and ratios",
    "immediate_actions": [
        "First action to take immediately",
        "Second urgent action",
        "Third preventive measure"
    ],
    "precautions": [
        "Long-term prevention strategy",
        "Field management practice",
        "Variety selection advice"
    ]
}}

Make recommendations based on:
1. Common agricultural practices in India
2. Availability of materials (consider Indian market)
3. Cost-effectiveness for small and medium farmers
4. Both preventive and curative approaches
5. Organic and chemical options

Ensure the JSON is valid and complete.
"""
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        # Extract JSON from response
        response_text = response.text.strip()
        
        # Try to find JSON in the response
        if response_text.startswith('{'):
            json_str = response_text
        else:
            # Find JSON block if wrapped in other text
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
            else:
                return None
        
        # Parse and validate JSON
        precautions = json.loads(json_str)
        
        # Ensure all required fields exist
        required_fields = [
            'chemical_treatment', 'natural_treatment', 'time_to_recovery',
            'yield_impact', 'cost_effectiveness', 'fertilizer_recommendation'
        ]
        
        for field in required_fields:
            if field not in precautions:
                precautions[field] = []
        
        print(f"✓ AI Generated precautions for {disease_name}")
        return precautions
        
    except json.JSONDecodeError as e:
        print(f"✗ JSON parsing error for {disease_name}: {e}")
        return None
    except Exception as e:
        print(f"✗ AI generation error for {disease_name}: {e}")
        return None

def fill_missing_with_ai(precautions_file):
    """
    Load precautions file and fill missing treatments using AI
    """
    
    if not os.path.exists(precautions_file):
        print(f"File not found: {precautions_file}")
        return
    
    with open(precautions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    updated_count = 0
    skipped_count = 0
    
    for disease_key, disease_data in data.items():
        # Check if treatments are missing or empty
        has_chemical = disease_data.get('chemical_treatment') and len(disease_data.get('chemical_treatment', [])) > 0
        has_natural = disease_data.get('natural_treatment') and len(disease_data.get('natural_treatment', [])) > 0
        
        if not has_chemical or not has_natural:
            print(f"\n🤖 Generating AI precautions for: {disease_data.get('disease_name', disease_key)}")
            
            # Prepare data for AI
            disease_name = disease_data.get('disease_name', disease_key)
            description = disease_data.get('description', '')
            symptoms = disease_data.get('symptoms', [])
            
            # Generate AI precautions
            ai_precautions = generate_ai_precautions(
                disease_name=disease_name,
                disease_class=disease_key,
                description=description,
                symptoms=symptoms
            )
            
            if ai_precautions:
                # Update only empty fields
                if not has_chemical and ai_precautions.get('chemical_treatment'):
                    disease_data['chemical_treatment'] = ai_precautions['chemical_treatment']
                    print(f"  ✓ Added chemical treatments")
                
                if not has_natural and ai_precautions.get('natural_treatment'):
                    disease_data['natural_treatment'] = ai_precautions['natural_treatment']
                    print(f"  ✓ Added natural treatments")
                
                # Fill other missing fields
                if not disease_data.get('time_to_recovery') or disease_data.get('time_to_recovery') == 'N/A':
                    disease_data['time_to_recovery'] = ai_precautions.get('time_to_recovery', '7-14 days')
                
                if not disease_data.get('yield_impact') or disease_data.get('yield_impact') == 'Unknown':
                    disease_data['yield_impact'] = ai_precautions.get('yield_impact', '15-30%')
                
                if not disease_data.get('cost_effectiveness') or disease_data.get('cost_effectiveness') == 'N/A':
                    disease_data['cost_effectiveness'] = ai_precautions.get('cost_effectiveness', 'Medium')
                
                if not disease_data.get('fertilizer_recommendation') or 'N/A' in disease_data.get('fertilizer_recommendation', ''):
                    disease_data['fertilizer_recommendation'] = ai_precautions.get('fertilizer_recommendation', 'Balanced NPK')
                
                updated_count += 1
            else:
                skipped_count += 1
                print(f"  ✗ Failed to generate AI precautions")
    
    # Save updated data
    with open(precautions_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Updated {updated_count} diseases with AI precautions")
    print(f"⚠ Skipped {skipped_count} diseases (AI generation failed)")
    return updated_count

if __name__ == "__main__":
    if not GEMINI_API_KEY or GEMINI_API_KEY == 'YOUR_API_KEY_HERE':
        print("⚠ GEMINI_API_KEY environment variable not set!")
        print("Please set it first: export GEMINI_API_KEY='your_key'")
    else:
        print("Starting AI Precautions Generation...")
        
        precautions_dir = "precautions_extra"
        files = [
            os.path.join(precautions_dir, "precautions_enriched_manual.json"),
            os.path.join(precautions_dir, "precautions_enriched_all.json"),
        ]
        
        total_updated = 0
        for file_path in files:
            if os.path.exists(file_path):
                print(f"\n{'='*60}")
                print(f"Processing: {file_path}")
                print(f"{'='*60}")
                updated = fill_missing_with_ai(file_path)
                total_updated += updated
        
        print(f"\n{'='*60}")
        print(f"✓ Total diseases updated: {total_updated}")
        print(f"{'='*60}")
