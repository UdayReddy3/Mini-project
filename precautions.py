"""
AI-Powered Disease Precautions and Recommendations Generator
Uses machine learning-based rules to provide disease-specific precautions.
"""

import json
import re
from typing import Dict, List, Tuple

# Comprehensive disease precaution database
DISEASE_PRECAUTIONS = {
    "Tomato___Early_blight": {
        "disease_name": "Early Blight",
        "severity": "High",
        "description": "Early blight is a fungal disease affecting tomato plants, characterized by brown spots with concentric rings on leaves.",
        "symptoms": [
            "Brown spots with concentric rings on lower leaves",
            "Yellow halo around affected areas",
            "Spots gradually spread upward",
            "Stem lesions may develop"
        ],
        "precautions": [
            "🌡️ **Environment Control**: Maintain 65-75°F temperature and 50-60% humidity",
            "💧 **Watering**: Water at soil level, avoid wetting foliage; water in early morning",
            "✂️ **Pruning**: Remove lower leaves (6-8 inches from ground) to improve air circulation",
            "🌿 **Sanitation**: Remove infected leaves immediately and dispose safely",
            "🔄 **Crop Rotation**: Don't plant tomatoes in same soil for 2-3 years",
            "🧴 **Fungicide Application**: Apply copper-based fungicides (Bordeaux mixture) every 7-10 days",
            "🛡️ **Disease-resistant Varieties**: Plant resistant cultivars like 'Defiant' or 'Mountain Magic'"
        ],
        "chemical_treatment": [
            "Chlorothalonil 75% (2g/L) - spray weekly",
            "Mancozeb (0.2-0.3%) - alternate with copper fungicides",
            "Copper oxychloride (1-1.5%) - apply during wet season"
        ],
        "natural_treatment": [
            "Neem oil (3%) spray every 7 days",
            "Sulfur dust application",
            "Trichoderma harzianum bioagent",
            "Bacillus subtilis spores"
        ],
        "time_to_recovery": "2-3 weeks with proper treatment",
        "yield_impact": "Can reduce yield by 50% if untreated",
        "cost_effectiveness": "High - early intervention saves crops"
    },
    
    "Tomato___Late_blight": {
        "disease_name": "Late Blight",
        "severity": "Critical",
        "description": "Late blight is a highly destructive oomycete disease causing rapid leaf and fruit rot in tomatoes.",
        "symptoms": [
            "Water-soaked spots on leaves and stems",
            "White mold on leaf undersides",
            "Rapid spread during wet conditions",
            "Fruit develops brown, oily lesions",
            "Plant collapse in severe cases"
        ],
        "precautions": [
            "🚨 **Immediate Action**: Remove infected plants immediately to prevent spread",
            "💧 **Reduce Humidity**: Improve ventilation and reduce watering frequency",
            "🌡️ **Temperature**: Maintain above 20°C; disease thrives in cool, wet conditions",
            "✂️ **Defoliation**: Remove lower infected leaves to reduce inoculum",
            "🔄 **Crop Rotation**: 3-year minimum rotation cycle",
            "🌾 **Resistant Varieties**: Use resistant cultivars when available",
            "⏰ **Early Warning**: Monitor weather for cool, wet conditions (high risk period)"
        ],
        "chemical_treatment": [
            "Mancozeb 75% WP (2-2.5 kg/ha) - spray 5-7 day intervals",
            "Cymoxanil + Mancozeb - apply preventively",
            "Metalaxyl (systemic) for early intervention",
            "Copper oxychloride - adjunct to other fungicides"
        ],
        "natural_treatment": [
            "Bacillus subtilis bioagent",
            "Trichoderma harzianum",
            "Pseudomonas species",
            "Potassium bicarbonate spray"
        ],
        "time_to_recovery": "4-6 weeks; may require complete crop loss",
        "yield_impact": "Can destroy entire crop if unchecked",
        "cost_effectiveness": "Very High - preventive measures save entire harvest"
    },
    
    "Tomato___Septoria_leaf_spot": {
        "disease_name": "Septoria Leaf Spot",
        "severity": "Medium-High",
        "description": "Septoria leaf spot is a fungal disease causing circular lesions with dark margins and gray centers on tomato leaves.",
        "symptoms": [
            "Small circular spots with dark brown margins",
            "Gray center with dark concentric rings",
            "Tiny black fruiting bodies in center",
            "Leaves turn yellow and drop",
            "Spots coalesce causing large dead areas"
        ],
        "precautions": [
            "💧 **Irrigation**: Water at base only, avoid overhead watering",
            "🌬️ **Air Flow**: Ensure good ventilation between plants",
            "✂️ **Pruning**: Remove lower leaves and thin canopy",
            "🌿 **Sanitation**: Dispose of infected plant material; don't compost",
            "🔄 **Rotation**: 3-year crop rotation minimum",
            "📍 **Spacing**: Space plants 60-90cm apart for airflow",
            "🌧️ **Weather**: Avoid planting during rainy season onset"
        ],
        "chemical_treatment": [
            "Copper fungicides - apply at 7-10 day intervals",
            "Chlorothalonil - preventive spray program",
            "Mancozeb - rotate with other fungicides",
            "Azoxystrobin (strobilurin class) - modern alternative"
        ],
        "natural_treatment": [
            "Bacillus subtilis application",
            "Neem oil spray (3%)",
            "Sulfur dust (avoid in high temperatures)",
            "Trichoderma harzianum"
        ],
        "time_to_recovery": "2-4 weeks with treatment",
        "yield_impact": "30-50% yield reduction if untreated",
        "cost_effectiveness": "High - early prevention very effective"
    },
    
    "Tomato___Spider_mites_Two_spotted_spider_mite": {
        "disease_name": "Two-Spotted Spider Mites",
        "severity": "Medium",
        "description": "Spider mites are tiny arachnids that pierce plant cells and extract sap, causing stippling and webbing.",
        "symptoms": [
            "Fine stippling (tiny dots) on leaves",
            "Yellow/brown discoloration",
            "Fine webbing on leaf undersides",
            "Leaves curl and drop",
            "Overall plant decline",
            "Mites visible under magnification"
        ],
        "precautions": [
            "💧 **Humidity**: Maintain 60-70% humidity; spider mites prefer dry conditions",
            "🌡️ **Temperature**: Keep below 25°C when possible",
            "🌍 **Sanitation**: Remove weeds and debris where mites hide",
            "🔍 **Monitoring**: Check leaf undersides weekly for early detection",
            "✂️ **Pruning**: Remove heavily infested leaves",
            "🚫 **Pesticides**: Avoid broad-spectrum insecticides that kill natural predators",
            "🐞 **Beneficial Insects**: Introduce predatory mites (Phytoseiulus) or ladybugs"
        ],
        "chemical_treatment": [
            "Sulfur dust/spray - apply weekly",
            "Acaricide (abamectin) - use for heavy infestations",
            "Miticides (dicofol) - alternate with sulfur",
            "Spinosad - organic acaricide option"
        ],
        "natural_treatment": [
            "Neem oil spray (3-5%) every 5-7 days",
            "Insecticidal soap spray",
            "Predatory mites (Phytoseiulus persimilis)",
            "Ladybugs and lacewings"
        ],
        "time_to_recovery": "1-2 weeks with proper treatment",
        "yield_impact": "20-40% reduction if severe",
        "cost_effectiveness": "High - organic methods very effective"
    },
    
    "Tomato___Target_Spot": {
        "disease_name": "Target Spot",
        "severity": "High",
        "description": "Target spot is a fungal disease causing circular lesions with concentric rings resembling a target.",
        "symptoms": [
            "Circular spots with concentric rings",
            "Brown lesions with yellow halo",
            "Spots coalesce into large dead areas",
            "Affects leaves, stems, and fruit",
            "Spore production in wet conditions"
        ],
        "precautions": [
            "🌡️ **Temperature**: Maintain 20-25°C; disease grows at 20-27°C",
            "💧 **Moisture**: Reduce leaf wetness duration below 12 hours",
            "🌬️ **Ventilation**: Use fans to reduce humidity; space plants wider",
            "✂️ **Pruning**: Remove lower leaves (8-10 inches) to reduce splash",
            "🌿 **Sanitation**: Remove infected leaves; clean greenhouse regularly",
            "🔄 **Crop Rotation**: Avoid planting tomatoes consecutively",
            "⏰ **Early Treatment**: Apply fungicides before disease establishes"
        ],
        "chemical_treatment": [
            "Mancozeb 75% WP (2.5 kg/ha)",
            "Chlorothalonil - spray every 7-10 days",
            "Azoxystrobin - systemic option",
            "Copper fungicides - adjunct treatment"
        ],
        "natural_treatment": [
            "Bacillus subtilis bioagent",
            "Trichoderma harzianum",
            "Neem oil (3%) spray",
            "Sulfur dust application"
        ],
        "time_to_recovery": "3-4 weeks with intensive treatment",
        "yield_impact": "Can destroy crop if untreated",
        "cost_effectiveness": "Very High - prevention critical"
    },
    
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "disease_name": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "severity": "Critical",
        "description": "TYLCV is a viral disease transmitted by whiteflies causing severe leaf curling and plant stunting.",
        "symptoms": [
            "Yellowing and upward curling of leaves",
            "Severe plant stunting and wilting",
            "Reduced flowering and fruit set",
            "Dark veins on leaf undersides",
            "Affected flowers drop without fruiting"
        ],
        "precautions": [
            "🦟 **Whitefly Control**: Use yellow sticky traps to monitor and control vectors",
            "🌿 **Resistant Varieties**: Plant TYLCV-resistant cultivars (essential)",
            "🚫 **Infected Plants**: Remove infected plants immediately (no cure)",
            "🛡️ **Isolation**: Maintain 10m distance from infected fields",
            "🌾 **Weed Control**: Remove host weeds where whiteflies breed",
            "📍 **Spacing**: Better spacing reduces whitefly congregation",
            "🛂 **Quarantine**: New plants must be virus-free certified"
        ],
        "chemical_treatment": [
            "No cure available once infected",
            "Whitefly control: Imidacloprid 17.8% SL",
            "Neem oil for whitefly management",
            "Reflective mulches to deter whiteflies"
        ],
        "natural_treatment": [
            "Yellow sticky traps for monitoring/control",
            "Reflective aluminum mulch (repels whiteflies)",
            "Neem-based insecticides for vectors",
            "Resistant varieties are primary defense"
        ],
        "time_to_recovery": "No recovery; prevention only",
        "yield_impact": "90-100% crop loss if severe",
        "cost_effectiveness": "Critical - prevention is only option"
    },
    
    "Tomato___Bacterial_spot": {
        "disease_name": "Bacterial Spot",
        "severity": "High",
        "description": "Bacterial spot is caused by Xanthomonas species, causing necrotic lesions on leaves and fruit.",
        "symptoms": [
            "Small water-soaked spots on leaves",
            "Spots expand with yellow halo",
            "Dark lesions on fruit (unmarketable)",
            "Leaves turn yellow and drop",
            "Plant defoliation in severe cases"
        ],
        "precautions": [
            "🌧️ **Overhead Watering**: Avoid; use drip irrigation",
            "✂️ **Pruning**: Remove infected leaves immediately",
            "🌿 **Sanitation**: Sterilize tools between cuts; clean greenhouse",
            "🔄 **Rotation**: 2-3 year minimum rotation",
            "💧 **Irrigation Hygiene**: Use clean water sources",
            "🛡️ **Copper Sprays**: Apply preventively in humid seasons",
            "🔍 **Early Detection**: Scout plants regularly for symptoms"
        ],
        "chemical_treatment": [
            "Copper oxychloride (1-2%) - spray every 7-10 days",
            "Copper sulfate (Bordeaux mixture) - preventive",
            "Streptomycin (where available) - early infection",
            "Fixed copper fungicides"
        ],
        "natural_treatment": [
            "Copper-based organic fungicides",
            "Bacillus subtilis bioagent",
            "Neem oil spray",
            "Sulfur dust (non-copper alternative)"
        ],
        "time_to_recovery": "No cure; disease progresses; prevent spread",
        "yield_impact": "40-70% yield loss if severe",
        "cost_effectiveness": "Very High - prevention critical"
    },
    
    "Pepper___bell___Bacterial_spot": {
        "disease_name": "Bacterial Spot (Pepper)",
        "severity": "High",
        "description": "Bacterial spot affects peppers similarly to tomatoes, causing necrotic lesions.",
        "symptoms": [
            "Small dark lesions with yellow halo",
            "Water-soaked appearance",
            "Lesions coalesce into large dead areas",
            "Fruit becomes unmarketable",
            "Leaf drop in severe infections"
        ],
        "precautions": [
            "💧 **Drip Irrigation**: Avoid overhead watering",
            "🌿 **Sanitation**: Remove infected plant parts; sterilize tools",
            "🔄 **Crop Rotation**: 3-year minimum rotation",
            "🛡️ **Copper Sprays**: Apply preventively, especially in wet seasons",
            "🔍 **Monitoring**: Weekly scouting for early detection",
            "🌬️ **Air Flow**: Improve ventilation to reduce leaf wetness"
        ],
        "chemical_treatment": [
            "Copper oxychloride (1-1.5%)",
            "Bordeaux mixture (1%) - preventive",
            "Fixed copper fungicides",
            "Streptomycin sulfate (where available)"
        ],
        "natural_treatment": [
            "Copper-based organic fungicides",
            "Bacillus subtilis",
            "Neem oil spray",
            "Sulfur alternatives"
        ],
        "time_to_recovery": "No cure; prevent and contain",
        "yield_impact": "50-80% marketability loss",
        "cost_effectiveness": "Very High - prevention essential"
    },
    
    "Potato___Early_blight": {
        "disease_name": "Early Blight (Potato)",
        "severity": "High",
        "description": "Early blight on potatoes causes brown spots with concentric rings on leaves and tubers.",
        "symptoms": [
            "Brown spots with concentric rings on lower leaves",
            "Yellow halo around lesions",
            "Spots expand and coalesce",
            "Brown lesions on tuber surface",
            "Disease progresses upward on plant"
        ],
        "precautions": [
            "✂️ **Pruning**: Remove lower leaves (8-10 inches from ground)",
            "💧 **Irrigation**: Water at soil level; avoid overhead watering",
            "🌬️ **Air Circulation**: Increase spacing and remove weeds",
            "🔄 **Rotation**: 3-year minimum rotation",
            "🌿 **Sanitation**: Remove infected leaves and tubers",
            "🛡️ **Fungicide Program**: Start preventively 3-4 weeks after emergence",
            "🧬 **Resistant Varieties**: Use resistant potato cultivars"
        ],
        "chemical_treatment": [
            "Mancozeb 75% WP (2-2.5 kg/ha) - weekly applications",
            "Chlorothalonil - alternate with mancozeb",
            "Copper fungicides - adjunct treatment",
            "Azoxystrobin - modern alternative"
        ],
        "natural_treatment": [
            "Bacillus subtilis bioagent",
            "Trichoderma harzianum",
            "Neem oil spray (3%)",
            "Sulfur dust application"
        ],
        "time_to_recovery": "2-3 weeks with proper management",
        "yield_impact": "30-50% tuber yield loss if untreated",
        "cost_effectiveness": "High - prevention saves significant yield"
    },
    
    "Potato___Late_blight": {
        "disease_name": "Late Blight (Potato)",
        "severity": "Critical",
        "description": "Late blight is the most destructive potato disease, causing rapid leaf and tuber rot.",
        "symptoms": [
            "Water-soaked lesions on leaves and stems",
            "White mold on leaf undersides",
            "Rapid plant collapse in wet weather",
            "Tuber rot (dry rot) in storage",
            "Brown lesions with depressed centers"
        ],
        "precautions": [
            "🚨 **Immediate Action**: Remove infected plants from field",
            "💧 **Reduce Moisture**: Improve drainage; reduce irrigation",
            "🌡️ **Crop Timing**: Avoid growing during high-risk seasons",
            "🔄 **Rotation**: Rigorous 4-year rotation mandatory",
            "🌾 **Resistant Varieties**: Plant resistant cultivars (critical)",
            "⏰ **Spray Schedule**: Implement mandatory fungicide program",
            "🗑️ **Tuber Sanitation**: Use disease-free seed potatoes"
        ],
        "chemical_treatment": [
            "Mancozeb 75% WP (2.5-3 kg/ha) - 5-7 day intervals",
            "Cymoxanil + Mancozeb - systemic-contact combination",
            "Metalaxyl (preventive for early intervention)",
            "Fluazinam - excellent late blight control"
        ],
        "natural_treatment": [
            "Bacillus subtilis bioagent",
            "Trichoderma species",
            "Pseudomonas bioagents",
            "Potassium bicarbonate spray"
        ],
        "time_to_recovery": "No recovery; disease destruction occurs rapidly",
        "yield_impact": "100% crop destruction possible",
        "cost_effectiveness": "Critical - mandatory preventive fungicide program"
    },
    
    "Corn___Cercospora_leaf_spot_Gray_leaf_spot": {
        "disease_name": "Gray Leaf Spot (Corn)",
        "severity": "High",
        "description": "Gray leaf spot is a fungal disease of corn causing rectangular lesions with gray centers.",
        "symptoms": [
            "Rectangular lesions with gray center and dark margin",
            "Lesions aligned with leaf veins",
            "Pycnidia (fruiting bodies) visible in lesions",
            "Leaf death and shredding",
            "Disease progresses from lower to upper leaves"
        ],
        "precautions": [
            "🔄 **Crop Rotation**: Mandatory 2-3 year rotation from corn",
            "🌾 **Resistant Hybrids**: Plant tolerant hybrids",
            "🗑️ **Residue Management**: Plow under infected crop residue",
            "💨 **Air Flow**: Improve field ventilation; avoid low-lying areas",
            "🌿 **Sanitation**: Remove infected plant parts during season",
            "⏰ **Spray Timing**: Apply fungicides at VT (tasseling) and R1 stages",
            "📍 **Planting Density**: Don't overcrowd; maintain recommended spacing"
        ],
        "chemical_treatment": [
            "Propiconazole - apply at VT stage (timing critical)",
            "Strobilurin fungicides (azoxystrobin) - effective option",
            "Tebuconazole - triazole fungicide",
            "Tank mix combinations for better control"
        ],
        "natural_treatment": [
            "Bacillus subtilis bioagent",
            "Trichoderma harzianum",
            "Sulfur dust (if applicable)",
            "Neem oil (preventive use)"
        ],
        "time_to_recovery": "3-4 weeks; focus on preventing grain fill stage infection",
        "yield_impact": "20-50% yield loss if grain fill stage affected",
        "cost_effectiveness": "High - single fungicide application cost-effective"
    }
}


class DiseasePrecautionGenerator:
    """Generate AI-powered disease precautions and recommendations."""
    
    def __init__(self):
        self.disease_db = DISEASE_PRECAUTIONS
    
    def get_precautions(self, disease_name: str) -> Dict:
        """Get comprehensive precautions for a detected disease."""
        # Normalize disease name and keys for robust matching
        def _normalize(s: str) -> str:
            s = s.lower()
            # replace non-alphanumeric with underscore
            s = re.sub(r'[^a-z0-9]+', '_', s)
            # collapse multiple underscores
            s = re.sub(r'_+', '_', s).strip('_')
            return s

        name_norm = _normalize(disease_name)
        # exact or partial match on normalized keys
        for key in self.disease_db:
            key_norm = _normalize(key)
            if key_norm == name_norm or name_norm in key_norm or key_norm in name_norm:
                # ensure the returned dict contains expected keys with defaults
                entry = dict(self.disease_db[key])
                defaults = {
                    'chemical_treatment': [],
                    'natural_treatment': [],
                    'time_to_recovery': 'N/A',
                    'yield_impact': 'Unknown',
                    'cost_effectiveness': 'N/A',
                    'fertilizer_recommendation': None,
                    'symptoms': [],
                    'precautions': []
                }
                for k, v in defaults.items():
                    if k not in entry or entry.get(k) is None:
                        entry[k] = v
                return entry
        
        # Return generic precautions if disease not found
        return self._get_generic_precautions(disease_name)
    
    def _get_generic_precautions(self, disease_name: str) -> Dict:
        """Return generic precautions for unknown diseases."""
        return {
            "disease_name": "Unknown",
            "severity": "Unknown",
            "description": "N/A",
            "symptoms": [],
            "precautions": [
                "Manageable with proper care",
                "🌿 **Sanitation**: Remove affected plant parts",
                "💧 **Irrigation**: Water at base, avoid foliage"
            ],
            "fertilizer_recommendation": "Balanced NPK following soil test (e.g., DAP as basal; Urea for split N top-dress; MOP for K if needed).",
            "chemical_treatment": [],
            "natural_treatment": [],
            "time_to_recovery": "N/A",
            "yield_impact": "Unknown",
            "cost_effectiveness": "N/A"
        }
    def get_smart_precautions(self, disease_name: str) -> Dict:
        """
        Get precautions with AI-powered fallback for missing data.
        If treatment data is missing, generate smart recommendations based on disease type.
        """
        precautions = self.get_precautions(disease_name)

        # Check if we have real treatment data or just generic fallbacks
        has_chemical = precautions.get('chemical_treatment') and len(precautions.get('chemical_treatment', [])) > 0
        has_natural = precautions.get('natural_treatment') and len(precautions.get('natural_treatment', [])) > 0

        # If missing treatments, generate smart recommendations
        if not has_chemical or not has_natural:
            smart_prec = self._generate_smart_treatments(disease_name, precautions)
            if smart_prec:
                # Merge smart recommendations with existing data
                for key, value in smart_prec.items():
                    if key not in precautions or not precautions[key]:
                        precautions[key] = value

        return precautions

    def _generate_smart_treatments(self, disease_name: str, precautions: Dict) -> Dict:
        """
        Generate smart treatment recommendations based on disease characteristics.
        Uses pattern matching and agricultural best practices.
        """
        disease_name_lower = disease_name.lower()
        disease_desc = precautions.get('description', '').lower()
        disease_symptoms = ' '.join([str(s).lower() for s in precautions.get('symptoms', [])])

        combined_text = f"{disease_name_lower} {disease_desc} {disease_symptoms}"

        # Detect disease type
        is_fungal = any(word in combined_text for word in ['fungal', 'mold', 'mildew', 'blight', 'rot', 'spot', 'rust', 'scab', 'powdery'])
        is_bacterial = any(word in combined_text for word in ['bacterial', 'blight', 'wilt', 'canker', 'leaf streak', 'spot'])
        is_viral = any(word in combined_text for word in ['virus', 'viral', 'mosaic', 'curl', 'leaf curl', 'yellow'])
        is_pest = any(word in combined_text for word in ['mite', 'bug', 'beetle', 'worm', 'caterpillar', 'aphid', 'thrips'])
        is_deficiency = any(word in combined_text for word in ['deficiency', 'chlorosis', 'necrosis', 'nutrient'])

        smart_treatments = {}

        # Generate chemical treatments based on disease type
        if not precautions.get('chemical_treatment'):
            if is_fungal:
                smart_treatments['chemical_treatment'] = [
                    "Fungicide spray: Mancozeb (2-2.5 g/L) or Carbendazim (1 g/L)",
                    "Apply every 7-14 days based on severity and weather",
                    "Alternate fungicides to prevent resistance",
                    "Start preventive sprays before disease onset during monsoon"
                ]
            elif is_bacterial:
                smart_treatments['chemical_treatment'] = [
                    "Copper-based bactericide: Copper oxychloride (3 g/L)",
                    "Streptomycin (300 ppm) where permitted",
                    "Apply weekly or every 10 days",
                    "Combine with good sanitation practices"
                ]
            elif is_viral:
                smart_treatments['chemical_treatment'] = [
                    "No direct chemical cure for viral diseases",
                    "Control vector insects (whiteflies, aphids) using recommended insecticides",
                    "Remove infected plants to prevent spread",
                    "Consult local agricultural department"
                ]
            elif is_pest:
                smart_treatments['chemical_treatment'] = [
                    "Recommended insecticide: Spinosad or Neem-based products",
                    "Apply according to manufacturer instructions",
                    "Rotate different classes of insecticides",
                    "Biological controls (predatory mites, ladybugs) recommended first"
                ]
            elif is_deficiency:
                smart_treatments['chemical_treatment'] = [
                    "Balanced fertilizer application: NPK (10-10-10)",
                    "Micronutrients: Iron sulfate, Zinc sulfate as needed",
                    "Foliar spray with micro-nutrients",
                    "Soil testing recommended for specific deficiencies"
                ]
        
            # Generate natural treatments
            if not precautions.get('natural_treatment'):
                if is_fungal:
                    smart_treatments['natural_treatment'] = [
                        "Neem oil spray (3-5%) every 7-10 days",
                        "Sulfur dust (25 kg/ha) for prevention",
                        "Bacillus subtilis or Trichoderma bioagent",
                        "Improve air circulation and reduce leaf wetness",
                        "Remove affected leaves and sanitize tools"
                    ]
                elif is_bacterial:
                    smart_treatments['natural_treatment'] = [
                        "Bacillus subtilis bioformulation treatment",
                        "Copper-based organic formulations",
                        "Avoid excess nitrogen fertilization",
                        "Remove infected plant parts immediately"
                    ]
                elif is_viral:
                    smart_treatments['natural_treatment'] = [
                        "No cure for viral diseases; focus on prevention",
                        "Use disease-free seeds and resistant varieties",
                        "Control vector insects organically (neem oil)",
                        "Rogue out infected plants immediately"
                    ]
                elif is_pest:
                    smart_treatments['natural_treatment'] = [
                        "Neem oil spray (3-5%) every 5-7 days",
                        "Insecticidal soap spray",
                        "Introduce beneficial insects (ladybugs, lacewings)",
                        "Manual removal of heavily affected leaves",
                        "Water misting to increase humidity (for spider mites)"
                    ]
                elif is_deficiency:
                    smart_treatments['natural_treatment'] = [
                        "Organic compost: 5-10 tons/ha rich in nutrients",
                        "Foliar spray with seaweed extract",
                        "Vermicompost for balanced slow-release nutrients",
                        "Crop rotation with legumes for nitrogen fixation"
                    ]
        
            # Generate recovery time if missing
            if not precautions.get('time_to_recovery'):
                if is_viral:
                    smart_treatments['time_to_recovery'] = "No recovery; remove plant; prevent spread"
                elif is_bacterial:
                    smart_treatments['time_to_recovery'] = "2-4 weeks with treatment; early intervention critical"
                elif is_fungal:
                    smart_treatments['time_to_recovery'] = "3-4 weeks with regular fungicide sprays"
                elif is_pest:
                    smart_treatments['time_to_recovery'] = "1-3 weeks depending on pest population"
                else:
                    smart_treatments['time_to_recovery'] = "1-4 weeks with proper treatment"
        
            # Generate yield impact if missing
            if not precautions.get('yield_impact'):
                smart_treatments['yield_impact'] = "10-50% depending on stage of disease and management"
        
            # Generate cost effectiveness
            if not precautions.get('cost_effectiveness'):
                smart_treatments['cost_effectiveness'] = "High - Early prevention much cheaper than late intervention"
        
            return smart_treatments
    
    def get_all_diseases(self) -> List[str]:
        """Get list of all known diseases."""
        return [self.disease_db[key]["disease_name"] for key in self.disease_db]
    
    def get_severity_level(self, disease_name: str) -> Tuple[str, str]:
        """Get severity level and color coding."""
        precautions = self.get_precautions(disease_name)
        severity = precautions.get("severity", "Unknown")
        
        severity_colors = {
            "Critical": "🔴",
            "High": "🟠",
            "Medium-High": "🟡",
            "Medium": "🟡",
            "Low": "🟢",
            "Unknown": "⚪"
        }
        
        return severity, severity_colors.get(severity, "⚪")
    
    def get_immediate_actions(self, disease_name: str) -> List[str]:
        """Get top 3 immediate actions to take."""
        precautions = self.get_precautions(disease_name)
        severity = precautions.get("severity", "Unknown")
        
        immediate = []
        # Add severity-driven immediate action
        if severity in ["Critical", "High"]:
            immediate.append("🚨 Start treatment immediately - consult expert if available")

        # First 3 precautions are typically most important
        immediate.extend(precautions.get("precautions", [])[:3])

        # Ensure we always return at least a minimal helpful set (do not return empty)
        if not immediate:
            immediate = [
                "Manageable with proper care",
                "🌿 Sanitation: Remove affected plant parts",
                "💧 Irrigation: Water at base, avoid foliage"
            ]

        # Deduplicate while preserving order and limit to 5 items
        seen = set()
        deduped = []
        for act in immediate:
            if act not in seen:
                deduped.append(act)
                seen.add(act)
            if len(deduped) >= 5:
                break

        return deduped


# Load additional precaution definitions from JSON files in `precautions_extra/`
def _load_extra_precautions(extra_dir: str = 'precautions_extra'):
    """Load and merge precautions from JSON files."""
    import os, json
    if not os.path.exists(extra_dir):
        return {}
    
    merged = {}
    
    # Load files in order - more complete files LAST to override partial ones
    # precautions_enriched_manual has only partial data, so load it first
    # precautions_enriched_all has more complete data, load it last to override
    files_to_load = [
        'precautions_complete.json',  # Base coverage with our data
        'precautions_enriched_manual.json',  # Manual enrichment (may be partial)
        'precautions_enriched_all.json',  # Most complete - load last to override!
    ]
    
    for fname in files_to_load:
        path = os.path.join(extra_dir, fname)
        if os.path.isfile(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        # Smart merge: only update if new data is better (has disease_name)
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
                        try:
                            print(f"[PRECAUTIONS] Loaded {len(data)} entries from {fname}")
                        except:
                            print(f"[PRECAUTIONS] Loaded entries from {fname}")
            except Exception as e:
                print(f"Warning: Could not load {fname}: {e}")
    
    # Load any other JSON files not in the priority list
    for fname in os.listdir(extra_dir):
        if not fname.lower().endswith('.json') or fname in files_to_load or fname.startswith('backup'):
            continue
        path = os.path.join(extra_dir, fname)
        if os.path.isdir(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for key, value in data.items():
                        if key not in merged:
                            merged[key] = value
        except Exception:
            pass
    
    return merged


_extras = _load_extra_precautions()

# Update DISEASE_PRECAUTIONS with extras using smart merge
if _extras:
    updated_count = 0
    added_count = 0
    for extra_key, extra_data in _extras.items():
        if extra_key in DISEASE_PRECAUTIONS:
            # Smart update: only override if extra data is better
            existing = DISEASE_PRECAUTIONS[extra_key]
            if extra_data.get('disease_name') and not existing.get('disease_name'):
                # Extra has disease_name, existing doesn't - use extra data
                DISEASE_PRECAUTIONS[extra_key] = extra_data
                updated_count += 1
            elif len(extra_data) > len(existing):
                # Extra data is more complete - merge it
                existing.update(extra_data)
                updated_count += 1
        else:
            # Add new entry
            DISEASE_PRECAUTIONS[extra_key] = extra_data
            added_count += 1
    try:
        print(f"[PRECAUTIONS] Updated {updated_count} existing entries, added {added_count} new entries")
    except:
        print(f"[PRECAUTIONS] Updated {updated_count} entries, added {added_count} entries")

# Recreate generator to pick up merged data
precaution_generator = DiseasePrecautionGenerator()
