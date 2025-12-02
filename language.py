"""
Multi-language support for Crop Disease Detection App
Supports English, Hindi, Telugu, and other regional languages
"""

TRANSLATIONS = {
    "en": {
        # Header and Navigation
        "app_title": "Crop Disease Detection",
        "app_subtitle": "AI-Powered Plant Health Analysis",
        
        # Authentication Pages
        "register": "Register",
        "login": "Login",
        "create_account": "Create a New Account",
        "welcome_back": "Welcome Back",
        "join_us": "Join us to start plant health analysis",
        "sign_in": "Sign in to your account",
        
        # Form Fields
        "username": "Username",
        "email": "Email",
        "full_name": "Full Name",
        "password": "Password",
        "confirm_password": "Confirm Password",
        "username_help": "The username you registered with",
        "email_help": "We'll never share your email",
        "full_name_help": "Optional - helps personalize your experience",
        "password_help": "Your account password",
        
        # Buttons
        "btn_register": "Register",
        "btn_login": "Login",
        "btn_go_login": "Login",
        "btn_go_register": "Register",
        "btn_predict": "Predict Disease",
        "btn_predict_all": "Predict All Images",
        "btn_logout": "Logout",
        
        # Messages
        "fill_required": "Please fill in all required fields",
        "password_mismatch": "Passwords do not match",
        "min_password": "Password must be at least 6 characters long",
        "registration_success": "Registration successful!",
        "proceed_login": "Now proceeding to login page...",
        "login_success": "successfully logged in",
        "login_failed": "Please enter both username and password",
        "demo_credentials": "Demo Credentials (for testing)",
        "have_account": "Already have an account?",
        "no_account": "Don't have an account?",
        
        # Main App
        "single_prediction": "Single Image Prediction",
        "batch_analysis": "Batch Analysis",
        "information": "Information",
        "upload_image": "Upload a plant leaf image",
        "upload_multiple": "Upload multiple plant images",
        "uploaded_image": "Uploaded Image",
        "analyzing": "Analyzing image...",
        "processing": "Processing images...",
        "prediction_results": "Prediction Results",
        "predicted_disease": "Predicted Disease",
        "confidence_score": "Confidence Score",
        "prediction_time": "Prediction Time",
        "top_predictions": "Top 3 Predictions",
        "all_predictions": "All Predictions",
        "detailed_predictions": "Detailed Predictions",
        "disease_class": "Disease Class",
        "confidence": "Confidence (%)",
        
        # Precautions Section
        "precautions_treatment": "AI-Powered Precautions & Treatment",
        "severity": "Severity",
        "urgent_action": "URGENT ACTION REQUIRED",
        "immediate_attention": "Immediate attention recommended",
        "manageable": "Manageable with proper care",
        "disease": "Disease",
        "description": "Description",
        "immediate_actions": "Immediate Actions",
        "symptoms": "Symptoms to Look For",
        "prevention": "Prevention & Management Strategies",
        "chemical_treatment": "Chemical Treatment Options",
        "natural_treatment": "Natural/Organic Treatment Options",
        "recovery_time": "Recovery Time",
        "yield_impact": "Yield Impact",
        "cost_effectiveness": "Cost Effectiveness",
        
        # Sidebar
        "about": "About",
        "about_text": "This application uses a deep learning model trained on the PlantVillage dataset to identify plant diseases from images.",
        "features": "Features",
        "model_details": "Model Details",
        "how_to_use": "How to use",
        "step1": "Upload a plant leaf image (JPG, PNG, or BMP)",
        "step2": "Click 'Predict Disease'",
        "step3": "View the results and confidence scores",
        "step4": "Explore predictions for all disease classes",
        
        # Errors
        "error_loading_model": "Error loading model",
        "error_model_files": "Please ensure the model files are in the 'models' directory",
        "error_prediction": "Prediction failed",
        "error_processing": "Failed to process",
        
        # Info Messages
        "upload_to_start": "Upload a plant leaf image and click 'Predict Disease' to get started!",
        "batch_description": "Upload multiple images for batch processing. The system will predict diseases for all images and provide a summary report.",
    },
    
    "hi": {
        # Header and Navigation
        "app_title": "फसल रोग पहचान",
        "app_subtitle": "AI-संचालित पौधा स्वास्थ्य विश्लेषण",
        
        # Authentication Pages
        "register": "पंजीकरण करें",
        "login": "लॉगिन करें",
        "create_account": "नया खाता बनाएं",
        "welcome_back": "स्वागत है",
        "join_us": "पौधा स्वास्थ्य विश्लेषण शुरू करने के लिए हमसे जुड़ें",
        "sign_in": "अपने खाते में साइन इन करें",
        
        # Form Fields
        "username": "उपयोगकर्ता नाम",
        "email": "ईमेल",
        "full_name": "पूरा नाम",
        "password": "पासवर्ड",
        "confirm_password": "पासवर्ड की पुष्टि करें",
        "username_help": "जो उपयोगकर्ता नाम आपने पंजीकृत किया था",
        "email_help": "हम आपकी ईमेल कभी साझा नहीं करेंगे",
        "full_name_help": "वैकल्पिक - आपके अनुभव को व्यक्तिगत बनाने में मदद करता है",
        "password_help": "आपके खाते का पासवर्ड",
        
        # Buttons
        "btn_register": "पंजीकरण",
        "btn_login": "लॉगिन",
        "btn_go_login": "लॉगिन",
        "btn_go_register": "पंजीकरण",
        "btn_predict": "रोग की भविष्यवाणी करें",
        "btn_predict_all": "सभी छवियों की भविष्यवाणी करें",
        "btn_logout": "लॉगआउट",
        
        # Messages
        "fill_required": "कृपया सभी आवश्यक फ़ील्ड भरें",
        "password_mismatch": "पासवर्ड मेल नहीं खाते",
        "min_password": "पासवर्ड कम से कम 6 वर्ण लंबा होना चाहिए",
        "registration_success": "पंजीकरण सफल!",
        "proceed_login": "अब लॉगिन पृष्ठ पर जा रहे हैं...",
        "login_success": "सफलतापूर्वक लॉगिन हुए",
        "login_failed": "कृपया उपयोगकर्ता नाम और पासवर्ड दोनों दर्ज करें",
        "demo_credentials": "डेमो प्रमाण पत्र (परीक्षण के लिए)",
        "have_account": "पहले से खाता है?",
        "no_account": "खाता नहीं है?",
        
        # Main App
        "single_prediction": "एकल छवि भविष्यवाणी",
        "batch_analysis": "बैच विश्लेषण",
        "information": "जानकारी",
        "upload_image": "पौधे की पत्ती की छवि अपलोड करें",
        "upload_multiple": "कई पौधे की छवियां अपलोड करें",
        "uploaded_image": "अपलोड की गई छवि",
        "analyzing": "छवि का विश्लेषण किया जा रहा है...",
        "processing": "छवियों को संसाधित किया जा रहा है...",
        "prediction_results": "भविष्यवाणी परिणाम",
        "predicted_disease": "पूर्वानुमानित रोग",
        "confidence_score": "आत्मविश्वास स्कोर",
        "prediction_time": "भविष्यवाणी समय",
        "top_predictions": "शीर्ष 3 भविष्यवाणियां",
        "all_predictions": "सभी भविष्यवाणियां",
        "detailed_predictions": "विस्तृत भविष्यवाणियां",
        "disease_class": "रोग वर्ग",
        "confidence": "आत्मविश्वास (%)",
        
        # Precautions Section
        "precautions_treatment": "AI-संचालित सावधानियां और उपचार",
        "severity": "गंभीरता",
        "urgent_action": "तत्काल कार्रवाई आवश्यक",
        "immediate_attention": "तत्काल ध्यान की सिफारिश की जाती है",
        "manageable": "उचित देखभाल के साथ प्रबंधनीय",
        "disease": "रोग",
        "description": "विवरण",
        "immediate_actions": "तत्काल कार्रवाई",
        "symptoms": "देखने के लिए लक्षण",
        "prevention": "रोकथाम और प्रबंधन रणनीति",
        "chemical_treatment": "रासायनिक उपचार विकल्प",
        "natural_treatment": "प्राकृतिक/जैविक उपचार विकल्प",
        "recovery_time": "पुनः प्राप्ति समय",
        "yield_impact": "उपज प्रभाव",
        "cost_effectiveness": "लागत प्रभावशीलता",
        
        # Sidebar
        "about": "परिचय",
        "about_text": "यह एप्लिकेशन PlantVillage डेटासेट पर प्रशिक्षित एक गहन शिक्षण मॉडल का उपयोग करके छवियों से पौधे के रोगों की पहचान करता है।",
        "features": "विशेषताएं",
        "model_details": "मॉडल विवरण",
        "how_to_use": "कैसे उपयोग करें",
        "step1": "एक पौधे की पत्ती की छवि अपलोड करें (JPG, PNG, या BMP)",
        "step2": "'रोग की भविष्यवाणी करें' पर क्लिक करें",
        "step3": "परिणाम और आत्मविश्वास स्कोर देखें",
        "step4": "सभी रोग वर्गों के लिए भविष्यवाणियां देखें",
        
        # Errors
        "error_loading_model": "मॉडल लोड करने में त्रुटि",
        "error_model_files": "कृपया सुनिश्चित करें कि मॉडल फ़ाइलें 'models' निर्देशिका में हैं",
        "error_prediction": "भविष्यवाणी विफल",
        "error_processing": "प्रक्रिया विफल",
        
        # Info Messages
        "upload_to_start": "शुरू करने के लिए एक पौधे की पत्ती की छवि अपलोड करें!",
        "batch_description": "बैच प्रोसेसिंग के लिए कई छवियां अपलोड करें।",
    },
    
    "te": {
        # Header and Navigation
        "app_title": "పంట వ్యాధి గుర్తింపు",
        "app_subtitle": "AI-శక్తిచేసిన మొక్క ఆరోగ్య విశ్లేషణ",
        
        # Authentication Pages
        "register": "నమోదు చేయండి",
        "login": "లాగిన్ చేయండి",
        "create_account": "కొత్త ఖాతా సృష్టించండి",
        "welcome_back": "మీరు సాగర్థక చేయబడ్డారు",
        "join_us": "మొక్క ఆరోగ్య విశ్లేషణ ప్రారంభించడానికి మమ్మల్ని సంధానించండి",
        "sign_in": "మీ ఖాతాకు సైన్ ఇన్ చేయండి",
        
        # Form Fields
        "username": "వినియోగదారు పేరు",
        "email": "ఇమెయిల్",
        "full_name": "పూర్తి పేరు",
        "password": "సంకేతపదం",
        "confirm_password": "సంకేతపదాన్ని నిర్ధారించండి",
        "username_help": "మీరు నమోదు చేసిన వినియోగదారు పేరు",
        "email_help": "మేము మీ ఇమెయిల్ను ఎప్పుడూ పంచుకోము",
        "full_name_help": "ఐచ్ఛికమైనది - మీ అనుభవాన్ని వ్యక్తిగతం చేయడానికి సహాయం చేస్తుంది",
        "password_help": "మీ ఖాతా సంకేతపదం",
        
        # Buttons
        "btn_register": "నమోదు",
        "btn_login": "లాగిన్",
        "btn_go_login": "లాగిన్",
        "btn_go_register": "నమోదు",
        "btn_predict": "వ్యాధిని అంచనా వేయండి",
        "btn_predict_all": "అన్ని చిత్రాలను అంచనా వేయండి",
        "btn_logout": "లాగ్ అవుట్",
        
        # Messages
        "fill_required": "దయచేసి అన్ని అవసరమైన ఫీల్డ్‌లను పూరించండి",
        "password_mismatch": "సంకేతపదాలు సరిపోలవు",
        "min_password": "సంకేతపదం కనీసం 6 అక్షరాలు ఉండాలి",
        "registration_success": "నమోదు విజయవంతమైనది!",
        "proceed_login": "ఇప్పుడు లాగిన్ పేజీకి వెళుతుంది...",
        "login_success": "విజయవంతంగా లాగిన్ చేసారు",
        "login_failed": "దయచేసి వినియోగదారు పేరు మరియు సంకేతపదం రెండింటిని నమోదు చేయండి",
        "demo_credentials": "డెమో ఆధారాలు (పరీక్ష కోసం)",
        "have_account": "ఇప్పటికే ఖాతా ఉందా?",
        "no_account": "ఖాతా లేదా?",
        
        # Main App
        "single_prediction": "ఏకైక చిత్ర అంచనా",
        "batch_analysis": "బ్యాచ్ విశ్లేషణ",
        "information": "సమాచారం",
        "upload_image": "మొక్క ఆకు చిత్రాన్ని అప్‌లోడ్ చేయండి",
        "upload_multiple": "బహుళ మొక్క చిత్రాలను అప్‌లోడ్ చేయండి",
        "uploaded_image": "అప్‌లోడ్ చేసిన చిత్రం",
        "analyzing": "చిత్రాన్ని విశ్లేషించుకుంటూ...",
        "processing": "చిత్రాలను ప్రక్రియ చేస్తూ...",
        "prediction_results": "అంచనా ఫలితాలు",
        "predicted_disease": "ఊహించిన వ్యాధి",
        "confidence_score": "విశ్వాస స్కోర్",
        "prediction_time": "అంచనా సమయం",
        "top_predictions": "అగ్ర 3 అంచనాలు",
        "all_predictions": "అన్ని అంచనాలు",
        "detailed_predictions": "వివరణాత్మక అంచనాలు",
        "disease_class": "వ్యాధి తరగతి",
        "confidence": "విశ్వాస (%)",
        
        # Precautions Section
        "precautions_treatment": "AI-శక్తిచేసిన జాగ్రత్తలు మరియు చికిత్స",
        "severity": "తీవ్రత",
        "urgent_action": "తక్షణ చర్య అవసరం",
        "immediate_attention": "తక్షణ శ్రద్ధ సిఫారసు చేయబడింది",
        "manageable": "సరైన సంరక్షణతో నిర్వహించదగినది",
        "disease": "వ్యాధి",
        "description": "వివరణ",
        "immediate_actions": "తక్షణ చర్యలు",
        "symptoms": "చూడటానికి లక్షణాలు",
        "prevention": "నిరోధక మరియు నిర్వహణ వ్యూహాలు",
        "chemical_treatment": "రసాయన చికిత్స ఎంపికలు",
        "natural_treatment": "ప్రకృతిజన్య/సేంద్రీయ చికిత్స ఎంపికలు",
        "recovery_time": "పున: స్థాపన సమయం",
        "yield_impact": "దిగుబడి ప్రభావం",
        "cost_effectiveness": "ధర ప్రభావం",
        
        # Sidebar
        "about": "గురించి",
        "about_text": "ఈ అప్లికేషన PlantVillage డేటాసెట్‌లో శిక్షణ పొందిన లోతైన నేర్పణ మోడల్‌ను ఉపయోగించి చిత్రాల నుండి మొక్క వ్యాధులను గుర్తిస్తుంది.",
        "features": "లక్షణాలు",
        "model_details": "మోడల్ వివరాలు",
        "how_to_use": "ఎలా ఉపయోగించాలి",
        "step1": "మొక్క ఆకు చిత్రాన్ని అప్‌లోడ్ చేయండి (JPG, PNG, లేదా BMP)",
        "step2": "'వ్యాధిని అంచనా వేయండి' పై క్లిక్ చేయండి",
        "step3": "ఫలితాలు మరియు విశ్వాస స్కోర్‌లను చూడండి",
        "step4": "అన్ని వ్యాధి తరగతుల కోసం అంచనాలను అన్వేషించండి",
        
        # Errors
        "error_loading_model": "మోడల్ లోడ్ చేయడంలో లోపం",
        "error_model_files": "దయచేసి మోడల్ ఫైళ్లు 'models' డైరెక్టరీలో ఉన్నాయని నిర్ధారించుకోండి",
        "error_prediction": "అంచనా విఫలమైంది",
        "error_processing": "ప్రక్రియ విఫలమైంది",
        
        # Info Messages
        "upload_to_start": "ప్రారంభించడానికి మొక్క ఆకు చిత్రాన్ని అప్‌లోడ్ చేయండి!",
        "batch_description": "బ్యాచ్ ప్రక్రియకరణ కోసం బహుళ చిత్రాలను అప్‌లోడ్ చేయండి.",
    }
}


def get_translation(language: str, key: str, default: str = "N/A") -> str:
    """
    Get translation for a given key in specified language.
    Falls back to English if language or key not found.
    """
    if language in TRANSLATIONS:
        return TRANSLATIONS[language].get(key, TRANSLATIONS["en"].get(key, default))
    return TRANSLATIONS["en"].get(key, default)


def get_available_languages() -> dict:
    """Get available languages with codes."""
    return {
        "en": "🇬🇧 English",
        "hi": "🇮🇳 हिंदी",
        "te": "🇮🇳 తెలుగు"
    }


class LanguageManager:
    """Manage language selection and translation."""
    
    def __init__(self):
        self.available_languages = get_available_languages()
        self.default_language = "en"
    
    def set_language(self, language_code: str):
        """Set the current language."""
        if language_code in self.available_languages:
            return language_code
        return self.default_language
    
    def translate(self, key: str, language: str = "en") -> str:
        """Translate a key to the specified language."""
        return get_translation(language, key)
    
    def get_all_languages(self) -> dict:
        """Get all available languages."""
        return self.available_languages


# Global language manager instance
language_manager = LanguageManager()
