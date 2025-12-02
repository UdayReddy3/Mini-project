"""
Streamlit Web Application for Plant Disease Detection
This module provides a web interface for uploading plant images
and getting real-time disease predictions with user authentication.
"""

import streamlit as st
import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from predict import DiseasePredictor
from datetime import datetime
import tempfile
from auth import init_session_state, show_login_page, show_user_profile
from db import init_database
from precautions import precaution_generator
from wallpaper_helper import WALLPAPER_PATH, get_wallpaper_css
from language import get_translation, get_available_languages


# Page configuration
st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database and session
init_database()
init_session_state()

# Initialize language in session state
if 'language' not in st.session_state:
    st.session_state.language = 'en'

# Custom CSS with wallpaper support
wallpaper_css = ""
if WALLPAPER_PATH:
    wallpaper_css = get_wallpaper_css(WALLPAPER_PATH)
    if wallpaper_css:
        print(f"[WALLPAPER] Loaded: {WALLPAPER_PATH}")

# Build CSS - use wallpaper if available, else default gradient
if wallpaper_css:
    final_css = wallpaper_css
else:
    final_css = """
body, .stApp {
    background: linear-gradient(135deg, #87CEEB 0%, #B0E0E6 15%, #FFE5B4 30%, #FFD700 45%, #FFA500 55%, #90EE90 65%, #98D98E 80%, #8BC34A 100%);
    background-attachment: fixed;
    min-height: 100vh;
}
"""

# Inject CSS using safe method
st.markdown("""<style>
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}
""" + final_css + """
/* Crop field texture */
.stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: radial-gradient(circle at 20% 30%, rgba(255, 255, 255, 0.15) 0%, transparent 50%), radial-gradient(circle at 80% 70%, rgba(255, 255, 255, 0.1) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

.main {
    padding: 2rem;
    position: relative;
    z-index: 1;
}

/* Main header styling */
.main-header {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.85) 0%, rgba(245, 245, 220, 0.9) 100%);
    backdrop-filter: blur(15px);
    padding: 4rem 2rem;
    border-radius: 20px;
    text-align: center;
    color: #2d5016;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
    margin-bottom: 2rem;
    border: 2px solid rgba(255, 255, 255, 0.5);
    position: relative;
    overflow: hidden;
}

.main-header h1 {
    font-size: 3.5rem;
    font-weight: 900;
    background: linear-gradient(135deg, #228B22 0%, #FFD700 50%, #FF8C00 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 3px;
}

.main-header h3 {
    font-size: 1.8rem;
    color: #2d5016;
    margin: 1rem 0;
    font-weight: 700;
}

.stButton > button {
    background: linear-gradient(135deg, #228B22 0%, #32CD32 100%) !important;
    color: white !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1a7a1a 0%, #2db82d 100%) !important;
}

.auth-container {
    max-width: 500px;
    margin: 0 auto;
    background: white;
    padding: 3rem 2rem;
    border-radius: 15px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
}

.disease-card {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    border-left: 5px solid #228B22;
}

.metric-card {
    background: linear-gradient(135deg, rgba(34, 139, 34, 0.1) 0%, rgba(50, 205, 50, 0.05) 100%);
    border-radius: 10px;
    padding: 1.5rem;
    text-align: center;
    border: 2px solid rgba(34, 139, 34, 0.3);
}

.metric-value {
    font-size: 2rem;
    color: #228B22;
    font-weight: 900;
}

/* Treatment cards - bright background with dark text */
.treatment-card {
    background: linear-gradient(135deg, #FFF9E6 0%, #FFFEF0 100%) !important;
    border: 2px solid #FFB81C !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    margin: 1rem 0 !important;
    box-shadow: 0 4px 12px rgba(255, 184, 28, 0.2) !important;
}

.treatment-card p, .treatment-card span {
    color: #1a1a1a !important;
    font-weight: 500 !important;
}

/* Impact assessment cards */
.impact-card {
    background: linear-gradient(135deg, #E3F2FD 0%, #F1F8FF 100%) !important;
    border: 2px solid #1976D2 !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    margin: 1rem 0 !important;
    box-shadow: 0 4px 12px rgba(25, 118, 210, 0.2) !important;
}

.impact-card p, .impact-card span {
    color: #0d47a1 !important;
    font-weight: 600 !important;
}

.language-selector {
    position: fixed;
    top: 10px;
    right: 20px;
    z-index: 999;
    background: white;
    padding: 8px 15px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}
</style>
""", unsafe_allow_html=True)

# Language Selector in Top Right
col_lang1, col_lang2 = st.columns([10, 1])
col_lang1, col_lang2 = st.columns([10, 1])
with col_lang2:
    available_languages = get_available_languages()
    selected_lang = st.selectbox(
        "🌐 Language",
        options=list(available_languages.keys()),
        format_func=lambda x: available_languages[x],
        key="language_selector",
        label_visibility="collapsed"
    )
    st.session_state.language = selected_lang

# Helper function to translate
def t(key: str) -> str:
    """Shorthand for translation"""
    return get_translation(st.session_state.language, key, key)


# --------------------- Farmer Mode Helpers ---------------------
def _extract_dosage_amount(text: str):
    """Try to extract a numeric dosage and unit from a treatment string.
    Returns (value_low, value_high, unit) or None if not found."""
    import re
    if not text:
        return None
    # common patterns: '2-2.5 g/L', '2 g/L', '2.5 kg/ha', '300 ppm'
    m = re.search(r"(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*(g/L|kg/ha|ppm|mg/L|%|g/ha)", text, flags=re.I)
    if not m:
        return None
    low = float(m.group(1))
    high = float(m.group(2)) if m.group(2) else None
    unit = m.group(3).lower()
    return (low, high, unit)


def _convert_to_per_ha(low, high, unit):
    """Convert common units to an approximate amount per hectare.
    - For g/L or mg/L or ppm: assume spray volume ~500 L/ha (common farmer assumption).
    - For kg/ha or g/ha: return as-is (convert g->kg where useful).
    Returns a string summary."""
    if unit in ('g/l', 'mg/l', 'ppm'):
        spray_vol = 500.0  # L/ha as a baseline
        if unit == 'mg/l' or unit == 'ppm':
            factor = 0.001  # mg -> g
        else:
            factor = 1.0
        low_gha = low * factor * spray_vol
        if high:
            high_gha = high * factor * spray_vol
            return f"~{low_gha:.0f}–{high_gha:.0f} g/ha (assuming {int(spray_vol)} L/ha spray volume)"
        return f"~{low_gha:.0f} g/ha (assuming {int(spray_vol)} L/ha spray volume)"
    elif unit == 'g/ha':
        if high:
            return f"{low:.0f}–{high:.0f} g/ha"
        return f"{low:.0f} g/ha"
    elif unit == 'kg/ha':
        if high:
            return f"{low:.2f}–{high:.2f} kg/ha"
        return f"{low:.2f} kg/ha"
    elif unit == '%':
        return "Use label % concentration; follow spray instructions per product"
    else:
        return None


def build_farmer_summary(precautions_data, result):
    """Create a simple text summary suitable for printing or downloading by farmers."""
    lines = []
    lines.append(f"Crop Image: {result.get('original_filename', 'uploaded_image')}")
    lines.append(f"Predicted: {result.get('disease_class', 'Unknown').replace('_', ' ')} ({result.get('confidence',0):.1f}% confidence)")
    lines.append("")
    lines.append("Immediate Actions:")
    for i, a in enumerate(precaution_generator.get_immediate_actions(result.get('disease_class')), 1):
        lines.append(f"{i}. {a}")
    lines.append("")
    lines.append("Simple Treatments:")
    chems = precautions_data.get('chemical_treatment', []) or []
    naturals = precautions_data.get('natural_treatment', []) or []
    if chems:
        lines.append("Chemical options:")
        for t in chems:
            lines.append(f"- {t}")
            parsed = _extract_dosage_amount(t)
            if parsed:
                per_ha = _convert_to_per_ha(*parsed)
                if per_ha:
                    lines.append(f"  -> Approx: {per_ha}")
    if naturals:
        lines.append("Natural/Organic options:")
        for t in naturals:
            lines.append(f"- {t}")
    lines.append("")
    lines.append(f"Fertilizer recommendation: {precautions_data.get('fertilizer_recommendation', 'Follow soil test')}")
    lines.append(f"Estimated recovery time: {precautions_data.get('time_to_recovery', 'Varies')}")
    return "\n".join(lines)

# --------------------- End Farmer Mode Helpers ---------------------


@st.cache_resource
def load_predictor():
    """Load the predictor model (cached to avoid reloading)."""
    try:
        # Try to load fine-tuned model first; fallback to original
        model_path = 'models/plant_disease_model_finetuned.h5' if os.path.exists('models/plant_disease_model_finetuned.h5') else 'models/plant_disease_model.h5'
        predictor = DiseasePredictor(
            model_path=model_path,
            class_names_path='models/class_names.json'
        )
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please ensure the model files are in the 'models' directory. "
                "Run 'python model.py' to train the model first.")
        return None


def display_prediction_results(result):
    """Display prediction results in a formatted way with enhanced metrics."""
    confidence = result['confidence']
    original_filename = result.get('original_filename', 'Unknown')
    
    # Color coding based on confidence
    if confidence >= 85:
        confidence_color = "🟢 Excellent"
        confidence_bg = "#d4edda"
        border_color = "#28a745"
    elif confidence >= 70:
        confidence_color = "🟡 Good"
        confidence_bg = "#fff3cd"
        border_color = "#ffc107"
    elif confidence >= 50:
        confidence_color = "🟠 Fair"
        confidence_bg = "#ffe5e5"
        border_color = "#fd7e14"
    else:
        confidence_color = "🔴 Low"
        confidence_bg = "#f8d7da"
        border_color = "#dc3545"
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📁 Image File",
            value=original_filename[:20] + "..." if len(original_filename) > 20 else original_filename,
        )
    
    with col2:
        st.metric(
            label="🎯 Predicted Disease",
            value=result['disease_class'].replace('_', ' '),
            delta="Primary Diagnosis"
        )
    
    with col3:
        st.metric(
            label="📊 Confidence Score",
            value=f"{confidence:.1f}%",
            delta=confidence_color
        )
    
    with col4:
        pred_time = datetime.fromisoformat(result['timestamp'])
        st.metric(
            label="⏱️ Prediction Time",
            value=pred_time.strftime("%H:%M:%S"),
            delta=pred_time.strftime("%b %d, %Y")
        )
    
    # Confidence bar visualization
    st.write("")
    col1, col2 = st.columns([1, 20])
    with col2:
        st.progress(confidence / 100)
        st.caption(f"Model Confidence: {confidence:.2f}% | Reliability: {confidence_color.split()[1]}")


def create_prediction_chart(all_predictions):
    """Create an interactive bar chart of all predictions with better styling."""
    # Get top 15 predictions for better visibility
    top_predictions = dict(list(all_predictions.items())[:15])
    
    df = pd.DataFrame({
        'Disease': [d.replace('_', ' ') for d in top_predictions.keys()],
        'Confidence (%)': list(top_predictions.values())
    })
    
    # Sort by confidence descending
    df = df.sort_values('Confidence (%)', ascending=True)
    
    fig = px.bar(
        df,
        x='Confidence (%)',
        y='Disease',
        orientation='h',
        title='🔬 Top 15 Disease Predictions',
        color='Confidence (%)',
        color_continuous_scale=['#ff6b6b', '#ffd93d', '#6bcf7f'],  # Red -> Yellow -> Green
        template='plotly_white',
        hover_data={'Confidence (%)': ':.2f'},
        labels={'Confidence (%)': 'Confidence Score (%)'},
    )
    
    fig.update_layout(
        height=500,
        xaxis_title='Confidence Score (%)',
        yaxis_title='Disease Classification',
        font=dict(size=12),
        hovermode='closest',
        plot_bgcolor='rgba(240, 242, 246, 0.5)',
        xaxis=dict(range=[0, 100]),
        showlegend=False,
    )
    
    fig.update_traces(
        hovertemplate='<b>%{y}</b><br>Confidence: %{x:.1f}%<extra></extra>',
        marker=dict(line=dict(width=0))
    )
    
    return fig


def main():
    """Main Streamlit application."""
    
    # Check if user is authenticated
    if not st.session_state.authenticated:
        show_login_page()
        return
    
    # User is authenticated - show main app
    show_user_profile()
    
    # Main header with wheat field background style
    st.markdown("""
    <div class='main-header'>
        <h1>🌾 CROP DISEASE DETECTION</h1>
        <h3>AI-Powered Agricultural Intelligence</h3>
        <p>Advanced Deep Learning System for Plant Health Analysis</p>
        <p style='font-size: 0.9rem; margin-top: 1rem;'>Identify diseases instantly • Get treatment recommendations • Protect your harvest</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("About")
        st.markdown("""
        ### Plant Disease Detection
        
        This application uses a deep learning model trained on the **PlantVillage dataset** 
        to identify plant diseases from images.
        
        **Features:**
        - 🖼️ Upload plant leaf images
        - 🤖 Real-time disease prediction
        - 📊 Confidence scores for all diseases
        - 📈 Interactive visualizations
        
        **Model Details:**
        - Architecture: ResNet50 (Transfer Learning)
        - Dataset: PlantVillage (~38 disease classes)
        - Image Size: 224 × 224 pixels
        """)
        
        st.divider()
        st.subheader("🆘 Resources & Help")
        st.markdown("If you need local assistance, contact your agricultural extension office:")
        # read from env if provided, else show placeholder
        ext_name = os.getenv('EXTENSION_OFFICE_NAME', 'Local Agriculture Extension')
        ext_phone = os.getenv('EXTENSION_OFFICE_PHONE', 'Your local helpline number')
        ext_email = os.getenv('EXTENSION_OFFICE_EMAIL', 'extension@example.com')
        st.markdown(f"**Office:** {ext_name}  ")
        st.markdown(f"**Phone:** {ext_phone}  ")
        st.markdown(f"**Email:** {ext_email}  ")
        st.markdown("If urgent, contact local agri-extension or nearest input dealer.")
        st.divider()
        st.markdown("**How to use:**")
        st.markdown("""
        1. Upload a plant leaf image (JPG, PNG, or BMP)
        2. Click 'Predict Disease'
        3. View the results and confidence scores
        4. Explore predictions for all disease classes
        """)
    
    # Load predictor
    predictor = load_predictor()
    if predictor is None:
        st.error("Failed to load the prediction model. Please check the setup.")
        return

    # Prediction display settings (user-configurable)
    confidence_threshold = st.sidebar.slider(
        "Unknown Confidence Threshold (%)",
        min_value=0,
        max_value=100,
        value=60,
        help="If model confidence falls below this percent, show generic 'Unknown' precautions instead of class-specific ones."
    )
    treat_unmapped_as_unknown = st.sidebar.checkbox(
        "Treat unmapped classes as Unknown",
        value=True,
        help="If a predicted class has no matching precautions entry, show the generic Unknown recommendations."
    )
    # Farmer Mode toggle
    farmer_mode = st.sidebar.checkbox(
        "Farmer Mode (Simplified actionable guidance)",
        value=False,
        help="Show plain-language immediate steps, dosage conversions and a printable summary for farmers."
    )
    
    # Main content
    # Local translator to avoid name-shadowing issues with 't'
    translator = globals().get('t', lambda key: get_translation(st.session_state.language, key, key))
    tab1, tab2, tab3 = st.tabs([f"📤 {translator('single_prediction')}", f"📊 {translator('batch_analysis')}", f"ℹ️ {translator('information')}"])
    
    with tab1:
        st.header(translator('single_prediction'))
        # Consent & Disclaimer for farmer safety and data usage
        st.markdown("**Data Use & Safety**")
        st.caption("By uploading images you consent to analysis for plant health diagnostics. No personal data is shared outside this app without permission.")
        consent = st.checkbox("I consent to upload this image for diagnosis and agree to the safety disclaimer", value=False)

        uploaded_file = st.file_uploader(
            translator('upload_image'),
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            accept_multiple_files=False,
            help="Please ensure you have permission to upload the image."
        )

        if not consent:
            st.info("Please check the consent box to upload images and receive recommendations.")
        else:
            if uploaded_file is not None:
                # Display uploaded image with original filename
                image = Image.open(uploaded_file)
                original_filename = uploaded_file.name
                st.image(image, caption=f"📷 Uploaded: {original_filename}", use_container_width=True)
            
            if st.button(f"🔍 {translator('btn_predict')}", use_container_width=True):
                with st.spinner(translator('analyzing')):
                    try:
                        # Save uploaded file temporarily with original extension
                        file_ext = os.path.splitext(original_filename)[1]
                        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                            tmp_file.write(uploaded_file.getbuffer())
                            tmp_path = tmp_file.name
                        
                        # Make prediction
                        result = predictor.predict_disease(tmp_path)
                        
                        # Store original filename in result for display
                        result['original_filename'] = original_filename
                        
                        # Clean up temp file
                        os.remove(tmp_path)
                        
                        st.divider()
                        st.subheader(translator('prediction_results'))
                        
                        # Display main metrics
                        display_prediction_results(result)

                        # Decide whether to show class-specific or generic Unknown precautions
                        detected_disease = result['disease_class']
                        precautions_data = precaution_generator.get_smart_precautions(detected_disease)
                        mapped_is_unknown = precautions_data.get('disease_name', 'Unknown') == 'Unknown'
                        low_confidence = result['confidence'] < confidence_threshold

                        if low_confidence or (treat_unmapped_as_unknown and mapped_is_unknown):
                            st.warning("Low confidence or unknown class — showing generic 'Unknown' recommendations.")
                            # Use the generator's generic payload for Unknown
                            precautions_data = precaution_generator._get_generic_precautions(detected_disease)
                            severity, severity_icon = precaution_generator.get_severity_level(precautions_data.get('disease_name', 'Unknown'))
                        else:
                            severity, severity_icon = precaution_generator.get_severity_level(detected_disease)

                        # Display top 3 predictions
                        st.subheader(translator('top_predictions'))
                        for i, (disease, confidence) in enumerate(result['top_3_predictions'], 1):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.progress(confidence / 100)
                            with col2:
                                st.write(f"**{i}. {disease}**")
                                st.write(f"`{confidence:.2f}%`")
                        
                        st.divider()
                        
                        # Display AI-Powered Precautions and Recommendations
                        st.subheader(f"🛡️ {translator('precautions_treatment')}")
                        
                        # Get precautions for detected disease
                        detected_disease = result['disease_class']
                        precautions_data = precaution_generator.get_smart_precautions(detected_disease)
                        severity, severity_icon = precaution_generator.get_severity_level(detected_disease)
                        
                        # Display severity badge in prominent box
                        col1, col2, col3, col4 = st.columns([1.5, 2, 2, 1.5])
                        with col1:
                            st.markdown(f"### {severity_icon}")
                        with col2:
                            st.markdown(f"#### **{severity}**")
                        with col3:
                            if severity in ["Critical", "High"]:
                                st.error(f"⚠️ {translator('urgent_action')}", icon="🚨")
                            elif severity == "Medium-High":
                                st.warning(f"📌 {translator('immediate_attention')}", icon="⚡")
                            else:
                                st.info(f"ℹ️ {translator('manageable')}", icon="✅")
                        
                        st.markdown("---")
                        
                        # Disease Overview Card
                        st.markdown("### 📋 Disease Overview")
                        col1, col2 = st.columns([2, 3])
                        
                        with col1:
                            st.markdown(f"""
                            **Disease Name:** {precautions_data.get('disease_name', 'Unknown')}
                            
                            **Classification:** {detected_disease.replace('_', ' ')}
                            """)
                        
                        with col2:
                            st.markdown(f"""
                            **Severity Level:** {severity} {severity_icon}
                            
                            **Prediction Confidence:** {result['confidence']:.1f}%
                            """)
                        
                        st.markdown("---")
                        
                        # Description
                        st.markdown("### 📖 Description")
                        description = precautions_data.get('description', 'No description available')
                        st.info(description)
                        
                        st.markdown("---")
                        
                        # Symptoms
                        if precautions_data.get('symptoms'):
                            st.markdown("### 🔍 Symptoms to Look For")
                            symptom_cols = st.columns(2)
                            symptoms = precautions_data.get('symptoms', [])
                            for idx, symptom in enumerate(symptoms):
                                with symptom_cols[idx % 2]:
                                    st.markdown(f"✓ {symptom}")
                            st.markdown("---")
                        
                        # Immediate Actions
                        st.markdown(f"### 🚨 Immediate Actions Required")
                        immediate_actions = precaution_generator.get_immediate_actions(detected_disease)
                        for i, action in enumerate(immediate_actions, 1):
                            st.markdown(f"**{i}.** {action}")
                        
                        st.markdown("---")
                        
                        # Prevention & Management Precautions
                        if precautions_data.get('precautions'):
                            st.markdown("### 🛡️ Prevention & Management Strategies")
                            precautions_list = precautions_data.get('precautions', [])
                            for precaution in precautions_list:
                                st.markdown(precaution)
                            st.markdown("---")
                        
                        # Fertilizer recommendation
                        if precautions_data.get('fertilizer_recommendation'):
                            st.markdown("### 🌾 Fertilizer Recommendations")
                            st.success(f"**Recommended:** {precautions_data.get('fertilizer_recommendation')}")
                            st.markdown("---")
                        
                        # Treatment options - Chemical vs Natural
                        st.markdown("### 💊 Treatment Options")
                        if farmer_mode:
                            # Farmer-friendly simplified display
                            st.markdown("#### 👩‍🌾 Farmer Mode: Simple Steps & Dosage")
                            st.info("Plain-language immediate steps and dosing guidance. Verify product label before application.")

                            # Immediate actionable steps (numbered)
                            st.markdown("**Immediate actions (simple):**")
                            for i, action in enumerate(precaution_generator.get_immediate_actions(detected_disease), 1):
                                st.markdown(f"**{i}.** {action}")

                            st.markdown("**Recommended treatments (simplified):**")
                            chems = precautions_data.get('chemical_treatment', []) or []
                            naturals = precautions_data.get('natural_treatment', []) or []

                            if chems:
                                st.markdown("**Chemical options (showing approximate per-hectare estimate if available):**")
                                for t_idx, ttxt in enumerate(chems, 1):
                                    st.markdown(f"- {ttxt}")
                                    parsed = _extract_dosage_amount(ttxt)
                                    if parsed:
                                        per_ha = _convert_to_per_ha(*parsed)
                                        if per_ha:
                                            st.caption(f"Approx: {per_ha}")
                            else:
                                st.warning("No specific chemical listed — consult local extension before applying pesticides.")

                            if naturals:
                                st.markdown("**Natural/organic options (simple):**")
                                for t_idx, ttxt in enumerate(naturals, 1):
                                    st.markdown(f"- {ttxt}")
                            else:
                                st.info("No natural options listed — focus on sanitation and crop hygiene.")

                            # Printable/downloadable farmer summary (text + HTML A4)
                            summary_text = build_farmer_summary(precautions_data, result)
                            st.download_button(
                                label="📥 Download Farmer Summary (Text)",
                                data=summary_text,
                                file_name=f"farmer_summary_{result.get('disease_class','unknown')}.txt",
                                mime='text/plain'
                            )

                            # Build simple A4 HTML pamphlet
                            html = """<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <title>Farmer Summary</title>
  <style>
    @page { size: A4; margin: 20mm; }
    body { font-family: Arial, sans-serif; color: #111; }
    h1 { color: #2d5016; }
    .section { margin-bottom: 12px; }
    .note { background: #fff3cd; padding: 8px; border-left: 4px solid #ffc107; }
  </style>
</head>
<body>
  <h1>Farmer Summary</h1>
  <div class='section'><strong>Image:</strong> {img}</div>
  <div class='section'><strong>Predicted:</strong> {pred} ({conf}% confidence)</div>
  <div class='section'><strong>Immediate Actions:</strong><ul>""" .format(img=result.get('original_filename'), pred=result.get('disease_class').replace('_',' '), conf=f"{result.get('confidence'):.1f}")

                            for a in precaution_generator.get_immediate_actions(result.get('disease_class')):
                                html += f"<li>{a}</li>"
                            html += """</ul></div>
  <div class='section'><strong>Chemical Treatments:</strong><ul>"""

                            for treatment in precautions_data.get('chemical_treatment', []):
                                html += f"<li>{treatment}</li>"
                            html += """</ul></div>
  <div class='section'><strong>Natural Treatments:</strong><ul>"""

                            for treatment in precautions_data.get('natural_treatment', []):
                                html += f"<li>{treatment}</li>"
                            html += """</ul></div>
  <div class='section note'><strong>Safety:</strong> Always follow product label instructions. Use PPE (gloves, mask) when handling chemicals and avoid re-entry until safe per label.</div>
  <div class='section'><strong>Contact:</strong> {contact}</div>
</body>
</html>"""

                            # Fill contact placeholder
                            html = html.replace('{contact}', f"{ext_name} — {ext_phone} — {ext_email}")

                            st.download_button(
                                label="📄 Download Printable Pamphlet (A4 HTML)",
                                data=html,
                                file_name=f"farmer_pamphlet_{result.get('disease_class','unknown')}.html",
                                mime='text/html'
                            )
                        else:
                            treatment_col1, treatment_col2 = st.columns(2)

                            with treatment_col1:
                                st.markdown(f"#### 🧪 Chemical Treatments")
                                if precautions_data.get('chemical_treatment') and len(precautions_data.get('chemical_treatment', [])) > 0:
                                    for idx, treatment in enumerate(precautions_data.get('chemical_treatment', []), 1):
                                        st.markdown(f"""
                                        <div style="background: linear-gradient(135deg, #FFF9E6 0%, #FFFEF0 100%); border: 2px solid #FFB81C; border-radius: 8px; padding: 12px; margin: 8px 0; color: #1a1a1a; font-weight: 500;">
                                        <strong>{idx}.</strong> {treatment}
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div style="background: #FFE6E6; border: 2px solid #FF6B6B; border-radius: 8px; padding: 12px; margin: 8px 0; color: #8B0000; font-weight: 600;">
                                    ❌ No specific chemical treatment available - Consult local agricultural extension officer
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            with treatment_col2:
                                st.markdown(f"#### 🌿 Natural/Organic Treatments")
                                if precautions_data.get('natural_treatment') and len(precautions_data.get('natural_treatment', [])) > 0:
                                    for idx, treatment in enumerate(precautions_data.get('natural_treatment', []), 1):
                                        st.markdown(f"""
                                        <div style="background: linear-gradient(135deg, #E8F5E9 0%, #F1F8F6 100%); border: 2px solid #4CAF50; border-radius: 8px; padding: 12px; margin: 8px 0; color: #1B5E20; font-weight: 500;">
                                        <strong>{idx}.</strong> {treatment}
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div style="background: #FFE6E6; border: 2px solid #FF6B6B; border-radius: 8px; padding: 12px; margin: 8px 0; color: #8B0000; font-weight: 600;">
                                    ❌ No natural treatments specified - Check fertilizer and management practices
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Recovery and yield impact
                        st.markdown("### 📊 Impact Assessment")
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            time_recovery = precautions_data.get('time_to_recovery', 'Consult expert')
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #E3F2FD 0%, #F1F8FF 100%); border: 2px solid #1976D2; border-radius: 10px; padding: 15px; text-align: center;">
                            <h3 style="color: #0d47a1; margin: 5px 0;">⏱️ Recovery Time</h3>
                            <p style="color: #0d47a1; font-weight: 600; font-size: 18px; margin: 10px 0;">{time_recovery}</p>
                            <small style="color: #1565c0;">Time to full recovery</small>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with metric_col2:
                            yield_impact = precautions_data.get('yield_impact', 'Variable')
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #FFF3E0 0%, #FFF8F0 100%); border: 2px solid #F57C00; border-radius: 10px; padding: 15px; text-align: center;">
                            <h3 style="color: #E65100; margin: 5px 0;">📉 Yield Impact</h3>
                            <p style="color: #E65100; font-weight: 600; font-size: 18px; margin: 10px 0;">{yield_impact}</p>
                            <small style="color: #EF6C00;">Potential crop loss</small>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with metric_col3:
                            cost_eff = precautions_data.get('cost_effectiveness', 'Medium')
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #F3E5F5 0%, #F9F5FF 100%); border: 2px solid #7B1FA2; border-radius: 10px; padding: 15px; text-align: center;">
                            <h3 style="color: #4A148C; margin: 5px 0;">💰 Cost Effectiveness</h3>
                            <p style="color: #4A148C; font-weight: 600; font-size: 18px; margin: 10px 0;">{cost_eff}</p>
                            <small style="color: #6A1B9A;">Treatment ROI</small>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Top 3 Predictions table
                        st.markdown("### 🎯 Top 3 Predictions")
                        top_3_data = {
                            'Rank': ['🥇 1st', '🥈 2nd', '🥉 3rd'],
                            'Disease': [d.replace('_', ' ') for d, _ in result['top_3_predictions']],
                            'Confidence': [f"{c:.2f}%" for _, c in result['top_3_predictions']]
                        }
                        top_3_df = pd.DataFrame(top_3_data)
                        st.table(top_3_df)
                        
                        st.markdown("---")
                        
                        # Prediction chart
                        fig = create_prediction_chart(result['all_predictions'])
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display detailed predictions table
                        st.markdown("### 📑 Complete Predictions for All Classes")
                        predictions_df = pd.DataFrame({
                            'Disease Class': [d.replace('_', ' ') for d in result['all_predictions'].keys()],
                            'Confidence (%)': list(result['all_predictions'].values())
                        }).reset_index(drop=True)
                        predictions_df.index = predictions_df.index + 1
                        predictions_df = predictions_df.sort_values('Confidence (%)', ascending=False)
                        
                        st.dataframe(
                            predictions_df,
                            use_container_width=True,
                            height=400,
                            column_config={
                                'Confidence (%)': st.column_config.ProgressColumn(
                                    'Confidence (%)',
                                    min_value=0,
                                    max_value=100,
                                ),
                            },
                        )
                    
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")

        # If the user has not given consent or hasn't uploaded a file, show guidance
        if not consent or uploaded_file is None:
            st.info("📤 Upload a plant leaf image and click 'Predict Disease' to get started!")
    
    with tab2:
        st.header("Batch Image Analysis")
        
        st.markdown("""
        Upload multiple images for batch processing. The system will predict 
        diseases for all images and provide a summary report.
        """)
        
        uploaded_files = st.file_uploader(
            "Upload multiple plant images",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("🔍 Predict All Images", use_container_width=True):
            with st.spinner("🔄 Processing images..."):
                results_list = []
                progress_bar = st.progress(0)
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                            tmp_file.write(uploaded_file.getbuffer())
                            tmp_path = tmp_file.name
                        
                        # Make prediction
                        result = predictor.predict_disease(tmp_path)
                        result['image_name'] = uploaded_file.name
                        results_list.append(result)
                        
                        # Clean up temp file
                        os.remove(tmp_path)
                    
                    except Exception as e:
                        st.warning(f"Failed to process {uploaded_file.name}: {str(e)}")
                    
                    # Update progress
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                if results_list:
                    # Display results summary
                    st.subheader("📊 Batch Results Summary")
                    
                    summary_data = {
                        'Image': [r['image_name'] for r in results_list],
                        'Predicted Disease': [r['disease_class'] for r in results_list],
                        'Confidence (%)': [r['confidence'] for r in results_list]
                    }
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        avg_confidence = summary_df['Confidence (%)'].mean()
                        st.metric("Average Confidence", f"{avg_confidence:.2f}%")
                    with col2:
                        st.metric("Images Processed", len(results_list))
                    with col3:
                        unique_diseases = summary_df['Predicted Disease'].nunique()
                        st.metric("Unique Diseases Detected", unique_diseases)
                    
                    # Disease distribution
                    st.subheader("🔬 Disease Distribution")
                    disease_counts = summary_df['Predicted Disease'].value_counts()
                    fig = px.pie(
                        values=disease_counts.values,
                        names=disease_counts.index,
                        title='Distribution of Detected Diseases'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("ℹ️ System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 Model Information")
            st.markdown(f"""
            - **Architecture:** ResNet50 (Transfer Learning)
            - **Input Size:** 224 × 224 pixels
            - **Number of Classes:** {len(predictor.class_names)}
            - **Framework:** TensorFlow/Keras
            """)
        
        with col2:
            st.subheader("📚 Dataset Information")
            st.markdown("""
            - **Dataset:** PlantVillage
            - **Total Classes:** ~38 disease categories
            - **Original Images:** Color images
            - **Augmentation:** Rotation, Zoom, Flip, Shift
            """)
        
        st.divider()
        st.subheader("🌱 Supported Disease Classes")
        
        # Display disease classes in columns
        cols = st.columns(3)
        for idx, disease in enumerate(sorted(predictor.class_names)):
            with cols[idx % 3]:
                st.write(f"• {disease}")
        
        st.divider()
        st.subheader("📖 How to Use")
        
        st.markdown("""
        ### Single Image Prediction
        1. Go to the "Single Image Prediction" tab
        2. Upload a plant leaf image (JPG, PNG, BMP, or TIFF)
        3. Click the "Predict Disease" button
        4. View the prediction results with confidence scores
        5. Explore the interactive chart showing all disease predictions
        
        ### Batch Processing
        1. Go to the "Batch Analysis" tab
        2. Upload multiple images at once
        3. Click "Predict All Images"
        4. View summary statistics and disease distribution
        
        ### Tips for Best Results
        - Use high-quality, clear images of affected plant leaves
        - Ensure good lighting conditions
        - Focus on the diseased area of the leaf
        - Avoid images that are too blurry or heavily rotated
        """)


if __name__ == '__main__':
    main()
