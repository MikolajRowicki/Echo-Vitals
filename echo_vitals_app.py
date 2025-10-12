from model_interpretability import ModelInterpreter
import streamlit as st
from st_audiorec import st_audiorec
import os
from PIL import Image
import parselmouth
from parselmouth.praat import call
import math
import random
from pd_voice_detection import ParkinsonVoiceDetector
from utils import *

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Echo-Vitals", page_icon="🔊", layout="centered")

# --- LOAD EXTERNAL CSS ---
def load_css(file_name: str):
    """Load external CSS file for styling."""
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("style.css")

# --- HEADER SECTION ---
st.markdown("---")
display_header()

# --- INITIALIZE MODEL ---
@st.cache_resource
def load_parkinson_model():
    """Load the trained Parkinson's detection model."""
    try:
        detector = ParkinsonVoiceDetector()
        if os.path.exists('parkinson_voice_model.h5'):
            detector.load_model('parkinson_voice_model.h5')
            return detector
        else:
            st.warning("Model file not found. Please ensure 'parkinson_voice_model.h5' is in the app directory.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# --- MAIN UI ---
st.sidebar.header("Input Audio")
option = st.sidebar.radio("Choose input method:", ["Upload .wav file", "Record audio"])
detector = load_parkinson_model()

# Initialize interpreter
if detector is not None:
    interpreter = ModelInterpreter(detector.model)
else:
    interpreter = None

if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

if option == "Record audio":
    st.subheader("🎙️ Record your voice")
    st.markdown("Click the **Start Recording** button below, then speak into your microphone.")
    with st.expander("Recording Instructions"):
        st.markdown("""
        For best Parkinson's detection results:
        1. Record the sustained vowel sound **\"Ahhh\"** for at least 1.5 seconds.
        2. Keep a steady tone and volume.
        3. Record in a quiet environment.
        4. Hold the sound as long as comfortable.
        """)
    wav_audio_data = st_audiorec()
    if wav_audio_data is not None:
        temp_audio_path = save_audio_to_temp(wav_audio_data)
        try:
            data, sr = load_audio_from_bytes(wav_audio_data)
        except Exception as e:
            st.error(f"Could not read recorded audio: {e}")
        else:
            display_audio_info(data, sr)
            extract_and_display_features(temp_audio_path, "recorded")
            analyze_parkinson(detector, temp_audio_path, "recorded", interpreter)
            find_and_display_animal()
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)

elif option == "Upload .wav file":
    uploaded = st.sidebar.file_uploader("Select a WAV file", type=["wav"])
    if uploaded is None:
        st.info("Upload a .wav file or switch to 'Record audio' mode.")
        st.markdown("""
        <div class='card'>
            <h4>Voice Analysis for Health</h4>
            <p class='muted'>
            This app analyzes voice recordings to detect potential indicators of Parkinson's disease.
            Upload a WAV file or record your voice saying "Ahhh" for at least 1.5 seconds.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        filename = uploaded.name
        if not filename.lower().endswith(".wav"):
            st.error("Error: please upload a file with the .wav extension.")
        else:
            file_bytes = uploaded.read()
            temp_audio_path = save_audio_to_temp(file_bytes)
            try:
                data, sr = load_audio_from_bytes(file_bytes)
            except Exception as e:
                st.error(f"Could not read WAV file: {e}")
            else:
                st.subheader("Audio Playback")
                st.audio(file_bytes, format="audio/wav")
                display_audio_info(data, sr, filename)
                extract_and_display_features(temp_audio_path, "uploaded")
                analyze_parkinson(detector, temp_audio_path, "uploaded", interpreter)
                find_and_display_animal()
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)

# --- MODEL INTERPRETABILITY SECTION ---
st.markdown("---")
st.header("🧠 Model Interpretability")

st.markdown("""
<div class='card'>
    <h4>Understanding Model Decisions</h4>
    <p class='muted'>
    Our deep learning model uses Inception V3 CNN architecture with transfer learning 
    to analyze voice spectrograms. Below we explain how the model makes its predictions.
    </p>
</div>
""", unsafe_allow_html=True)

with st.expander("📊 About Model Interpretability Methods"):
    st.markdown("""
    **SHAP (SHapley Additive exPlanations)**
    - Shows which parts of the spectrogram image contribute most to the prediction
    - Based on game theory concepts to fairly distribute prediction contribution
    - Red areas increase Parkinson's probability, blue areas decrease it
    
    **LIME (Local Interpretable Model-agnostic Explanations)**
    - Creates local interpretable approximations around specific predictions
    - Highlights superpixels (image regions) that influence the decision
    - Red regions support the prediction
    
    **Feature Importance from Research**
    Based on the Iyer et al. (2023) study, key acoustic features include:
    - Standard deviation of fundamental frequency (F0)
    - Formant frequency variations (especially F2 and F4)
    - Jitter and shimmer measurements
    - Harmonics-to-noise ratio (HNR)
    - Duration of sustained vowel
    """)

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Model Architecture")
    st.markdown("""
    **Inception V3 with Transfer Learning**
    - Pre-trained on ImageNet dataset
    - 48 layers deep convolutional neural network
    - Custom classification head for Parkinson's detection
    - Analyzes 600×600 pixel spectrograms
    - Input: 1.5s sustained vowel /a/ at 8kHz
    
    **Training Approach:**
    - 70% training, 30% testing split
    - Batch size: 4 samples
    - Epochs: 10 with early stopping
    - Adam optimizer (lr=0.001)
    """)

with col2:
    st.subheader("📈 Model Performance")
    st.markdown("""
    **Reported Metrics (Iyer et al., 2023):**
    - **AUC: 0.97** (colored spectrograms)
    - **AUC: 0.96** (grayscale spectrograms)
    - Outperformed traditional acoustic features (AUC: 0.60-0.73)
    - Tested on 40 PwPD and 41 HC subjects
    - Validated across 100 random train-test splits
    
    **Key Advantages:**
    - Works with telephone-quality recordings (8kHz)
    - No need for controlled recording environment
    - Analyzes time-frequency patterns holistically
    - Less sensitive to individual voice variations
    """)
