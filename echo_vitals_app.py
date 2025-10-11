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
            analyze_parkinson(detector, temp_audio_path, "recorded")
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
                analyze_parkinson(detector, temp_audio_path, "uploaded")
                find_and_display_animal()
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
