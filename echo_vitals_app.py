# ===============================================================
# Echo-Vitals - Streamlit application with Parkinson's Detection
# Integrated with trained model
# ===============================================================

import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from datetime import timedelta
from st_audiorec import st_audiorec
import tempfile
import os
from PIL import Image

# Import your Parkinson's detector from pd_voice_detection
from pd_voice_detection import ParkinsonVoiceDetector

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
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if os.path.exists("logo.png"):
            st.image("logo.png", width=620)
        else:
            st.title("🔊 Echo-Vitals")
            st.markdown("*Advanced Voice Analysis for Health*")
st.markdown("---")

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

# --- HELPER FUNCTIONS ---
def load_audio_from_bytes(wav_bytes):
    bio = io.BytesIO(wav_bytes)
    y, sr = librosa.load(bio, sr=None)
    return y, sr

def save_audio_to_temp(wav_bytes):
    """Save audio bytes to a temporary file and return the path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(wav_bytes)
        return tmp_file.name

def get_audio_info(data, sr):
    """Return dict with basic info about audio file."""
    if data.ndim == 1:
        channels = 1
        frames = data.shape[0]
    else:
        channels = data.shape[1]
        frames = data.shape[0]
    duration_seconds = frames / sr
    return {
        "Duration (s)": round(duration_seconds, 2),
        "Duration (hh:mm:ss)": str(timedelta(seconds=int(duration_seconds))),
        "Sample rate (Hz)": int(sr),
        "Channels": int(channels),
        "Frames (samples)": int(frames),
    }

def plot_spectrogram(data, sr):
    """Generate matplotlib figure with log-amplitude spectrogram."""
    if data.ndim > 1:
        mono = np.mean(data, axis=1)
    else:
        mono = data
    mono = librosa.util.normalize(mono)
    S = np.abs(librosa.stft(mono, n_fft=2048, hop_length=512))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    img = librosa.display.specshow(S_db, sr=sr, hop_length=512,
                                   x_axis="time", y_axis="hz", ax=ax)
    ax.set_title("Spectrogram (dB)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig

def analyze_parkinson_risk(detector, audio_path):
    """Analyze audio for Parkinson's disease risk."""
    try:
        S_db = detector.create_spectrogram(audio_path)
        image = detector.spectrogram_to_image(S_db)
        image = np.expand_dims(image, axis=0)
        probability = detector.model.predict(image, verbose=0)[0][0]

        if probability < 0.3:
            risk_level = "Low"
            risk_color = "green"
            recommendation = "Low risk indicators detected. Continue monitoring if symptoms present."
        elif probability < 0.7:
            risk_level = "Moderate"
            risk_color = "orange"
            recommendation = "Moderate risk indicators. Consider consultation with a healthcare provider."
        else:
            risk_level = "High"
            risk_color = "red"
            recommendation = "High risk indicators detected. Recommend consultation with a neurologist."

        return {
            'probability': float(probability),
            'percentage': probability * 100,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'recommendation': recommendation,
            'success': True
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

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
            st.subheader("Audio Information")
            info = get_audio_info(data, sr)
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Duration**")
                st.write(f"{info['Duration (s)']} s ({info['Duration (hh:mm:ss)']})")
            with col2:
                st.write("**Sample rate**")
                st.write(f"{info['Sample rate (Hz)']} Hz")
                st.write("**Channels**")
                st.write(info["Channels"])

            st.write("---")
            st.subheader("Spectrogram")
            fig = plot_spectrogram(data, sr)
            st.pyplot(fig)

            st.write("---")
            st.subheader("🏥 Parkinson's Disease Risk Analysis")
            if detector is not None:
                st.write("Analyze the recorded audio for potential Parkinson's disease indicators.")
                if st.button("🔍 Analyze for Parkinson's Risk", key="analyze_recorded"):
                    with st.spinner("Analyzing voice patterns..."):
                        results = analyze_parkinson_risk(detector, temp_audio_path)
                        st.session_state.analysis_done = True
                        st.session_state.analysis_results = results

            if st.session_state.analysis_done and st.session_state.analysis_results:
                results = st.session_state.analysis_results
                if results['success']:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(label="Probability", value=f"{results['percentage']:.1f}%")
                    with col2:
                        st.metric(label="Risk Level", value=results['risk_level'])
                    with col3:
                        if results['risk_color'] == 'green':
                            st.success(f"Risk: {results['risk_level']}")
                        elif results['risk_color'] == 'orange':
                            st.warning(f"Risk: {results['risk_level']}")
                        else:
                            st.error(f"Risk: {results['risk_level']}")
                    st.info(f"**Recommendation:** {results['recommendation']}")
                    st.caption("⚠️ **Disclaimer:** This analysis is for screening purposes only and should not replace professional medical diagnosis. Always consult with a healthcare provider for medical concerns.")
                else:
                    st.error(f"Analysis failed: {results['error']}")

            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)

elif option == "Upload .wav file":
    st.sidebar.header("Upload a .wav file")

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

        with st.expander("About Parkinson's Voice Detection"):
            st.markdown("""
            **How it works:**
            - The app uses deep learning to analyze voice patterns
            - Based on research showing voice changes occur early in Parkinson's disease
            - Analyzes features like tremor, hoarseness, and voice stability
            - Achieves ~97% accuracy based on clinical studies
            **Important:**
            - This is a screening tool, not a diagnosis
            - Always consult healthcare professionals for medical concerns
            - Early detection can help with treatment planning
            """)
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

                st.subheader("Audio Information")
                info = get_audio_info(data, sr)
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**File name**")
                    st.write(filename)
                    st.write("**Duration**")
                    st.write(f"{info['Duration (s)']} s ({info['Duration (hh:mm:ss)']})")
                with col2:
                    st.write("**Sample rate**")
                    st.write(f"{info['Sample rate (Hz)']} Hz")
                    st.write("**Channels**")
                    st.write(info["Channels"])

                st.write("---")
                st.subheader("Spectrogram")
                fig = plot_spectrogram(data, sr)
                st.pyplot(fig)

                st.write("---")
                st.subheader("🏥 Parkinson's Disease Risk Analysis")

                if detector is not None:
                    st.write("Analyze the uploaded audio for potential Parkinson's disease indicators.")
                    if st.button("🔍 Analyze for Parkinson's Risk", key="analyze_uploaded"):
                        with st.spinner("Analyzing voice patterns..."):
                            results = analyze_parkinson_risk(detector, temp_audio_path)
                            st.session_state.analysis_done = True
                            st.session_state.analysis_results = results

                if st.session_state.analysis_done and st.session_state.analysis_results:
                    results = st.session_state.analysis_results
                    if results['success']:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(label="Probability", value=f"{results['percentage']:.1f}%")
                        with col2:
                            st.metric(label="Risk Level", value=results['risk_level'])
                        with col3:
                            if results['risk_color'] == 'green':
                                st.success(f"Risk: {results['risk_level']}")
                            elif results['risk_color'] == 'orange':
                                st.warning(f"Risk: {results['risk_level']}")
                            else:
                                st.error(f"Risk: {results['risk_level']}")
                        st.info(f"**Recommendation:** {results['recommendation']}")
                        st.write("Risk Score Visualization:")
                        st.progress(results['probability'])
                        st.caption("⚠️ **Disclaimer:** This analysis is for screening purposes only and should not replace professional medical diagnosis. Always consult with a healthcare provider for medical concerns.")
                    else:
                        st.error(f"Analysis failed: {results['error']}")

            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    Echo-Vitals v1.0 | Voice Analysis for Health | For screening purposes only
</div>
""", unsafe_allow_html=True)
