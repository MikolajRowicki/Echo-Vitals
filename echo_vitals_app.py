# ===============================================================
# Echo-Vitals - Streamlit application
# Author: Your Name
# Description:
#   A clean, modern audio analysis app for .wav files.
#   Allows upload, playback, basic info display, and spectrogram.
# ===============================================================

import streamlit as st
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
from datetime import timedelta
from st_audiorec import st_audiorec


# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Echo-Vitals", page_icon="🔊", layout="centered")


# --- LOAD EXTERNAL CSS ---
def load_css(file_name: str):
    """Load external CSS file for styling."""
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css("style.css")


# --- HEADER SECTION ---
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("logo.png", width=620)

st.write("---")


# --- HELPER FUNCTIONS ---
def load_audio_from_bytes(wav_bytes):
    """Return (data, samplerate) — data as numpy array, samplerate as int."""
    bio = io.BytesIO(wav_bytes)
    data, sr = sf.read(bio)
    data = np.asarray(data)
    return data, sr


def get_audio_info(data, sr):
    """Return a dict with basic info about the audio file."""
    if data.ndim == 1:
        channels = 1
        frames = data.shape[0]
    else:
        channels = data.shape[1]
        frames = data.shape[0]

    duration_seconds = frames / sr
    info = {
        "Duration (s)": round(duration_seconds, 2),
        "Duration (hh:mm:ss)": str(timedelta(seconds=int(duration_seconds))),
        "Sample rate (Hz)": int(sr),
        "Channels": int(channels),
        "Frames (samples)": int(frames),
    }
    return info


def plot_spectrogram(data, sr):
    """Generate a matplotlib figure with a log-amplitude spectrogram."""
    if data.ndim > 1:
        mono = np.mean(data, axis=1)
    else:
        mono = data

    # Normalize signal (avoid clipping)
    mono = librosa.util.normalize(mono)

    # Short-time Fourier Transform (STFT)
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


# --- MAIN UI ---
st.sidebar.header("Input Audio")

option = st.sidebar.radio(
    "Choose input method:",
    ["Upload .wav file", "Record audio"]
)

if option == "Record audio":
    st.subheader("🎙️ Record your voice")
    st.markdown("Click the **Start Recording** button below, then speak into your microphone.")

    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        # Wyświetl player audio
        #st.audio(wav_audio_data, format='audio/wav')

        # Przetwarzanie dźwięku (tak samo jak przy uploadzie)
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

            # --- SPECTROGRAM ---
            st.write("---")
            st.subheader("Spectrogram")
            fig = plot_spectrogram(data, sr)
            st.pyplot(fig)

else:
    # --- UPLOAD MODE (twój dotychczasowy kod) ---
    st.sidebar.header("Upload a .wav file")
    uploaded = st.sidebar.file_uploader("Select a WAV file", type=["wav"])

    if uploaded is None:
        st.info("Upload a .wav file or switch to 'Record audio' mode.")
        st.markdown("""
            <div class='card'>
                <h4>Supported files</h4>
                <p class='muted'>
                    This app only supports WAV files. Once uploaded, 
                    you can play the audio, view basic information, 
                    and generate a spectrogram.
                </p>
            </div>
        """, unsafe_allow_html=True)

    else:
        filename = uploaded.name
        if not filename.lower().endswith(".wav"):
            st.error("Error: please upload a file with the .wav extension.")
        else:
            file_bytes = uploaded.read()

            try:
                data, sr = load_audio_from_bytes(file_bytes)
            except Exception as e:
                st.error(f"Could not read WAV file: {e}")
            else:
                # --- AUDIO PLAYER ---
                st.subheader("Audio Playback")
                st.audio(file_bytes, format="audio/wav")

                # --- AUDIO INFO ---
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

                # --- SPECTROGRAM ---
                st.write("---")
                st.subheader("Spectrogram")
                fig = plot_spectrogram(data, sr)
                st.pyplot(fig)

                # --- PLACEHOLDER ANALYSIS BUTTON ---
                st.write("---")
                st.subheader("Acoustic Analysis")
                st.write("The button below is a placeholder — in the future, it will trigger acoustic feature analysis.")
                if st.button("Analyze"):
                    st.info("Feature under development — this will run acoustic analysis in future versions.")