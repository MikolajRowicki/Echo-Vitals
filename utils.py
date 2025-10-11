import tempfile
import io
import math
import random
from datetime import timedelta
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import parselmouth
from parselmouth.praat import call
import streamlit as st
import os

def display_header():
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if os.path.exists("logo.png"):
                st.image("logo.png", width=620)
            else:
                st.title("🔊 Echo-Vitals")
                st.markdown("*Advanced Voice Analysis for Health*")

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

def _safe_round(val, ndigits=3):
    try:
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            return None
        return round(float(val), ndigits)
    except Exception:
        return None

def extract_voice_features_librosa(audio_path):

    try:
        # --- Load audio ---
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        if duration <= 0:
            return {"success": False, "error": "Audio duration is zero or invalid."}

        # --- Fundamental frequency (F0) using PYIN ---
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=75, fmax=500, sr=sr)
        f0 = f0[~np.isnan(f0)]  # usuń NaN
        f0_mean = float(np.mean(f0)) if len(f0) > 0 else None
        f0_std = float(np.std(f0)) if len(f0) > 0 else None

        # --- Energy (RMS) ---
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = float(np.mean(rms)) if len(rms) > 0 else None
        rms_std = float(np.std(rms)) if len(rms) > 0 else None

        # --- Approximate Jitter (frequency variation) ---
        if len(f0) > 2:
            periods = 1 / f0
            jitter_local = np.mean(np.abs(np.diff(periods))) / np.mean(periods)
        else:
            jitter_local = None

        # --- Approximate Shimmer (amplitude variation) ---
        if len(rms) > 2:
            shimmer_local = np.mean(np.abs(np.diff(rms))) / np.mean(rms)
        else:
            shimmer_local = None

        # --- MFCCs (1–9) ---
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=9)
        mfcc_mean = np.mean(mfcc, axis=1) if mfcc is not None else None

        # --- Collect results ---
        results = {
            "Duration (s)": round(duration, 2),
            "F0 mean (Hz)": round(f0_mean, 1) if f0_mean else None,
            "F0 std (Hz)": round(f0_std, 1) if f0_std else None,
            "RMS mean": round(rms_mean, 4) if rms_mean else None,
            "RMS std": round(rms_std, 4) if rms_std else None,
            "Jitter": round(jitter_local * 100, 3) if jitter_local else None,
            "Shimmer": round(shimmer_local * 100, 3) if shimmer_local else None,
        }

        if mfcc_mean is not None:
            for i, coef in enumerate(mfcc_mean, 1):
                results[f"MFCC {i} mean"] = round(coef, 3)

        results["success"] = True
        return results

    except Exception as e:
        return {"success": False, "error": str(e)}


def extract_voice_features(audio_path):
    """
    Extract advanced vocal features: jitter, shimmer, HNR, formants (F1-F3).
    Uses Parselmouth / Praat. Returns dict with results and 'success' flag.
    """
    try:
        snd = parselmouth.Sound(audio_path)
        duration = snd.get_total_duration()
        if duration <= 0:
            return {"success": False, "error": "Audio duration is zero or invalid."}

        pitch = call(snd, "To Pitch", 0.0, 75, 500)
        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)

        try:
            jitter_local = call(point_process, "Get jitter (local)", 0, 0, 75, 500, 1.3)
        except Exception:
            jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)

        try:
            shimmer_local = call([snd, point_process], "Get shimmer (local)", 0, 0, 75, 500, 1.3, 1.6)
        except Exception:
            shimmer_local = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        try:
            harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
        except Exception:
            hnr = None

        formant = call(snd, "To Formant (burg)", 0.0, 5.0, 5500, 0.025, 50)
        f1 = f2 = f3 = None
        try:
            f1 = call(formant, "Get mean", 1, 0, 0, "Hertz")
            f2 = call(formant, "Get mean", 2, 0, 0, "Hertz")
            f3 = call(formant, "Get mean", 3, 0, 0, "Hertz")
        except Exception:
            times = np.linspace(0.01, max(0.01, duration - 0.01), num=10)
            vals_f1, vals_f2, vals_f3 = [], [], []
            for t in times:
                try:
                    v1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
                    v2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
                    v3 = call(formant, "Get value at time", 3, t, "Hertz", "Linear")
                    if v1 is not None and v1 != 0:
                        vals_f1.append(v1)
                    if v2 is not None and v2 != 0:
                        vals_f2.append(v2)
                    if v3 is not None and v3 != 0:
                        vals_f3.append(v3)
                except Exception:
                    continue
            f1 = np.mean(vals_f1) if len(vals_f1) > 0 else None
            f2 = np.mean(vals_f2) if len(vals_f2) > 0 else None
            f3 = np.mean(vals_f3) if len(vals_f3) > 0 else None

        results = {
            "Jitter (local, %)": _safe_round(jitter_local * 100 if jitter_local is not None else None, 3),
            "Shimmer (local, %)": _safe_round(shimmer_local * 100 if shimmer_local is not None else None, 3),
            "HNR (dB)": _safe_round(hnr, 2),
            "Formant F1 (Hz)": _safe_round(f1, 1),
            "Formant F2 (Hz)": _safe_round(f2, 1),
            "Formant F3 (Hz)": _safe_round(f3, 1),
            "success": True
        }

        return results

    except Exception as e:
        return {"success": False, "error": str(e)}
    
# --- ANIMAL MATCHING --- 
def match_animal_by_voice(features):
    animals = [
        "Cat", "Dog", "Horse", "Cow", "Sheep", "Goat", "Lion", "Tiger",
        "Wolf", "Fox", "Monkey", "Donkey", "Deer", "Elephant", "Rabbit"
    ]

    # Pobierz cechy
    f0 = features.get("F0 mean (Hz)", 200)
    f0_std = features.get("F0 std (Hz)", 5)
    jitter = features.get("Jitter", 0.5)
    shimmer = features.get("Shimmer", 2.0)
    rms = features.get("RMS mean", 0.2)

    # Normalizacja cech
    f0_norm = (f0 - 75) / (500 - 75)
    f0_std_norm = min(f0_std / 50, 1)
    jitter_norm = min(jitter / 5, 1)
    shimmer_norm = min(shimmer / 10, 1)
    rms_norm = min(rms / 1, 1)

    scores = {}
    for animal in animals:
        score = 0

        # Małe, szybkie zwierzęta
        if animal in ["Cat", "Rabbit", "Monkey", "Fox"]:
            score += f0_norm * 0.5 + f0_std_norm * 0.3
        # Duże zwierzęta
        elif animal in ["Elephant", "Cow", "Horse", "Donkey"]:
            score += (1 - f0_norm) * 0.5 + rms_norm * 0.3
        # Drapieżniki
        elif animal in ["Lion", "Tiger", "Wolf"]:
            score += shimmer_norm * 0.4 + jitter_norm * 0.4
        # Psowate
        elif animal in ["Dog"]:
            score += jitter_norm * 0.3 + f0_norm * 0.2
        # Owce i kozy
        elif animal in ["Sheep", "Goat", "Deer"]:
            score += (0.5 * f0_norm + 0.5 * rms_norm)

        # Dodajemy kontrolowaną losowość ±10%
        score *= random.uniform(0.9, 1.1)

        scores[animal] = score

    # Posortuj i wybierz top 3, potem losowo wybierz jedno z nich
    top_animals = sorted(scores, key=scores.get, reverse=True)[:3]
    selected_animal = random.choice(top_animals)

    return selected_animal



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
    

def display_audio_info(data, sr, filename=None):
    info = get_audio_info(data, sr)
    col1, col2 = st.columns(2)
    with col1:
        if filename:
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

def extract_and_display_features(audio_path, key_suffix):
    st.subheader("🎵 Advanced Voice Feature Extraction")
    if st.button("🧮 Extract Voice Features", key=f"extract_features_{key_suffix}"):
        with st.spinner("Extracting voice features..."):
            features = extract_voice_features_librosa(audio_path)
            if features["success"]:
                st.success("Voice features extracted successfully!")
                st.session_state.features = features
                st.table({
                    "Feature": list(features.keys())[:-1],
                    "Value": list(features.values())[:-1]
                })
                with st.expander("Click to see parameter explanations"):
                    st.markdown("""
                    ### 🔍 Voice Feature Descriptions

                    **Duration (s)** – total length of the audio recording in seconds.  
                    **F0 mean (Hz)** – average fundamental frequency (pitch) of the voice.  
                    **F0 std (Hz)** – variability of F0; higher values indicate more pitch instability or expressive intonation.  
                    **RMS mean** – average Root Mean Square energy, representing the overall loudness of the signal.  
                    **RMS std** – variability of RMS; shows how much the loudness fluctuates across the recording.  
                    **Jitter** – relative variation of pitch period; higher jitter suggests tremor or instability in vocal fold vibration.  
                    **Shimmer** – relative variation of amplitude; higher shimmer reflects irregularities in vocal intensity.  
                    **MFCC 1–9 mean** – average values of the first nine Mel-Frequency Cepstral Coefficients, describing the spectral and timbral qualities of the voice:
                    - **MFCC 1** – overall spectral slope or brightness of the sound  
                    - **MFCC 2–3** – vowel articulation and general timbre characteristics  
                    - **MFCC 4–9** – fine-grained spectral details, resonances (formants), and tonal richness  
                    """)
            else:
                st.error(f"Feature extraction failed: {features['error']}")

def analyze_parkinson(detector, audio_path, key_suffix):
    st.subheader("🏥 Parkinson's Disease Risk Analysis")
    if detector is not None:
        st.write("Analyze the audio for potential Parkinson's disease indicators.")
        if st.button("🔍 Analyze for Parkinson's Risk", key=f"analyze_{key_suffix}"):
            with st.spinner("Analyzing voice patterns..."):
                results = analyze_parkinson_risk(detector, audio_path)
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
            st.info(f"**Recommendation:** {results['recommendation']}")
            st.caption("⚠️ **Disclaimer:** This analysis is for screening purposes only and should not replace professional medical diagnosis. Always consult with a healthcare provider for medical concerns.")
        else:
            st.error(f"Analysis failed: {results['error']}")

def find_and_display_animal():
    st.subheader("🦊 What Animal Are You?")
    if 'animal_result' not in st.session_state:
        st.session_state.animal_result = None
    if st.button("🎯 Find your animal", key="find_animal"):
        if 'features' in st.session_state and st.session_state.features["success"]:
            animal = match_animal_by_voice(st.session_state.features)
            st.session_state.animal_result = animal
        else:
            st.warning("Please extract voice features first!")
    if st.session_state.animal_result:
        animal = st.session_state.animal_result
        st.success(f"Based on your 'aaa' sound, you are most like a **{animal}**!")
        img_path = f"images/animals/{animal.lower()}.png"
        if os.path.exists(img_path):
            st.image(img_path, width=250)
        else:
            st.info(f"🐾 Imagine a {animal} here!")