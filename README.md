# Echo-Vitals: Advanced Voice Analysis for Health

## Overview

Echo-Vitals is an advanced voice analysis application that uses artificial intelligence to analyze voice recordings for potential indicators of Parkinson's disease. The application features a Streamlit web interface that allows users to either upload audio files or record their voice directly for analysis.

## Features

- **Voice Recording**: Record audio directly through the web interface
- **Audio File Upload**: Support for WAV file uploads
- **Parkinson's Disease Detection**: AI-powered analysis using Inception V3 CNN with transfer learning
- **Voice Feature Extraction**: Advanced vocal feature analysis including jitter, shimmer, harmonics-to-noise ratio (HNR), and formants
- **Model Interpretability**: SHAP and LIME explanations for model predictions
- **Spectrogram Visualization**: Real-time spectrogram generation and display
- **Fun Spirit Animal Matching**: Match users to animals based on their voice characteristics

## Project Structure

```
echo-vitals/
├── echo_vitals_app.py          # Main Streamlit application
├── pd_voice_detection.py       # Parkinson's detection model
├── model_interpretability.py   # SHAP and LIME explanations
├── utils.py                    # Utility functions for audio processing
├── style.css                   # Custom CSS styling
├── README.md                   # This file
└── parkinson_voice_model.h5    # Pre-trained model (not included)
```

## Technology Stack

- **Frontend**: Streamlit web framework
- **Machine Learning**: TensorFlow/Keras with Inception V3 architecture
- **Voice Processing**: librosa, parselmouth (Praat)
- **Interpretability**: SHAP, LIME
- **Visualization**: matplotlib, PIL
- **Audio Processing**: soundfile, scipy

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd echo-vitals
```

2. Install required dependencies:
```bash
pip install streamlit tensorflow librosa parselmouth matplotlib scikit-learn scipy soundfile pillow lime shap st-audiorec
```

3. Run the application:
```bash
streamlit run echo_vitals_app.py
```

## Usage

### Recording Voice

1. Select "Record audio" from the sidebar
2. Follow the recording instructions (record "Ahhh" sound for at least 1.5 seconds)
3. Click the recording button and speak into your microphone
4. The app will automatically process and analyze your recording

### Uploading Audio Files

1. Select "Upload .wav file" from the sidebar
2. Choose a WAV audio file from your computer
3. The app will process and analyze the uploaded file

### Analysis Features

- **Voice Features**: Extract advanced vocal parameters including F0 (pitch), jitter, shimmer, MFCCs, and formants
- **Parkinson's Risk Analysis**: Get probability scores and risk level assessments
- **Model Interpretability**: View SHAP and LIME explanations showing which parts of the audio contribute to predictions
- **Spirit Animal**: Fun feature that matches your voice to an animal based on acoustic characteristics

## Model Architecture

The Parkinson's detection model uses:
- **Inception V3** pre-trained on ImageNet as the base model
- **Transfer Learning** with custom classification layers
- **Input**: 600x600 RGB spectrogram images
- **Output**: Binary classification (Parkinson's/Healthy)
- **Training Parameters**: 8kHz sampling rate, 1.5-second duration, 32ms window

## Voice Features Explained

- **F0 (Fundamental Frequency)**: Average pitch and variability
- **Jitter**: Variation in vocal pitch periods (instability indicator)
- **Shimmer**: Variation in vocal amplitude (voice quality indicator)
- **HNR (Harmonics-to-Noise Ratio)**: Voice clarity measure
- **Formants (F1-F3)**: Resonant frequencies of the vocal tract
- **MFCCs**: Mel-frequency cepstral coefficients for spectral analysis

## Model Interpretability

The application provides two types of explanations:

1. **SHAP (SHapley Additive exPlanations)**: Shows which regions of the spectrogram most influence the model's decision
2. **LIME (Local Interpretable Model-agnostic Explanations)**: Highlights important superpixel regions in the spectrogram

## Disclaimer

⚠️ **Important**: This application is for screening and research purposes only and should not replace professional medical diagnosis. Always consult with a healthcare provider for medical concerns.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Streamlit
- librosa
- parselmouth
- matplotlib
- scikit-learn
- scipy
- soundfile
- PIL (Pillow)
- LIME
- SHAP
- st-audiorec

## Development

### Training the Model

To train your own model, prepare a dataset following this structure:
```
dataset/
├── healthy/
│   ├── sample1.wav
│   └── sample2.wav
└── parkinson/
    ├── sample1.wav
    └── sample2.wav
```

Then run:
```python
from pd_voice_detection import train_model
detector, metrics = train_model("dataset")
```

### Custom Styling

The application uses custom CSS defined in `style.css` with:
- Poppins font family
- Blue gradient background
- Card-based layout with shadows
- Hover effects and smooth transitions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on research by Iyer et al. (2023) on Parkinson's detection using voice spectrograms
- Uses Inception V3 architecture from Google
- Built with Streamlit framework
- Voice processing powered by librosa and Praat/parselmouth
