import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy import signal
import soundfile as sf
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import io
from PIL import Image
import tempfile

class ParkinsonVoiceDetector:
    """
    A neural network model for detecting Parkinson's disease from voice spectrograms.
    Based on the approach from Iyer et al. (2023) using Inception V3 with transfer learning.
    """
    
    def __init__(self, input_shape=(600, 600, 3)):
        """
        Initialize the Parkinson's Voice Detector.
        
        Args:
            input_shape: Shape of input spectrograms (height, width, channels)
        """
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
    def create_spectrogram(self, audio_path, duration=1.5, sr=8000):
        """
        Create a spectrogram from an audio file.
        Following the paper's approach: 8kHz sampling, 1.5s duration, 32ms window
        
        Args:
            audio_path: Path to audio file
            duration: Duration to analyze (default 1.5 seconds as per paper)
            sr: Sample rate (default 8000 Hz as per paper)
        
        Returns:
            Spectrogram as numpy array
        """
        # Load audio
        try:
            y, original_sr = librosa.load(audio_path, sr=None, duration=duration)
            
            # Resample to 8kHz if needed (matching paper's telephone quality)
            if original_sr != sr:
                y = librosa.resample(y, orig_sr=original_sr, target_sr=sr)
        except:
            # If librosa fails, try soundfile
            y, original_sr = sf.read(audio_path)
            if len(y.shape) > 1:
                y = y.mean(axis=1)  # Convert to mono
            if original_sr != sr:
                y = librosa.resample(y, orig_sr=original_sr, target_sr=sr)
        
        # Ensure exactly 1.5 seconds
        target_length = int(duration * sr)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            y = y[:target_length]
        
        # Parameters from paper: 32ms window, 50% overlap
        n_fft = 1024  # As specified in paper
        win_length = int(0.032 * sr)  # 32ms window
        hop_length = win_length // 2  # 50% overlap
        
        # Create spectrogram
        D = librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        return S_db
    
    def spectrogram_to_image(self, S_db, target_size=(600, 600)):
        """
        Alternatywna metoda konwersji spektrogramu na obraz RGB.
        Używa PIL i io.BytesIO zamiast buffer_rgba.
        """
        # Opcja 1: Użyj BytesIO
        fig, ax = plt.subplots(figsize=(10, 10), dpi=60)
        
        # Display spectrogram
        librosa.display.specshow(S_db, sr=8000, hop_length=128, 
                                x_axis='time', y_axis='hz', ax=ax,
                                cmap='viridis')
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Zapisz do bufora pamięci
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        
        # Wczytaj jako PIL Image
        img = Image.open(buf)
        
        # Konwertuj na RGB jeśli jest w innym formacie
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Zmień rozmiar
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Konwertuj na numpy array
        image = np.array(img) / 255.0
        
        plt.close(fig)
        buf.close()
        
        return image

    def spectrogram_to_image_tempfile(S_db, target_size=(600, 600)):
        """
        Alternatywna metoda używająca pliku tymczasowego.
        Najbardziej niezawodna, ale wolniejsza.
        """
        fig, ax = plt.subplots(figsize=(10, 10), dpi=60)
        
        # Display spectrogram
        librosa.display.specshow(S_db, sr=8000, hop_length=128, 
                                x_axis='time', y_axis='hz', ax=ax,
                                cmap='viridis')
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Zapisz do pliku tymczasowego
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            plt.savefig(tmp_path, bbox_inches='tight', pad_inches=0, dpi=100)
        
        plt.close(fig)
        
        # Wczytaj obraz
        img = Image.open(tmp_path)
        
        # Konwertuj na RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Zmień rozmiar
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Konwertuj na numpy array
        image = np.array(img) / 255.0
    
    def build_model(self):
        """
        Build the CNN model using Inception V3 with transfer learning.
        Architecture follows the paper: Inception V3 + custom classification layers
        """
        # Load pre-trained Inception V3 (excluding top classification layer)
        base_model = InceptionV3(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers for transfer learning
        base_model.trainable = False
        
        # Build the complete model
        inputs = keras.Input(shape=self.input_shape)
        
        # Data augmentation (optional, helps with small datasets)
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        
        # Pass through Inception V3
        x = base_model(x, training=False)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Custom classification head (as described in paper)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)  # Binary classification
        
        self.model = keras.Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=4):
        """
        Train the model.
        
        Args:
            X_train: Training spectrograms
            y_train: Training labels (0 for healthy, 1 for Parkinson's)
            X_val: Validation spectrograms
            y_val: Validation labels
            epochs: Number of training epochs (default 10 as per paper)
            batch_size: Batch size (default 4 as per paper)
        
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=3,
                restore_best_weights=True,
                mode='max'
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def fine_tune(self, X_train, y_train, X_val, y_val, epochs=5, batch_size=4):
        """
        Fine-tune the model by unfreezing some layers of the base model.
        
        Args:
            X_train, y_train, X_val, y_val: Training and validation data
            epochs: Number of fine-tuning epochs
            batch_size: Batch size
        """
        # Unfreeze the top layers of the base model
        base_model = self.model.layers[4]  # The Inception V3 layer
        base_model.trainable = True
        
        # Freeze all layers except the last 50
        for layer in base_model.layers[:-50]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        # Fine-tune
        history_fine = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return history_fine
    
    def predict(self, audio_path):
        """
        Predict Parkinson's disease probability from an audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Probability of Parkinson's disease (0-1)
        """
        # Create spectrogram
        S_db = self.create_spectrogram(audio_path)
        
        # Convert to image
        image = self.spectrogram_to_image(S_db)
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        # Predict
        probability = self.model.predict(image)[0][0]
        
        return probability
    
    def predict_from_spectrogram_image(self, image_path):
        """
        Predict from a pre-generated spectrogram image.
        
        Args:
            image_path: Path to spectrogram image
        
        Returns:
            Probability of Parkinson's disease (0-1)
        """
        # Load and preprocess image
        image = keras.preprocessing.image.load_img(
            image_path, 
            target_size=self.input_shape[:2]
        )
        image = keras.preprocessing.image.img_to_array(image)
        image = image / 255.0  # Normalize to [0, 1]
        image = np.expand_dims(image, axis=0)
        
        # Predict
        probability = self.model.predict(image)[0][0]
        
        return probability
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Args:
            X_test: Test spectrograms
            y_test: Test labels
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Get predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        loss, accuracy, auc = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'auc': auc,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
    
    def save_model(self, path='parkinson_voice_model.h5'):
        """Save the trained model."""
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path='parkinson_voice_model.h5'):
        """Load a pre-trained model."""
        self.model = keras.models.load_model(path)
        print(f"Model loaded from {path}")


# Example usage and training pipeline
def prepare_dataset(audio_dir, labels_file=None):
    """
    Prepare dataset from directory of audio files.
    
    Args:
        audio_dir: Directory containing audio files
        labels_file: Optional CSV file with filenames and labels
    
    Returns:
        X (spectrograms), y (labels)
    """
    detector = ParkinsonVoiceDetector()
    X = []
    y = []
    
    # This is a placeholder - you'll need to implement based on your data structure
    # The paper mentions 50 PD and 50 HC samples
    
    # Example structure:
    # audio_dir/
    #   healthy/
    #     sample1.wav
    #     sample2.wav
    #   parkinson/
    #     sample1.wav
    #     sample2.wav
    
    for class_name in ['healthy', 'parkinson']:
        class_dir = os.path.join(audio_dir, class_name)
        label = 0 if class_name == 'healthy' else 1
        
        if os.path.exists(class_dir):
            for audio_file in os.listdir(class_dir):
                if audio_file.endswith(('.wav', '.mp3', '.m4a')):
                    audio_path = os.path.join(class_dir, audio_file)
                    try:
                        # Create spectrogram
                        S_db = detector.create_spectrogram(audio_path)
                        # Convert to image
                        image = detector.spectrogram_to_image(S_db)
                        
                        X.append(image)
                        y.append(label)
                    except Exception as e:
                        print(f"Error processing {audio_path}: {e}")
    
    return np.array(X), np.array(y)


# Training script
def train_model(audio_dir):
    """
    Complete training pipeline.
    
    Args:
        audio_dir: Directory containing audio files
    """
    # Prepare dataset
    print("Preparing dataset...")
    X, y = prepare_dataset(audio_dir)
    print(f"Dataset size: {len(X)} samples")
    
    # Split dataset (70% train, 30% test as per paper)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Further split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create and train model
    detector = ParkinsonVoiceDetector()
    detector.build_model()
    print("Model built successfully")
    
    # Initial training
    print("Starting training...")
    history = detector.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=4)
    
    # Fine-tuning (optional)
    print("Fine-tuning model...")
    history_fine = detector.fine_tune(X_train, y_train, X_val, y_val, epochs=5, batch_size=4)
    
    # Evaluate
    print("Evaluating model...")
    metrics = detector.evaluate(X_test, y_test)
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test AUC: {metrics['auc']:.4f}")
    
    # Save model
    detector.save_model()
    
    return detector, metrics


# Inference function for app integration
class ParkinsonDetectorApp:
    """Simple interface for app integration."""
    
    def __init__(self, model_path='parkinson_voice_model.h5'):
        """
        Initialize the detector for app use.
        
        Args:
            model_path: Path to pre-trained model
        """
        self.detector = ParkinsonVoiceDetector()
        self.detector.load_model(model_path)
    
    def predict_from_audio(self, audio_path):
        """
        Predict from audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Dictionary with prediction results
        """
        probability = self.detector.predict(audio_path)
        
        return {
            'probability': float(probability),
            'percentage': f"{probability * 100:.1f}%",
            'risk_level': self._get_risk_level(probability),
            'recommendation': self._get_recommendation(probability)
        }
    
    def predict_from_image(self, image_path):
        """
        Predict from spectrogram image.
        
        Args:
            image_path: Path to spectrogram image
        
        Returns:
            Dictionary with prediction results
        """
        probability = self.detector.predict_from_spectrogram_image(image_path)
        
        return {
            'probability': float(probability),
            'percentage': f"{probability * 100:.1f}%",
            'risk_level': self._get_risk_level(probability),
            'recommendation': self._get_recommendation(probability)
        }
    
    def _get_risk_level(self, probability):
        """Categorize risk level based on probability."""
        if probability < 0.3:
            return "Low"
        elif probability < 0.7:
            return "Moderate"
        else:
            return "High"
    
    def _get_recommendation(self, probability):
        """Provide recommendation based on probability."""
        if probability < 0.3:
            return "Low risk indicators detected. Continue monitoring if symptoms present."
        elif probability < 0.7:
            return "Moderate risk indicators. Consider consultation with a healthcare provider."
        else:
            return "High risk indicators detected. Recommend consultation with a neurologist."


# Example usage for your app
if __name__ == "__main__":
    # For training (run once)
    detector, metrics = train_model("dataset")
    
    # For inference in your app
    app = ParkinsonDetectorApp("parkinson_voice_model.h5")
    
    # Predict from audio
    result = app.predict_from_audio("dataset/healthy/AH_064F_7AB034C9-72E4-438B-A9B3-AD7FDA1596C5.wav")
    
    # Or predict from spectrogram image
    #result = app.predict_from_image("path/to/spectrogram.jpg")
    
    # Print results
    print(f"Parkinson's Probability: {result['percentage']}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Recommendation: {result['recommendation']}")
