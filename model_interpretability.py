"""
Model Interpretability Module for Parkinson's Voice Detection
Fast and clear SHAP and LIME explanations for CNN predictions on spectrograms
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from lime import lime_image
from skimage.segmentation import mark_boundaries
import io
from PIL import Image

class ModelInterpreter:
    """
    Provides fast and clear SHAP and LIME explanations for Parkinson's detection model predictions.
    """
    
    def __init__(self, model):
        """
        Initialize the interpreter.
        
        Args:
            model: Trained Keras model
        """
        self.model = model
    
    def explain_with_shap(self, image):
        """
        Generate fast SHAP-style explanation using gradients with conservative visualization.
        
        Args:
            image: Input spectrogram image (H, W, C)
            
        Returns:
            gradients: Gradient values for the image
            explanation_image: Conservative visualization as PIL Image
        """
        if len(image.shape) == 3:
            image_batch = np.expand_dims(image, axis=0)
        else:
            image_batch = image
        
        image_tensor = tf.convert_to_tensor(image_batch, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(image_tensor)
            predictions = self.model(image_tensor)
            if len(predictions.shape) > 1:
                prediction = predictions[:, 0]
            else:
                prediction = predictions
        
        gradients = tape.gradient(prediction, image_tensor).numpy()[0]
        
        # Calculate importance with conservative enhancement
        importance = np.abs(gradients).mean(axis=-1)  # Average across channels
        
        # Normalize with conservative scaling
        importance_norm = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
        
        # Conservative enhancement - only highlight truly important areas
        # Apply threshold to only show top 20% most important areas
        threshold = np.percentile(importance_norm, 80)  # 80th percentile
        importance_thresholded = np.where(importance_norm > threshold, importance_norm, 0)
        
        # Gentle power transformation for the thresholded areas
        importance_final = np.power(importance_thresholded, 0.8)
        
        # Create conservative visualization with larger size
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Show original image with higher opacity
        ax.imshow(image, alpha=0.85)
        
        # Overlay importance heatmap with conservative alpha
        # Use 'Reds' colormap for more conservative coloring
        heatmap = ax.imshow(importance_final, cmap='Reds', alpha=0.4, vmin=0, vmax=1)
        
        # Add clear, informative title
        ax.set_title('SHAP: Red areas indicate regions with highest impact on prediction', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add colorbar for reference
        cbar = plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
        cbar.set_label('Impact Strength', rotation=270, labelpad=20, fontsize=12)
        
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
        buf.seek(0)
        explanation_image = Image.open(buf)
        plt.close(fig)
        
        return gradients, explanation_image
    
    def explain_with_lime(self, image, num_samples=30, num_features=3):
        """
        Generate fast LIME explanation with clear boundary visualization.
        
        Args:
            image: Input spectrogram image (H, W, C)
            num_samples: Number of perturbed samples (reduced for speed)
            num_features: Number of superpixels to highlight
            
        Returns:
            explanation: LIME explanation object
            explanation_image: Clear visualization as PIL Image
        """
        if len(image.shape) == 4:
            image = image[0]
        
        if image.max() > 1.0:
            image = image / 255.0
        
        # Reduce image size for faster processing
        original_shape = image.shape
        if image.shape[0] > 150:
            pil_img = Image.fromarray((image * 255).astype(np.uint8)).resize((150, 150))
            image_small = np.array(pil_img) / 255.0
        else:
            image_small = image.copy()
        
        explainer = lime_image.LimeImageExplainer()
        
        def predict_fn(images):
            """Prediction function that resizes images back to model input size."""
            predictions = []
            for img in images:
                # Resize back to original model input size
                if img.shape != original_shape:
                    pil_img = Image.fromarray((img * 255).astype(np.uint8)).resize(
                        (original_shape[1], original_shape[0])
                    )
                    img_resized = np.array(pil_img) / 255.0
                else:
                    img_resized = img
                predictions.append(img_resized)
            
            predictions = np.array(predictions)
            model_predictions = self.model.predict(predictions, verbose=0)
            
            # Return probabilities for both classes
            if len(model_predictions.shape) == 1 or model_predictions.shape[1] == 1:
                # Binary classification with single output
                probs = np.hstack([1 - model_predictions, model_predictions])
            else:
                probs = model_predictions
            return probs
        
        # Generate LIME explanation with reduced parameters for speed
        explanation = explainer.explain_instance(
            image_small,
            predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=num_samples,
            batch_size=16,
            num_features=20  # More superpixels for segmentation, but only show top ones
        )
        
        # Get the most important features
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=num_features,
            hide_rest=False
        )
        
        # Create clear visualization with enhanced boundaries - same size as SHAP
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Use more subtle boundary colors
        overlay = mark_boundaries(
            temp, 
            mask, 
            color=(0.8, 0, 0),  # Red color for boundaries
            outline_color=None,
            mode='thick'
        )
        
        ax.imshow(overlay)
        
        # Add clear, informative title
        ax.set_title(f'LIME: Red boundaries highlight most important regions for prediction', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
        buf.seek(0)
        explanation_image = Image.open(buf)
        plt.close(fig)
        
        return explanation, explanation_image
    
    def get_feature_importance_scores(self, explanation, top_n=3):
        """
        Extract and format feature importance scores from LIME explanation.
        
        Args:
            explanation: LIME explanation object
            top_n: Number of top features to return
            
        Returns:
            Dictionary with feature information
        """
        if not explanation or not hasattr(explanation, 'top_labels'):
            return None
            
        label = explanation.top_labels[0]
        
        # Get feature weights
        if label in explanation.local_exp:
            features = explanation.local_exp[label][:top_n]
        else:
            return None
        
        importance_data = {
            'feature_ids': [f[0] for f in features],
            'weights': [f[1] for f in features],
            'supporting': sum(1 for f in features if f[1] > 0),
            'contradicting': sum(1 for f in features if f[1] < 0)
        }
        
        return importance_data

def create_fast_interpretation(interpreter, image, mode='both'):
    """
    Create fast interpretation with different modes.
    
    Args:
        interpreter: ModelInterpreter instance
        image: Input spectrogram
        mode: 'shap_only', 'lime_only', 'both' (default)
        
    Returns:
        Dictionary with interpretation results
    """
    report = {
        'shap_image': None,
        'lime_image': None,
        'feature_scores': None,
        'error': None
    }
    
    try:
        if mode in ['shap_only', 'both']:
            # Generate SHAP explanation
            _, shap_image = interpreter.explain_with_shap(image)
            report['shap_image'] = shap_image
            
        if mode in ['lime_only', 'both']:
            # Generate LIME explanation
            lime_explanation, lime_image = interpreter.explain_with_lime(image, num_samples=30, num_features=3)
            report['lime_image'] = lime_image
            
            # Get feature importance scores
            if lime_explanation:
                report['feature_scores'] = interpreter.get_feature_importance_scores(lime_explanation)
                
    except Exception as e:
        report['error'] = str(e)
        print(f"Interpretation error: {e}")
    
    return report

def create_interpretation_report(interpreter, image, prediction_prob):
    """
    Create a comprehensive interpretation report (legacy function for compatibility).
    
    Args:
        interpreter: ModelInterpreter instance
        image: Input spectrogram
        prediction_prob: Model prediction probability
        
    Returns:
        Dictionary with all interpretation results
    """
    report = create_fast_interpretation(interpreter, image, mode='both')
    report['prediction'] = prediction_prob
    return report
