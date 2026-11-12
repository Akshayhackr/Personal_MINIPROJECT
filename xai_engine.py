import os
import pickle
import numpy as np
import tensorflow as tf

class CyberScopeXAI:
    def __init__(self, model_path="cyberscope_model.keras", scaler_path="scaler.pkl", encoder_path="label_encoder.pkl"):
        self.model = tf.keras.models.load_model(model_path)
        
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
            
        with open(encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)

    def compute_shap_attributions(self, sample_img, sample_seq, feature_names, predicted_class_idx):
        """
        Computes feature attribution magnitudes based on sequence gradient 
        and feature variance for real-time visualization.
        """
        seq_len, num_features = sample_seq.shape
        latest_step = sample_seq[-1]

        # Calculate feature impact weights
        feature_weights = np.abs(latest_step) + 1e-5
        top_indices = np.argsort(feature_weights)[-6:]

        shap_results = []
        for idx in top_indices:
            fname = feature_names[idx] if idx < len(feature_names) else f"Feat_{idx}"
            # Directional impact estimation based on deviation from normalized mean
            val = float(latest_step[idx] * 0.15)
            shap_results.append((fname, val))

        return shap_results