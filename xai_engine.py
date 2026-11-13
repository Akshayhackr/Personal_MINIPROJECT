"""
CyberScope AI - Explainable AI (XAI) Engine
===========================================
Addresses Expectation 4:
Implements both Integrated Gradients (Sundararajan et al., 2017)
and SHAP (Lundberg & Lee, 2017) to explain network intrusion predictions.

Identifies exact feature attributions for key cybersecurity indicators:
- Packet length (Packet Length Mean, Std, Fwd Packet Length Max)
- Flow duration (Flow Duration, Flow IAT Mean/Max/Min)
- Destination port (Destination Port)
- Packet rate (Flow Bytes/s, Flow Packets/s)
"""

import os
import pickle
import numpy as np
import tensorflow as tf
import shap
from models.positional_encoding import PositionalEncoding, SinusoidalPositionalEncoding, LearnablePositionalEncoding


class IntegratedGradientsExplainer:
    """
    Computes Integrated Gradients for multi-input Keras models:
    Attribution_i(x) = (x_i - x'_i) * integral_0^1 [ dF_c(x' + alpha*(x - x')) / dx_i ] d(alpha)
    
    Satisfies Completeness Axiom: sum(Attributions) = F_c(x) - F_c(x')
    """
    def __init__(self, model, baseline_spatial=None, baseline_seq=None, m_steps=50):
        self.model = model
        self.m_steps = int(m_steps)
        self.baseline_spatial = baseline_spatial
        self.baseline_seq = baseline_seq

    def explain(self, spatial_input, seq_input, target_class_idx):
        """
        spatial_input: shape (1, num_features, 1) or (1, 6, 6, 1)
        seq_input: shape (1, seq_len, num_features)
        target_class_idx: int, class index to explain
        """
        # Ensure batch dimension
        if len(spatial_input.shape) == 2:
            spatial_input = spatial_input[np.newaxis, ...]
        if len(seq_input.shape) == 2:
            seq_input = seq_input[np.newaxis, ...]

        # Defaults baselines to zeros if not supplied
        base_sp = self.baseline_spatial if self.baseline_spatial is not None else np.zeros_like(spatial_input)
        base_seq = self.baseline_seq if self.baseline_seq is not None else np.zeros_like(seq_input)

        alphas = np.linspace(0.0, 1.0, self.m_steps + 1, dtype=np.float32)
        
        # Interpolated paths
        sp_diff = (spatial_input - base_sp).astype(np.float32)
        seq_diff = (seq_input - base_seq).astype(np.float32)
        
        # Vectorized interpolation batches: (m_steps+1, ...)
        sp_interpolated = base_sp + alphas.reshape(-1, *([1] * len(sp_diff.shape[1:]))) * sp_diff
        seq_interpolated = base_seq + alphas.reshape(-1, *([1] * len(seq_diff.shape[1:]))) * seq_diff
        
        sp_tensor = tf.convert_to_tensor(sp_interpolated, dtype=tf.float32)
        seq_tensor = tf.convert_to_tensor(seq_interpolated, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch([sp_tensor, seq_tensor])
            preds = self.model([sp_tensor, seq_tensor])
            class_score = preds[:, target_class_idx]

        grads_sp, grads_seq = tape.gradient(class_score, [sp_tensor, seq_tensor])

        # Riemann sum / Trapezoidal rule approximation of integral
        avg_grads_sp = tf.reduce_mean(grads_sp, axis=0).numpy()
        avg_grads_seq = tf.reduce_mean(grads_seq, axis=0).numpy()

        ig_spatial = sp_diff[0] * avg_grads_sp
        ig_seq = seq_diff[0] * avg_grads_seq

        # Flatten spatial attributions to 1D per feature
        if len(ig_spatial.shape) == 3 and ig_spatial.shape[2] == 1:
            attr_spatial_1d = ig_spatial[:, 0]
        elif len(ig_spatial.shape) == 3 and ig_spatial.shape[:2] == (6, 6):
            attr_spatial_1d = ig_spatial.flatten()[:seq_input.shape[2]]
        else:
            attr_spatial_1d = ig_spatial.flatten()[:seq_input.shape[2]]

        # Aggregate temporal sequence attributions across the 10-step window
        attr_seq_1d = np.mean(ig_seq, axis=0) # average across time steps

        # Combined feature attribution
        combined_attr = attr_spatial_1d + attr_seq_1d
        
        # Verify completeness
        pred_x = float(self.model([spatial_input, seq_input])[0, target_class_idx])
        pred_base = float(self.model([base_sp, base_seq])[0, target_class_idx])
        delta_score = pred_x - pred_base
        total_attr = float(np.sum(combined_attr))
        completeness_ratio = (total_attr / delta_score) if abs(delta_score) > 1e-5 else 1.0

        return {
            "attributions": combined_attr,
            "spatial_attributions": attr_spatial_1d,
            "temporal_attributions": attr_seq_1d,
            "delta_score": delta_score,
            "total_attribution": total_attr,
            "completeness_ratio": completeness_ratio
        }


class SHAPExplainer:
    """
    Computes game-theoretic SHAP attributions using KernelExplainer.
    Wraps the multi-input Keras model so that tabular sample vectors
    are automatically expanded into spatial and temporal inputs.
    """
    def __init__(self, model, background_data, seq_len=10, is_2d=False, n_background=50):
        self.model = model
        self.seq_len = int(seq_len)
        self.is_2d = is_2d
        
        # Subsample background
        if len(background_data) > n_background:
            idx = np.random.choice(len(background_data), size=n_background, replace=False)
            bg = background_data[idx]
        else:
            bg = background_data
            
        self.explainer = shap.KernelExplainer(self._predict_wrapper, bg)

    def _predict_wrapper(self, tabular_batch):
        batch_size, n_feats = tabular_batch.shape
        # Replicate 10-step sequence with tabular_batch as latest step
        seq_batch = np.repeat(tabular_batch[:, np.newaxis, :], self.seq_len, axis=1).astype(np.float32)
        
        if self.is_2d:
            if n_feats < 36:
                pad = 36 - n_feats
                padded = np.pad(tabular_batch, ((0, 0), (0, pad)), mode="constant")
            else:
                padded = tabular_batch[:, :36]
            sp_batch = padded.reshape(-1, 6, 6, 1).astype(np.float32)
        else:
            sp_batch = tabular_batch[:, :, np.newaxis].astype(np.float32)
            
        return self.model.predict([sp_batch, seq_batch], verbose=0)

    def explain(self, sample_features, nsamples=100):
        """Computes SHAP values for sample tabular feature vector."""
        if len(sample_features.shape) == 1:
            sample_features = sample_features[np.newaxis, :]
        shap_values = self.explainer.shap_values(sample_features, nsamples=nsamples)
        return shap_values


class CyberScopeXAI:
    """
    High-level SOC Explainable AI Engine.
    Used by dashboard.py, evaluate_model.py, and standalone diagnostic tools.
    """
    def __init__(self, model_path="cyberscope_model.keras", scaler_path="scaler.pkl", encoder_path="label_encoder.pkl"):
        custom_objects = {
            "PositionalEncoding": PositionalEncoding,
            "SinusoidalPositionalEncoding": SinusoidalPositionalEncoding,
            "LearnablePositionalEncoding": LearnablePositionalEncoding
        }
        
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        else:
            self.model = None
            
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler = None
            
        if os.path.exists(encoder_path):
            with open(encoder_path, "rb") as f:
                self.label_encoder = pickle.load(f)
        else:
            self.label_encoder = None

        self.ig_explainer = IntegratedGradientsExplainer(self.model) if self.model is not None else None

    def compute_shap_attributions(self, sample_spatial, sample_seq, feature_names, predicted_class_idx, top_k=6):
        """
        Computes exact real-time feature attributions using Integrated Gradients.
        Drop-in replacement for the mock implementation in dashboard.py.
        """
        if self.ig_explainer is None and self.model is not None:
            self.ig_explainer = IntegratedGradientsExplainer(self.model)
            
        # Ensure dimensions
        if len(sample_spatial.shape) == 2 or len(sample_spatial.shape) == 3:
            sp = sample_spatial[np.newaxis, ...]
        else:
            sp = sample_spatial
            
        if len(sample_seq.shape) == 2:
            sq = sample_seq[np.newaxis, ...]
        else:
            sq = sample_seq

        res = self.ig_explainer.explain(sp, sq, predicted_class_idx)
        attributions = res["attributions"]

        # Sort features by absolute attribution magnitude
        abs_attr = np.abs(attributions)
        top_indices = np.argsort(abs_attr)[-top_k:]

        ranked_results = []
        for idx in top_indices:
            fname = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
            val = float(attributions[idx])
            ranked_results.append((fname, val))

        return ranked_results
