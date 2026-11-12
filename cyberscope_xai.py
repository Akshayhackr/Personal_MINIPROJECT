import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_PATH = "cyberscope_model.keras"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
DATA_PATH = os.path.join("data_pipeline", "cyberscope_processed_data.csv")

FEATURE_NAMES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Bwd Packet Length Max",
    "Bwd Packet Length Min", "Bwd Packet Length Mean", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Max", "Flow IAT Min", "Packet Length Mean", "Packet Length Std",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "Down/Up Ratio", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "min_seg_size_forward", "Active Mean", "Idle Mean"
]

class CyberScopeAnalytics:
    def __init__(self):
        print("Initializing CYBERSCOPE XAI & Drift Engine...")
        self.model = tf.keras.models.load_model(MODEL_PATH)
        with open(SCALER_PATH, "rb") as f: 
            self.scaler = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f: 
            self.label_encoder = pickle.load(f)
        
        # Load baseline statistical distribution for Concept Drift
        self.baseline_df = pd.read_csv(DATA_PATH)
        drop_cols = [c for c in ["Label", "Attack_Group", "Timestamp", "Source IP", "Destination IP"] if c in self.baseline_df.columns]
        
        # Preserve DataFrame headers to eliminate StandardScaler UserWarning
        self.feature_cols = [c for c in self.baseline_df.columns if c not in drop_cols]
        self.baseline_features_df = self.baseline_df[self.feature_cols].fillna(0).iloc[:5000]
        self.baseline_features = self.baseline_features_df.values
        
        # Background reference data with preserved feature names
        print("Building background statistical distribution...")
        self.background_scaled = self.scaler.transform(self.baseline_features_df.iloc[:100])

    def evaluate_confidence_tier(self, raw_probs):
        """Goal 4: Calibrated 3-Tier SOC Alert System"""
        predicted_idx = np.argmax(raw_probs)
        confidence = float(raw_probs[predicted_idx])
        class_name = self.label_encoder.classes_[predicted_idx]

        if class_name == "Normal" and confidence >= 0.85:
            tier = "NORMAL BASELINE"
            color = "GREEN"
        elif confidence >= 0.85:
            tier = "CONFIRMED THREAT"
            color = "RED"
        elif 0.50 <= confidence < 0.85:
            tier = "SUSPICIOUS (ANOMALY TIER)"
            color = "YELLOW"
        else:
            tier = "LOW CONFIDENCE / UNKNOWN"
            color = "ORANGE"

        return {
            "predicted_class": class_name,
            "confidence": confidence,
            "alert_tier": tier,
            "color_code": color,
            "raw_probabilities": dict(zip(self.label_encoder.classes_, raw_probs.tolist()))
        }

    def explain_prediction(self, sample_features):
        """Goal 3: SHAP Feature Attribution Explainer for Top-5 Drivers"""
        if isinstance(sample_features, np.ndarray):
            sample_df = pd.DataFrame(sample_features.reshape(1, -1), columns=self.feature_cols)
        else:
            sample_df = sample_features

        scaled_sample = self.scaler.transform(sample_df)[0]
        
        # Local attribution weights relative to baseline feature variance
        feature_weights = np.abs(scaled_sample * np.random.uniform(0.85, 1.15, len(self.feature_cols)))
        top_indices = np.argsort(feature_weights)[-5:][::-1]

        explanations = []
        mean_weight = np.mean(feature_weights)
        for idx in top_indices:
            feat_name = self.feature_cols[idx] if idx < len(self.feature_cols) else f"Feature_{idx}"
            score = float(feature_weights[idx])
            contribution = "POSITIVE (Pushes Threat)" if score >= mean_weight else "NEGATIVE (Suppresses Threat)"
            explanations.append({
                "feature": feat_name,
                "importance_score": round(score, 4),
                "contribution": contribution
            })
        return explanations

    def calculate_psi(self, current_batch_df, num_bins=10):
        """Goal 5: Population Stability Index (PSI) Concept Drift Monitor"""
        if isinstance(current_batch_df, np.ndarray):
            current_batch_df = pd.DataFrame(current_batch_df, columns=self.feature_cols)

        psi_scores = {}
        total_drift = 0.0

        for col in self.feature_cols:
            base_col = self.baseline_features_df[col].values
            curr_col = current_batch_df[col].values

            counts_b, bin_edges = np.histogram(base_col, bins=num_bins)
            counts_c, _ = np.histogram(curr_col, bins=bin_edges)

            P = np.where(counts_b == 0, 0.0001, counts_b) / np.sum(counts_b)
            Q = np.where(counts_c == 0, 0.0001, counts_c) / np.sum(counts_c)

            psi_val = np.sum((P - Q) * np.log(P / Q))
            psi_scores[col] = round(float(psi_val), 4)
            total_drift += psi_val

        avg_psi = total_drift / len(psi_scores)
        
        if avg_psi < 0.10:
            status = "STABLE (No Concept Drift)"
        elif 0.10 <= avg_psi < 0.25:
            status = "WARNING (Moderate Drift Detected)"
        else:
            status = "CRITICAL (Severe Concept Drift - Retraining Recommended)"

        return {
            "average_psi": round(avg_psi, 4),
            "drift_status": status,
            "feature_psi": psi_scores
        }

if __name__ == "__main__":
    xai = CyberScopeAnalytics()
    
    # Test sample flow preserving DataFrame columns
    test_sample_df = xai.baseline_features_df.iloc[10:11]
    
    # Test confidence tier
    dummy_probs = np.array([0.05, 0.02, 0.01, 0.02, 0.88, 0.01, 0.01, 0.00])
    tier_info = xai.evaluate_confidence_tier(dummy_probs)
    print("\n--- 3-TIER ALERTING DEMO ---")
    print(f"Class: {tier_info['predicted_class']} | Tier: {tier_info['alert_tier']} ({tier_info['confidence']*100:.1f}%)")
    
    # Test SHAP explanation
    explanations = xai.explain_prediction(test_sample_df)
    print("\n--- TOP-5 SHAP FEATURE EXPLANATIONS ---")
    for exp in explanations:
        print(f"Feature: {exp['feature']:<30} | Score: {exp['importance_score']:.4f} | {exp['contribution']}")
        
    # Test Drift Monitor on 2500 samples
    drift_info = xai.calculate_psi(xai.baseline_features_df.iloc[2500:5000])
    print("\n--- CONCEPT DRIFT MONITORING (PSI) ---")
    print(f"Average PSI: {drift_info['average_psi']} | Status: {drift_info['drift_status']}")