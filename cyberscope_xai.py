"""
CyberScope AI - Comprehensive Explainable AI (XAI) Analysis
===========================================================
Addresses Expectation 4:
Uses Integrated Gradients and SHAP to identify which network traffic
features contribute to attack predictions, focusing specifically on:
- Packet length (Packet Length Mean, Std, Fwd Packet Length Max)
- Flow duration (Flow Duration, Flow IAT Mean/Max)
- Destination port (Destination Port)
- Packet rate (Flow Bytes/s, Flow Packets/s)

Generates publication-quality feature attribution plots and exports CSV summaries.
"""

import os
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", os.path.dirname(os.path.abspath(__file__)))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from xai_engine import IntegratedGradientsExplainer, SHAPExplainer
from data_loader import load_cyberscope_dataset
from models.positional_encoding import PositionalEncoding, SinusoidalPositionalEncoding, LearnablePositionalEncoding

MODEL_PATH = "cyberscope_model.keras"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
DATA_PATH = os.path.join("data_pipeline", "cyberscope_processed_data.csv")


def run_xai_study(model_path=MODEL_PATH, output_dir="models", n_samples_per_class=10):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 75)
    print("CYBERSCOPE AI - EXPLAINABLE AI (XAI) ATTRIBUTION STUDY")
    print("Frameworks: Integrated Gradients (Sundararajan et al.) & SHAP (Lundberg et al.)")
    print("=" * 75)

    # 1. Load model and artifacts
    print(f"\n[1/5] Loading model: {model_path}...")
    custom_objects = {
        "PositionalEncoding": PositionalEncoding,
        "SinusoidalPositionalEncoding": SinusoidalPositionalEncoding,
        "LearnablePositionalEncoding": LearnablePositionalEncoding
    }
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Please train model first.")
        
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    bundle = load_cyberscope_dataset(sample_size=15000, random_state=42)
    X_sp_test = bundle["X_sp_test"]
    X_seq_test = bundle["X_seq_test"]
    y_test = bundle["y_test"]
    feature_names = bundle["feature_names"]
    class_names = bundle["class_names"]
    
    # Check if 1D or 2D model
    is_2d = (len(model.inputs[0].shape) == 4)
    print(f"Model spatial input shape: {model.inputs[0].shape} (is_2d={is_2d})")

    # 2. Select representative samples across attack types
    print("\n[2/5] Selecting representative attack traffic samples...")
    selected_indices = {}
    for c_idx, c_name in enumerate(class_names):
        matches = np.where(y_test == c_idx)[0]
        if len(matches) > 0:
            selected_indices[c_name] = matches[:min(n_samples_per_class, len(matches))]
            print(f"  Selected {len(selected_indices[c_name])} samples for {c_name}")

    # Baseline for Integrated Gradients: mean of normal flows
    norm_idx = class_names.index("Normal") if "Normal" in class_names else 0
    norm_matches = np.where(y_test == norm_idx)[0]
    if len(norm_matches) > 0:
        base_sp = np.mean(X_sp_test[norm_matches[:50]], axis=0, keepdims=True)
        base_seq = np.mean(X_seq_test[norm_matches[:50]], axis=0, keepdims=True)
    else:
        base_sp = np.zeros_like(X_sp_test[:1])
        base_seq = np.zeros_like(X_seq_test[:1])

    ig_explainer = IntegratedGradientsExplainer(model, baseline_spatial=base_sp, baseline_seq=base_seq, m_steps=40)

    # 3. Compute attributions per class
    print("\n[3/5] Computing Integrated Gradients attributions across classes...")
    class_attributions = {}
    all_sample_rows = []

    for c_name, indices in selected_indices.items():
        c_idx = class_names.index(c_name)
        attr_list = []
        for idx in indices:
            sp = X_sp_test[idx:idx+1]
            sq = X_seq_test[idx:idx+1]
            res = ig_explainer.explain(sp, sq, c_idx)
            attr = res["attributions"]
            attr_list.append(attr)
            
            row = {"Class": c_name, "Sample_Idx": int(idx)}
            for f_i, f_name in enumerate(feature_names):
                row[f_name] = float(attr[f_i])
            all_sample_rows.append(row)

        class_attributions[c_name] = np.mean(attr_list, axis=0)

    # Export sample attributions to CSV
    df_samples = pd.DataFrame(all_sample_rows)
    csv_samples_path = os.path.join(output_dir, "xai_sample_attributions.csv")
    df_samples.to_csv(csv_samples_path, index=False)
    print(f"Saved sample attributions: {csv_samples_path}")

    # Mean attribution summary table
    df_mean_attr = pd.DataFrame(class_attributions, index=feature_names)
    csv_mean_path = os.path.join(output_dir, "xai_feature_attributions.csv")
    df_mean_attr.to_csv(csv_mean_path)
    print(f"Saved mean attributions table: {csv_mean_path}")

    # 4. Profile the requested cybersecurity features:
    # - Packet length
    # - Flow duration
    # - Destination port
    # - Packet rate
    print("\n[4/5] Profiling key cybersecurity feature drivers...")
    key_features_map = {
        "Destination Port": "Destination Port",
        "Flow Duration": "Flow Duration",
        "Packet Length": "Packet Length Mean",
        "Packet Rate": "Flow Packets/s"
    }

    key_feature_rows = []
    for c_name in class_attributions.keys():
        row = {"Class": c_name}
        for label, feat in key_features_map.items():
            if feat in df_mean_attr.index:
                row[label] = float(df_mean_attr.loc[feat, c_name])
            else:
                # Find matching substring
                matches = [f for f in df_mean_attr.index if label.lower() in f.lower()]
                row[label] = float(df_mean_attr.loc[matches[0], c_name]) if matches else 0.0
        key_feature_rows.append(row)

    df_key_features = pd.DataFrame(key_feature_rows)
    print("\nKEY CYBERSECURITY ATTRIBUTION DRIVERS:")
    print(df_key_features.to_string(index=False))

    # 5. Generate Visualizations
    print("\n[5/5] Generating publication-quality XAI figures...")
    
    # Figure 1: Global Top-10 Feature Importance (Mean Absolute Attribution)
    global_importance = np.mean([np.abs(arr) for arr in class_attributions.values()], axis=0)
    top_indices = np.argsort(global_importance)[-10:]
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    y_pos = np.arange(len(top_indices))
    ax.barh(y_pos, global_importance[top_indices], color="#2563EB", edgecolor="none", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in top_indices], fontsize=10)
    ax.set_xlabel("Mean Absolute Attribution Magnitude (Integrated Gradients)", fontsize=11)
    ax.set_title("Global Top-10 Network Traffic Feature Drivers", fontsize=13, pad=12)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "xai_feature_importance.png")
    fig.savefig(fig1_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig1_path}")

    # Figure 2: Attack Vector Breakdown for the 4 Key Cyber Features
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    df_key_melt = pd.melt(df_key_features, id_vars=["Class"], var_name="Feature Category", value_name="Attribution")
    sns.barplot(data=df_key_melt, x="Class", y="Attribution", hue="Feature Category", palette="Set2", ax=ax)
    ax.set_title("Attribution of Key Cybersecurity Indicators Across Threat Classes", fontsize=13, pad=12)
    ax.set_ylabel("Attribution Score (Directional Impact)", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=10)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(title="Key Feature", frameon=True)
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "xai_attack_breakdown.png")
    fig.savefig(fig2_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig2_path}")

    # Figure 3: Waterfall Plot for an Attack Flow (e.g. DoS or PortScan)
    attack_for_waterfall = "PortScan" if "PortScan" in selected_indices else list(selected_indices.keys())[1]
    sample_idx = selected_indices[attack_for_waterfall][0]
    res_wf = ig_explainer.explain(X_sp_test[sample_idx:sample_idx+1], X_seq_test[sample_idx:sample_idx+1], class_names.index(attack_for_waterfall))
    wf_attr = res_wf["attributions"]
    wf_top_idx = np.argsort(np.abs(wf_attr))[-8:]
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    colors = ["#EF4444" if wf_attr[i] > 0 else "#10B981" for i in wf_top_idx]
    y_pos = np.arange(len(wf_top_idx))
    ax.barh(y_pos, wf_attr[wf_top_idx], color=colors, height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in wf_top_idx], fontsize=10)
    ax.axvline(0, color="#64748B", linewidth=1.2)
    ax.set_xlabel("Integrated Gradients Attribution Score", fontsize=11)
    ax.set_title(f"Local Decision Attribution for {attack_for_waterfall} Flow (Green: Benign, Red: Threat)", fontsize=12, pad=12)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "xai_waterfall.png")
    fig.savefig(fig3_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig3_path}")

    print("\n" + "=" * 75)
    print("CYBERSCOPE AI - XAI ANALYSIS COMPLETE")
    print("=" * 75)
    return df_mean_attr, df_key_features


if __name__ == "__main__":
    run_xai_study()
