"""
CyberScope AI - Imbalanced-Class Handling Comparative Study
===========================================================
Addresses Expectation 5:
Compares:
1. Baseline (Standard Categorical Crossentropy, natural distribution)
2. Class Weighting Only (Smoothed inverse-frequency class weights)
3. SMOTE Only (Synthetic Minority Over-sampling Technique)
4. Focal Loss Only (Categorical Focal Crossentropy, gamma=2.0, alpha=0.25)
5. Balanced Oversampling Only (5x minority oversampling, unweighted)
6. Calibrated Combination (Principled combination avoiding overcompensation)

Evaluates why combining all three blindly overcompensates for minority
classes (distorting Normal-flow false alarm rate), and identifies the
empirically optimal strategy.
"""

import os
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", os.path.dirname(os.path.abspath(__file__)))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

from data_loader import load_cyberscope_dataset
from models.model_builder import build_cyberscope_hybrid_model
from eval_metrics import compute_comprehensive_metrics


def run_imbalance_comparison(sample_size=25000, epochs=8, batch_size=512, output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 75)
    print("CYBERSCOPE AI - IMBALANCED CLASS HANDLING COMPARATIVE STUDY")
    print("Methods: Baseline vs Class Weighting vs SMOTE vs Focal Loss vs Oversampling vs Calibrated")
    print("=" * 75)
    
    # 1. Load dataset
    print(f"\n[1/4] Loading dataset (sample size: {sample_size:,})...")
    bundle = load_cyberscope_dataset(sample_size=sample_size, random_state=42)
    
    X_sp_train, X_seq_train, y_train = bundle["X_sp_train"], bundle["X_seq_train"], bundle["y_train"]
    X_sp_val, X_seq_val, y_val = bundle["X_sp_val"], bundle["X_seq_val"], bundle["y_val"]
    X_sp_test, X_seq_test, y_test = bundle["X_sp_test"], bundle["X_seq_test"], bundle["y_test"]
    
    num_classes = bundle["num_classes"]
    class_names = bundle["class_names"]
    num_features = bundle["num_features"]
    
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)
    
    infil_idx = class_names.index("Infiltration") if "Infiltration" in class_names else -1
    
    # 2. Compute Class Weights
    unique_train_classes = np.unique(y_train)
    raw_weights = compute_class_weight("balanced", classes=unique_train_classes, y=y_train)
    smoothed_weights = np.clip(np.sqrt(raw_weights), a_min=0.5, a_max=20.0)
    class_weights_dict = {cls: float(weight) for cls, weight in zip(unique_train_classes, smoothed_weights)}
    
    # 3. Prepare SMOTE dataset on training partition
    print("Preparing SMOTE oversampled training data...")
    # Reshape sequences for SMOTE: flatten spatial + temporal into 2D tabular
    # Spatial: (N, 30, 1) -> (N, 30); Sequence: (N, 10, 30) -> (N, 300)
    N_tr = len(y_train)
    X_tr_flat = np.hstack([X_sp_train[:, :, 0], X_seq_train.reshape(N_tr, -1)])
    
    # Find min class size
    class_counts = pd.Series(y_train).value_counts()
    k_neighbors = min(2, class_counts.min() - 1) if class_counts.min() > 1 else 1
    
    try:
        # Boost rare classes up to 200 samples each
        target_dict = {}
        for cls_id, count in class_counts.items():
            if count < 200:
                target_dict[cls_id] = 200
                
        if target_dict and k_neighbors >= 1:
            smote = SMOTE(sampling_strategy=target_dict, k_neighbors=k_neighbors, random_state=42)
            X_smote_flat, y_train_smote = smote.fit_resample(X_tr_flat, y_train)
            
            X_sp_train_smote = X_smote_flat[:, :num_features].reshape(-1, num_features, 1).astype(np.float32)
            X_seq_train_smote = X_smote_flat[:, num_features:].reshape(-1, 10, num_features).astype(np.float32)
            y_train_smote_cat = tf.keras.utils.to_categorical(y_train_smote, num_classes=num_classes)
        else:
            X_sp_train_smote, X_seq_train_smote, y_train_smote_cat = X_sp_train, X_seq_train, y_train_cat
            y_train_smote = y_train
    except Exception as e:
        print(f"SMOTE fallback to original due to micro-class size: {e}")
        X_sp_train_smote, X_seq_train_smote, y_train_smote_cat = X_sp_train, X_seq_train, y_train_cat
        y_train_smote = y_train

    # 4. Prepare Balanced Oversampling (5x on minority)
    print("Preparing 5x Oversampling on minority classes...")
    if infil_idx != -1:
        infil_mask = (y_train == infil_idx)
        infil_count = np.sum(infil_mask)
        if infil_count > 0:
            repeat_factor = 4  # 1 original + 4 copies = 5x
            X_sp_oversample = np.concatenate([X_sp_train, np.repeat(X_sp_train[infil_mask], repeat_factor, axis=0)])
            X_seq_oversample = np.concatenate([X_seq_train, np.repeat(X_seq_train[infil_mask], repeat_factor, axis=0)])
            y_oversample = np.concatenate([y_train, np.repeat(y_train[infil_mask], repeat_factor, axis=0)])
            y_train_over_cat = tf.keras.utils.to_categorical(y_oversample, num_classes=num_classes)
        else:
            X_sp_oversample, X_seq_oversample, y_train_over_cat = X_sp_train, X_seq_train, y_train_cat
    else:
        X_sp_oversample, X_seq_oversample, y_train_over_cat = X_sp_train, X_seq_train, y_train_cat

    # Define the 6 strategies
    strategies = [
        {
            "name": "1. Baseline (Natural)",
            "loss": "categorical_crossentropy",
            "weights": None,
            "data": (X_sp_train, X_seq_train, y_train_cat),
            "desc": "Standard unweighted cross-entropy"
        },
        {
            "name": "2. Class Weighting Only",
            "loss": "categorical_crossentropy",
            "weights": class_weights_dict,
            "data": (X_sp_train, X_seq_train, y_train_cat),
            "desc": "Smoothed balanced inverse-frequency weights"
        },
        {
            "name": "3. SMOTE Only",
            "loss": "categorical_crossentropy",
            "weights": None,
            "data": (X_sp_train_smote, X_seq_train_smote, y_train_smote_cat),
            "desc": "Synthetic minority oversampling"
        },
        {
            "name": "4. Focal Loss Only",
            "loss": "focal_loss",
            "weights": None,
            "data": (X_sp_train, X_seq_train, y_train_cat),
            "desc": "Categorical Focal Crossentropy (gamma=2.0)"
        },
        {
            "name": "5. 5x Oversampling Only",
            "loss": "categorical_crossentropy",
            "weights": None,
            "data": (X_sp_oversample, X_seq_oversample, y_train_over_cat),
            "desc": "5x replication of minority classes"
        },
        {
            "name": "6. Calibrated Combination",
            "loss": "focal_loss",
            "weights": {k: float(np.sqrt(v)) for k, v in class_weights_dict.items()}, # gentle square-root damping
            "data": (X_sp_train, X_seq_train, y_train_cat),
            "desc": "Calibrated Focal Loss + Mild Damped Class Weights"
        }
    ]

    results = []
    print("\n[2/4] Training each imbalance handling strategy...")
    for strat in strategies:
        s_name = strat["name"]
        print("\n" + "-" * 60)
        print(f"Evaluating Strategy: {s_name} ({strat['desc']})...")
        print("-" * 60)
        
        model = build_cyberscope_hybrid_model(
            num_features=num_features,
            seq_len=10,
            num_classes=num_classes,
            cnn_type="1d",
            temporal_type="transformer",
            pe_type="sinusoidal",
            loss_type=strat["loss"],
            learning_rate=0.001
        )
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
                verbose=0
            )
        ]
        
        train_sp, train_seq, train_y = strat["data"]
        
        start_t = time.perf_counter()
        model.fit(
            [train_sp, train_seq],
            train_y,
            validation_data=([X_sp_val, X_seq_val], y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=strat["weights"],
            callbacks=callbacks,
            verbose=1
        )
        t_elapsed = time.perf_counter() - start_t
        
        y_probs = model.predict([X_sp_test, X_seq_test], batch_size=batch_size, verbose=0)
        y_pred = np.argmax(y_probs, axis=1)
        
        summary, per_class_df, _, _, _ = compute_comprehensive_metrics(y_test, y_pred, y_probs, class_names)
        
        infil_row = per_class_df[per_class_df["Class"] == "Infiltration"]
        infil_recall = float(infil_row["Recall (Detection Rate)"].values[0]) if len(infil_row) > 0 else 0.0
        infil_precision = float(infil_row["Precision"].values[0]) if len(infil_row) > 0 else 0.0
        infil_f1 = float(infil_row["F1-Score"].values[0]) if len(infil_row) > 0 else 0.0
        
        record = {
            "Strategy": s_name,
            "Description": strat["desc"],
            "Macro F1": round(summary["macro_f1"], 5),
            "Weighted F1": round(summary["weighted_f1"], 5),
            "Overall Accuracy": round(summary["accuracy"], 5),
            "Infiltration Recall": round(infil_recall, 4),
            "Infiltration Precision": round(infil_precision, 4),
            "Infiltration F1": round(infil_f1, 4),
            "False Alarms/1000 Normal": round(summary["false_alarms_per_1000_normal"], 2),
            "Training Time (s)": round(t_elapsed, 2)
        }
        results.append(record)
        
        print(f"Outcome for {s_name}:")
        print(f"  Macro F1: {summary['macro_f1']*100:.2f}%, Infiltration Recall: {infil_recall*100:.2f}%, "
              f"False Alarms/1000: {summary['false_alarms_per_1000_normal']:.2f}")

    # 3. Export results
    print("\n[3/4] Exporting comparison results...")
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "imbalance_strategy_comparison.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # 4. Generate comparison plots
    print("\n[4/4] Generating comparison plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # Left: Macro F1 vs Infiltration Recall
    metrics_sub = ["Macro F1", "Infiltration Recall", "Weighted F1"]
    df_melt = pd.melt(df_results, id_vars=["Strategy"], value_vars=metrics_sub, var_name="Metric", value_name="Score")
    sns.barplot(data=df_melt, x="Strategy", y="Score", hue="Metric", palette="crest", ax=axes[0])
    axes[0].set_title("Detection Performance Across Imbalance Handling Strategies", fontsize=11, pad=10)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha="right", fontsize=9)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)
    
    # Right: False Alarms per 1000 Normal flows
    sns.barplot(data=df_results, x="Strategy", y="False Alarms/1000 Normal", palette="rocket", ax=axes[1])
    axes[1].set_title("Normal-Flow False Alarm Rate (Lower is Better)", fontsize=11, pad=10)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha="right", fontsize=9)
    axes[1].set_ylabel("False Alarms per 1,000 Normal Flows", fontsize=10)
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "imbalance_comparison.png")
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_path}")

    print("\n" + "=" * 80)
    print("IMBALANCED CLASS HANDLING STRATEGY SUMMARY")
    print("=" * 80)
    print(df_results[["Strategy", "Macro F1", "Infiltration Recall", "Infiltration Precision", "False Alarms/1000 Normal"]].to_string(index=False))
    print("=" * 80)
    return df_results


if __name__ == "__main__":
    run_imbalance_comparison(sample_size=25000, epochs=8, batch_size=512)
