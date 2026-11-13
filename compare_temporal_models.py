"""
CyberScope AI - Temporal Architecture Benchmark & Comparison
============================================================
Addresses Expectation 3:
Compares the Transformer branch against:
- LSTM
- BiLSTM
- GRU

Evaluates whether the Positional-Encoded Transformer actually provides
superior temporal feature extraction, multi-packet pattern recognition,
and inference throughput compared to recurrent baselines.
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

from data_loader import load_cyberscope_dataset
from models.model_builder import build_cyberscope_hybrid_model
from eval_metrics import compute_comprehensive_metrics


def run_temporal_comparison(
    sample_size=30000,
    epochs=10,
    batch_size=512,
    output_dir="models"
):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 75)
    print("CYBERSCOPE AI - TEMPORAL ARCHITECTURE COMPARATIVE STUDY")
    print("Architectures: Transformer (with Positional Encoding) vs LSTM vs BiLSTM vs GRU")
    print("=" * 75)
    
    # 1. Load benchmark dataset
    print(f"\n[1/4] Loading benchmark dataset (sample size: {sample_size:,})...")
    bundle = load_cyberscope_dataset(sample_size=sample_size, random_state=42)
    
    X_sp_train, X_seq_train, y_train = bundle["X_sp_train"], bundle["X_seq_train"], bundle["y_train"]
    X_sp_val, X_seq_val, y_val = bundle["X_sp_val"], bundle["X_seq_val"], bundle["y_val"]
    X_sp_test, X_seq_test, y_test = bundle["X_sp_test"], bundle["X_seq_test"], bundle["y_test"]
    
    num_classes = bundle["num_classes"]
    class_names = bundle["class_names"]
    num_features = bundle["num_features"]
    
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)
    
    print(f"Train samples: {len(y_train):,}, Val samples: {len(y_val):,}, Test samples: {len(y_test):,}")
    
    architectures = [
        {"name": "Transformer (Pos-Enc)", "type": "transformer", "pe": "sinusoidal"},
        {"name": "LSTM", "type": "lstm", "pe": "none"},
        {"name": "BiLSTM", "type": "bilstm", "pe": "none"},
        {"name": "GRU", "type": "gru", "pe": "none"}
    ]
    
    results = []
    
    print("\n[2/4] Training and benchmarking each temporal branch...")
    for arch in architectures:
        model_name = arch["name"]
        t_type = arch["type"]
        pe_type = arch["pe"]
        
        print("\n" + "-" * 60)
        print(f"Training: {model_name}...")
        print("-" * 60)
        
        # Build model with consistent 1D-CNN spatial branch
        model = build_cyberscope_hybrid_model(
            num_features=num_features,
            seq_len=10,
            num_classes=num_classes,
            cnn_type="1d",
            temporal_type=t_type,
            pe_type=pe_type,
            learning_rate=0.001
        )
        
        param_count = model.count_params()
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
                verbose=0
            )
        ]
        
        start_train = time.perf_counter()
        history = model.fit(
            [X_sp_train, X_seq_train],
            y_train_cat,
            validation_data=([X_sp_val, X_seq_val], y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        train_time = time.perf_counter() - start_train
        actual_epochs = len(history.history["loss"])
        time_per_epoch = train_time / actual_epochs
        
        # Benchmark Inference Latency
        # Warmup
        _ = model.predict([X_sp_test[:64], X_seq_test[:64]], verbose=0)
        
        start_inf = time.perf_counter()
        y_probs = model.predict([X_sp_test, X_seq_test], batch_size=batch_size, verbose=0)
        total_inf_time = time.perf_counter() - start_inf
        latency_ms_per_flow = (total_inf_time / len(X_sp_test)) * 1000.0
        
        y_pred = np.argmax(y_probs, axis=1)
        
        # Compute comprehensive metrics
        summary, per_class_df, _, _, _ = compute_comprehensive_metrics(
            y_test, y_pred, y_probs, class_names
        )
        
        # Infiltration Recall if present
        infil_idx = class_names.index("Infiltration") if "Infiltration" in class_names else -1
        infil_rec = per_class_df.loc[per_class_df["Class"] == "Infiltration", "Recall (Detection Rate)"].values[0] if infil_idx != -1 else 0.0
        
        record = {
            "Architecture": model_name,
            "Temporal Type": t_type.upper(),
            "Positional Encoding": pe_type.title(),
            "Parameters": param_count,
            "Accuracy": round(summary["accuracy"], 5),
            "Macro F1": round(summary["macro_f1"], 5),
            "Weighted F1": round(summary["weighted_f1"], 5),
            "Macro Precision": round(summary["macro_precision"], 5),
            "Macro Recall": round(summary["macro_recall"], 5),
            "ROC-AUC (OvR)": round(summary["macro_roc_auc"], 5),
            "Infiltration Recall": round(float(infil_rec), 4),
            "False Alarms/1000": round(summary["false_alarms_per_1000_normal"], 2),
            "Training Time (s)": round(train_time, 2),
            "Time/Epoch (s)": round(time_per_epoch, 2),
            "Inference Latency (ms/flow)": round(latency_ms_per_flow, 4)
        }
        results.append(record)
        
        print(f"Results for {model_name}:")
        print(f"  Accuracy: {summary['accuracy']*100:.2f}%, Macro F1: {summary['macro_f1']*100:.2f}%, "
              f"ROC-AUC: {summary['macro_roc_auc']:.4f}, Latency: {latency_ms_per_flow:.4f} ms/flow")

    # 3. Save results table
    print("\n[3/4] Exporting comparison results to CSV and JSON...")
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "temporal_architecture_comparison.csv")
    df_results.to_csv(csv_path, index=False)
    
    json_path = os.path.join(output_dir, "temporal_architecture_comparison.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved results: {csv_path}")

    # 4. Generate Visualization Charts
    print("\n[4/4] Generating comparative visualization figures...")
    
    # Figure 1: Performance Metrics Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # Bar Chart of Key Metrics
    metrics_to_plot = ["Macro F1", "Weighted F1", "Accuracy", "ROC-AUC (OvR)"]
    df_melted = pd.melt(
        df_results,
        id_vars=["Architecture"],
        value_vars=metrics_to_plot,
        var_name="Metric",
        value_name="Score"
    )
    
    sns.barplot(data=df_melted, x="Metric", y="Score", hue="Architecture", palette="viridis", ax=axes[0])
    axes[0].set_title("Performance Metrics across Temporal Architectures", fontsize=12, pad=10)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Score (0.0 to 1.0)", fontsize=11)
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)
    axes[0].legend(title="Model", frameon=True)
    
    # Figure 2: Trade-off: Macro F1 vs Inference Latency
    sns.scatterplot(
        data=df_results,
        x="Inference Latency (ms/flow)",
        y="Macro F1",
        hue="Architecture",
        s=180,
        palette="tab10",
        ax=axes[1]
    )
    for _, row in df_results.iterrows():
        axes[1].text(
            row["Inference Latency (ms/flow)"] * 1.01,
            row["Macro F1"] + 0.005,
            row["Architecture"],
            fontsize=10,
            weight="bold"
        )
    axes[1].set_title("Macro F1 vs. Real-Time Inference Latency", fontsize=12, pad=10)
    axes[1].set_xlabel("Inference Latency (ms per flow)", fontsize=11)
    axes[1].set_ylabel("Macro F1-Score", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "temporal_comparison_metrics.png")
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metrics comparison plot: {plot_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("TEMPORAL ARCHITECTURE COMPARISON SUMMARY")
    print("=" * 80)
    print(df_results[["Architecture", "Macro F1", "Weighted F1", "Accuracy", "ROC-AUC (OvR)", "Inference Latency (ms/flow)"]].to_string(index=False))
    print("=" * 80)
    
    return df_results


if __name__ == "__main__":
    run_temporal_comparison(sample_size=25000, epochs=8, batch_size=512)
