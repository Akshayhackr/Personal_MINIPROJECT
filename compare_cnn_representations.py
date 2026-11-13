"""
CyberScope AI - CNN Feature Representation Benchmark
====================================================
Addresses Expectation 2:
Compares:
1. 2D-CNN (6x6 matrix with zero-padding)
2. 1D-CNN (Direct tabular network traffic)
3. 1D-CNN (Domain-ordered semantic grouping)

Demonstrates whether 1D-CNN designed for tabular data eliminates
artificial 2D spatial adjacency artifacts and improves feature representation.
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
from models.feature_representation import get_feature_reordering_indices
from models.model_builder import build_cyberscope_hybrid_model
from eval_metrics import compute_comprehensive_metrics


def prepare_2d_grid_input(X_spatial, target_dim=36):
    """
    Pads tabular features to 36 and reshapes into 6x6x1 matrix.
    """
    n_samples, n_feats, _ = X_spatial.shape
    if n_feats < target_dim:
        pad_width = target_dim - n_feats
        padded = np.pad(X_spatial[:, :, 0], ((0, 0), (0, pad_width)), mode="constant")
    else:
        padded = X_spatial[:, :target_dim, 0]
    return padded.reshape(-1, 6, 6, 1).astype(np.float32)


def run_cnn_comparison(sample_size=25000, epochs=8, batch_size=512, output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 75)
    print("CYBERSCOPE AI - CNN FEATURE REPRESENTATION ABLATION STUDY")
    print("Architectures: 2D-CNN (6x6 Grid) vs 1D-CNN (Tabular) vs 1D-CNN (Domain-Ordered)")
    print("=" * 75)
    
    # 1. Load dataset
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
    
    # Domain reordered features
    reorder_idx = get_feature_reordering_indices(bundle["feature_names"])
    X_sp_train_domain = X_sp_train[:, reorder_idx, :]
    X_sp_val_domain = X_sp_val[:, reorder_idx, :]
    X_sp_test_domain = X_sp_test[:, reorder_idx, :]
    
    # 2D grid features (6x6)
    X_2d_train = prepare_2d_grid_input(X_sp_train)
    X_2d_val = prepare_2d_grid_input(X_sp_val)
    X_2d_test = prepare_2d_grid_input(X_sp_test)
    
    experiments = [
        {
            "name": "2D-CNN (6x6 Grid)",
            "cnn_type": "2d",
            "train_spatial": X_2d_train,
            "val_spatial": X_2d_val,
            "test_spatial": X_2d_test,
            "description": "Legacy 6x6 grid with zero padding"
        },
        {
            "name": "1D-CNN (Standard Order)",
            "cnn_type": "1d",
            "train_spatial": X_sp_train,
            "val_spatial": X_sp_val,
            "test_spatial": X_sp_test,
            "description": "Direct 1D-CNN on raw tabular traffic"
        },
        {
            "name": "1D-CNN (Domain-Ordered)",
            "cnn_type": "1d",
            "train_spatial": X_sp_train_domain,
            "val_spatial": X_sp_val_domain,
            "test_spatial": X_sp_test_domain,
            "description": "1D-CNN with protocol-domain feature grouping"
        }
    ]
    
    results = []
    print("\n[2/4] Training and evaluating CNN spatial representations...")
    for exp in experiments:
        name = exp["name"]
        print("\n" + "-" * 60)
        print(f"Training: {name} ({exp['description']})...")
        print("-" * 60)
        
        model = build_cyberscope_hybrid_model(
            num_features=num_features,
            seq_len=10,
            num_classes=num_classes,
            cnn_type=exp["cnn_type"],
            temporal_type="transformer",
            pe_type="sinusoidal",
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
        
        start_train = time.perf_counter()
        history = model.fit(
            [exp["train_spatial"], X_seq_train],
            y_train_cat,
            validation_data=([exp["val_spatial"], X_seq_val], y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        train_time = time.perf_counter() - start_train
        
        # Benchmark inference
        _ = model.predict([exp["test_spatial"][:64], X_seq_test[:64]], verbose=0)
        start_inf = time.perf_counter()
        y_probs = model.predict([exp["test_spatial"], X_seq_test], batch_size=batch_size, verbose=0)
        total_inf_time = time.perf_counter() - start_inf
        latency_ms_per_flow = (total_inf_time / len(y_probs)) * 1000.0
        
        y_pred = np.argmax(y_probs, axis=1)
        summary, per_class_df, _, _, _ = compute_comprehensive_metrics(y_test, y_pred, y_probs, class_names)
        
        record = {
            "Representation": name,
            "Input Shape": str(exp["train_spatial"].shape[1:]),
            "Parameters": model.count_params(),
            "Accuracy": round(summary["accuracy"], 5),
            "Macro F1": round(summary["macro_f1"], 5),
            "Weighted F1": round(summary["weighted_f1"], 5),
            "Macro Precision": round(summary["macro_precision"], 5),
            "Macro Recall": round(summary["macro_recall"], 5),
            "ROC-AUC (OvR)": round(summary["macro_roc_auc"], 5),
            "Training Time (s)": round(train_time, 2),
            "Inference Latency (ms/flow)": round(latency_ms_per_flow, 4)
        }
        results.append(record)
        print(f"Results for {name}:")
        print(f"  Accuracy: {summary['accuracy']*100:.2f}%, Macro F1: {summary['macro_f1']*100:.2f}%, "
              f"ROC-AUC: {summary['macro_roc_auc']:.4f}, Latency: {latency_ms_per_flow:.4f} ms/flow")

    # 3. Export results
    print("\n[3/4] Exporting CNN ablation results...")
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "cnn_representation_comparison.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # 4. Plot Comparison
    print("\n[4/4] Plotting comparison chart...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    metrics_plot = ["Macro F1", "Weighted F1", "Accuracy", "ROC-AUC (OvR)"]
    df_melt = pd.melt(df_results, id_vars=["Representation"], value_vars=metrics_plot, var_name="Metric", value_name="Score")
    
    sns.barplot(data=df_melt, x="Metric", y="Score", hue="Representation", palette="mako", ax=ax)
    ax.set_title("CNN Feature Representation Ablation on Tabular Network Traffic", fontsize=13, pad=12)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(title="Spatial Architecture", frameon=True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "cnn_representation_comparison.png")
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_path}")

    print("\n" + "=" * 80)
    print("CNN FEATURE REPRESENTATION COMPARISON SUMMARY")
    print("=" * 80)
    print(df_results[["Representation", "Input Shape", "Macro F1", "Weighted F1", "Accuracy", "Inference Latency (ms/flow)"]].to_string(index=False))
    print("=" * 80)
    return df_results


if __name__ == "__main__":
    run_cnn_comparison(sample_size=25000, epochs=8, batch_size=512)
