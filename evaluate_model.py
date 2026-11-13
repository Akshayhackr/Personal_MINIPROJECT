"""
CyberScope AI - Comprehensive Model Evaluation Tool
===================================================
Addresses Expectation 6:
Performs independent, deep evaluation on the trained CyberScope model
monitoring all critical cybersecurity metrics:
- Precision (Macro & Weighted)
- Recall (Macro & Weighted)
- F1-Score (Macro & Weighted)
- Confusion Matrix (Raw & Normalized)
- Per-Class Detection Rates (Sensitivity for each attack vector)
- ROC-AUC One-vs-Rest (Macro, Weighted, and Per-Class ROC curves)
- Normal-flow False Alarm Rate per 1,000 flows
- Inference Latency and Throughput
"""

import os
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", os.path.dirname(os.path.abspath(__file__)))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from models.positional_encoding import PositionalEncoding, SinusoidalPositionalEncoding, LearnablePositionalEncoding
from data_loader import load_cyberscope_dataset
from eval_metrics import compute_comprehensive_metrics, plot_confusion_matrix_and_roc, format_evaluation_report

MODEL_PATH = "cyberscope_model.keras"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
OUTPUT_DIR = "models"


def main():
    print("=" * 75)
    print("CYBERSCOPE AI - MODEL EVALUATION BENCHMARK")
    print("=" * 75)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file {MODEL_PATH} not found.")

    # 1. Load model with custom layer definitions
    print(f"\nLoading trained model: {MODEL_PATH}...")
    custom_objects = {
        "PositionalEncoding": PositionalEncoding,
        "SinusoidalPositionalEncoding": SinusoidalPositionalEncoding,
        "LearnablePositionalEncoding": LearnablePositionalEncoding
    }
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    print("Model architecture loaded successfully:")
    model.summary()

    # 2. Load evaluation test dataset
    print("\nLoading holdout test dataset...")
    bundle = load_cyberscope_dataset(sample_size=30000, random_state=42)
    X_sp_test = bundle["X_sp_test"]
    X_seq_test = bundle["X_seq_test"]
    y_test = bundle["y_test"]
    class_names = bundle["class_names"]

    # Adapt if model expects 2D grid input (legacy model)
    if len(model.inputs[0].shape) == 4:
        print("Detected 2D spatial input shape. Converting features to 6x6 grid...")
        if X_sp_test.shape[1] < 36:
            pad = 36 - X_sp_test.shape[1]
            padded = np.pad(X_sp_test[:, :, 0], ((0, 0), (0, pad)), mode="constant")
        else:
            padded = X_sp_test[:, :36, 0]
        X_sp_eval = padded.reshape(-1, 6, 6, 1).astype(np.float32)
    else:
        X_sp_eval = X_sp_test

    total_flows = len(y_test)
    print(f"\nExecuting inference on {total_flows:,} holdout sequences...")

    # Warmup
    _ = model.predict([X_sp_eval[:64], X_seq_test[:64]], verbose=0)

    start_time = time.perf_counter()
    y_probs = model.predict([X_sp_eval, X_seq_test], batch_size=512, verbose=1)
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    latency_ms_per_flow = (elapsed / total_flows) * 1000.0
    throughput_flows_sec = total_flows / elapsed

    y_pred = np.argmax(y_probs, axis=1)

    # 3. Compute Comprehensive Metrics
    summary, per_class_df, cm, cm_norm, y_onehot = compute_comprehensive_metrics(
        y_test, y_pred, y_probs, class_names
    )

    # 4. Print Detailed Report
    report_text = format_evaluation_report(summary, per_class_df, "CYBERSCOPE AI - INDEPENDENT EVALUATION REPORT")
    print(report_text)
    print(f"Inference Latency         : {latency_ms_per_flow:.4f} ms per flow")
    print(f"Inference Throughput      : {throughput_flows_sec:.1f} flows / second")

    # Infiltration Specific Callout
    if "Infiltration" in class_names:
        infil_idx = class_names.index("Infiltration")
        infil_row = per_class_df[per_class_df["Class"] == "Infiltration"].iloc[0]
        print("\n" + "-" * 50)
        print("MINORITY ATTACK DETECTION: INFILTRATION")
        print("-" * 50)
        print(f"Support                : {infil_row['Support']}")
        print(f"True Positives (TP)    : {infil_row['TP']}")
        print(f"False Negatives (FN)   : {infil_row['FN']}")
        print(f"Detection Rate (Recall): {infil_row['Recall (Detection Rate)'] * 100:.2f}%")
        print(f"Precision              : {infil_row['Precision'] * 100:.2f}%")
        print(f"F1-Score               : {infil_row['F1-Score'] * 100:.2f}%")

    # 5. Save Plots and CSVs
    cm_path, cm_norm_path, roc_path = plot_confusion_matrix_and_roc(
        cm, cm_norm, y_onehot, y_probs, class_names, output_dir=OUTPUT_DIR, prefix="eval"
    )

    # Also save to root
    cm_root, _, roc_root = plot_confusion_matrix_and_roc(
        cm, cm_norm, y_onehot, y_probs, class_names, output_dir=".", prefix=""
    )

    per_class_df.to_csv(os.path.join(OUTPUT_DIR, "evaluation_per_class_report.csv"), index=False)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(os.path.join(OUTPUT_DIR, "evaluation_confusion_matrix.csv"))

    print("\nVisual Artifacts Saved:")
    print(f"  Confusion Matrix       : {cm_path} & {cm_root}")
    print(f"  Normalized Confusion   : {cm_norm_path}")
    print(f"  One-vs-Rest ROC Curves : {roc_path} & {roc_root}")


if __name__ == "__main__":
    main()
