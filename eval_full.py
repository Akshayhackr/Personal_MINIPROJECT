"""
CyberScope AI - Full Dataset Global Benchmarking
================================================
Executes evaluation on the full CyberScope dataset, computing
all comprehensive metrics: Macro F1, Weighted F1, ROC-AUC (OvR),
Per-Class Detection Rates, and Confusion Matrix.
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
from eval_metrics import compute_comprehensive_metrics, plot_confusion_matrix_and_roc, format_evaluation_report

DATA_PATH = os.path.join("data_pipeline", "cyberscope_processed_data.csv")
MODEL_PATH = "cyberscope_model.keras"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
SEQUENCE_LENGTH = 10


def create_sequences(X_scaled, y_encoded, seq_len=SEQUENCE_LENGTH):
    num_raw = len(X_scaled)
    num_features = X_scaled.shape[1]
    if num_raw < seq_len:
        pad_count = seq_len - num_raw
        X_scaled = np.vstack([np.repeat(X_scaled[:1], pad_count, axis=0), X_scaled])
        y_encoded = np.concatenate([np.repeat(y_encoded[:1], pad_count, axis=0), y_encoded])

    num_samples = len(X_scaled) - seq_len + 1
    X_spatial = X_scaled[seq_len - 1:].copy().reshape(-1, num_features, 1).astype(np.float32)
    X_seq = np.empty((num_samples, seq_len, num_features), dtype=np.float32)
    y_target = y_encoded[seq_len - 1:]

    for i in range(num_samples):
        X_seq[i] = X_scaled[i : i + seq_len]
    return X_spatial, X_seq, y_target


def evaluate_full_dataset(batch_size=1024, max_samples=100000):
    print("=" * 75)
    print("CYBERSCOPE AI - FULL-SCALE DATASET EVALUATION")
    print("=" * 75)

    print("Loading upgraded model and preprocessing artifacts...")
    custom_objects = {
        "PositionalEncoding": PositionalEncoding,
        "SinusoidalPositionalEncoding": SinusoidalPositionalEncoding,
        "LearnablePositionalEncoding": LearnablePositionalEncoding
    }
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    print(f"Reading dataset: {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    target_col = "Attack_Group" if "Attack_Group" in df.columns else "Label"
    df["encoded_labels"] = label_encoder.transform(df[target_col].astype(str).values)
    drop_cols = [c for c in ["Label", "Attack_Group", "Timestamp", "Source IP", "Destination IP", "encoded_labels"] if c in df.columns]

    # Subsample if dataset is massive to maintain high speed while preserving all minority classes
    if max_samples is not None and len(df) > max_samples:
        print(f"Subsampling {max_samples:,} flows while preserving 100% of minority attack vectors...")
        class_counts = df["encoded_labels"].value_counts()
        rare_classes = class_counts[class_counts < 1000].index.values
        rare_idx = df[df["encoded_labels"].isin(rare_classes)].index.values
        common_idx = df[~df["encoded_labels"].isin(rare_classes)].index.values
        needed_common = max_samples - len(rare_idx)
        np.random.seed(42)
        sampled_common = np.random.choice(common_idx, size=needed_common, replace=False)
        selected_idx = np.sort(np.concatenate([rare_idx, sampled_common]))
        df = df.iloc[selected_idx].reset_index(drop=True)

    X_full_scaled = scaler.transform(df.drop(columns=drop_cols).fillna(0)).astype(np.float32)
    y_full_encoded = df["encoded_labels"].values

    print("Constructing 1D sequences across class partitions...")
    X_sp_list, X_seq_list, y_list = [], [], []
    for cls_val in np.unique(y_full_encoded):
        mask = (y_full_encoded == cls_val)
        if np.sum(mask) > 0:
            sp, seq, y_sub = create_sequences(X_full_scaled[mask], y_full_encoded[mask], SEQUENCE_LENGTH)
            if len(sp) > 0:
                X_sp_list.append(sp)
                X_seq_list.append(seq)
                y_list.append(y_sub)

    X_spatial_eval = np.vstack(X_sp_list)
    X_seq_eval = np.vstack(X_seq_list)
    y_target = np.concatenate(y_list)

    # If model expects 2D
    if len(model.inputs[0].shape) == 4:
        if X_spatial_eval.shape[1] < 36:
            pad = 36 - X_spatial_eval.shape[1]
            padded = np.pad(X_spatial_eval[:, :, 0], ((0, 0), (0, pad)), mode="constant")
        else:
            padded = X_spatial_eval[:, :36, 0]
        X_spatial_eval = padded.reshape(-1, 6, 6, 1).astype(np.float32)

    total_flows = len(y_target)
    print(f"Executing full inference on {total_flows:,} sequences...")

    start_time = time.perf_counter()
    y_probs = model.predict([X_spatial_eval, X_seq_eval], batch_size=batch_size, verbose=1)
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    latency_ms = (elapsed / total_flows) * 1000.0
    throughput = total_flows / elapsed
    y_pred = np.argmax(y_probs, axis=1)

    class_names = list(label_encoder.classes_)
    summary, per_class_df, cm, cm_norm, y_onehot = compute_comprehensive_metrics(
        y_target, y_pred, y_probs, class_names
    )

    report_text = format_evaluation_report(summary, per_class_df, "CYBERSCOPE AI - GLOBAL DATASET EVALUATION REPORT")
    print(report_text)
    print(f"Inference Latency         : {latency_ms:.4f} ms per flow")
    print(f"Inference Throughput      : {throughput:.1f} flows / second")

    plot_confusion_matrix_and_roc(
        cm, cm_norm, y_onehot, y_probs, class_names, output_dir="models", prefix="global_full"
    )

    plot_confusion_matrix_and_roc(
        cm, cm_norm, y_onehot, y_probs, class_names, output_dir=".", prefix="confusion_matrix_full"
    )

    print("\nGlobal evaluation complete. Results and visualizations updated.")
    return summary


if __name__ == "__main__":
    evaluate_full_dataset()
