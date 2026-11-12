# [SYSTEM & SERIALIZATION IMPORTS]: Standard utilities for file paths, inference timing, and loading saved preprocessing artifacts
import os
import time
import pickle

# [DATA & DEEP LEARNING IMPORTS]: NumPy/Pandas for array/dataframe manipulation; TensorFlow for loading the trained CNN-Transformer model
import numpy as np
import pandas as pd
import tensorflow as tf

# [EVALUATION METRICS IMPORTS]: Scikit-Learn functions for multi-class classification reports, confusion matrices, and macro metrics
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# [VISUALIZATION IMPORTS]: Matplotlib and Seaborn for plotting and exporting the confusion matrix heatmap
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress TensorFlow INFO and WARNING logs for cleaner terminal output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# [FILE PATHS & MODEL HYPERPARAMETERS]: Paths to dataset and saved artifacts, plus dual-branch input dimensions (6x6 CNN, 10-step Transformer)
DATA_PATH = os.path.join("data_pipeline", "cyberscope_processed_data.csv")
MODEL_PATH = "cyberscope_model.keras"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
SEQUENCE_LENGTH = 10
IMAGE_DIM = (6, 6)

# [DUAL-BRANCH INPUT FORMATTER]: Prepares 6x6 spatial grids for the CNN branch and 10-step sequences for the Transformer branch
def create_sequences(X_scaled, y_encoded, seq_len):
    """Generates 10-step time-series windows per class block without boundary contamination."""
    num_raw = len(X_scaled)
    if num_raw == 0:
        return (
            np.empty((0, IMAGE_DIM[0], IMAGE_DIM[1], 1), dtype=np.float32),
            np.empty((0, seq_len, X_scaled.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=int)
        )

    # Front-pad rare micro-classes (e.g., Infiltration) if sample count is smaller than the 10-step sequence window
    if num_raw < seq_len:
        pad_count = seq_len - num_raw
        pad_X = np.repeat(X_scaled[:1], pad_count, axis=0)
        X_scaled = np.vstack([pad_X, X_scaled])
        pad_y = np.repeat(y_encoded[:1], pad_count, axis=0)
        y_encoded = np.concatenate([pad_y, y_encoded])

    num_samples = len(X_scaled) - seq_len + 1
    target_num_features = IMAGE_DIM[0] * IMAGE_DIM[1]

    # [CNN SPATIAL RESHAPING]: Zero-pads the 30 features to 36 and reshapes into a 6x6x1 image tensor for the 2D CNN branch
    if X_scaled.shape[1] < target_num_features:
        pad_width = target_num_features - X_scaled.shape[1]
        X_padded = np.pad(X_scaled, ((0, 0), (0, pad_width)), mode="constant").astype(np.float32)
    else:
        X_padded = X_scaled[:, :target_num_features].astype(np.float32)

    X_images = X_padded.reshape(-1, IMAGE_DIM[0], IMAGE_DIM[1], 1)

    # [TRANSFORMER TEMPORAL WINDOWING]: Constructs sliding 10-step sequential windows paired with the corresponding 6x6 CNN spatial frame
    X_seq = np.empty((num_samples, seq_len, X_scaled.shape[1]), dtype=np.float32)
    X_img = X_images[seq_len - 1:].copy()
    y_target = y_encoded[seq_len - 1:]

    for i in range(num_samples):
        X_seq[i] = X_scaled[i : i + seq_len]

    return X_img, X_seq, y_target

# [MAIN EVALUATION PIPELINE]: Loads the trained hybrid model and benchmarks performance across 100% of the dataset
def evaluate_full_dataset():
    # [MODEL & ARTIFACT LOADING]: Loads the compiled CNN-Transformer Keras model along with the training-fitted StandardScaler and LabelEncoder
    print("Loading CYBERSCOPE AI model and preprocessing artifacts...")
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    # [DATASET LOADING & PREPROCESSING]: Reads the full processed CSV, encodes string labels, and scales features using training statistics
    print("Loading 100% of dataset for full-scale evaluation...")
    df = pd.read_csv(DATA_PATH)
    target_col = "Attack_Group" if "Attack_Group" in df.columns else "Label"

    df['encoded_labels'] = label_encoder.transform(df[target_col].astype(str).values)
    drop_cols = [
        c for c in ["Label", "Attack_Group", "Timestamp", "Source IP", "Destination IP", "encoded_labels"]
        if c in df.columns
    ]

    # Transform using DataFrame columns to preserve feature names cleanly
    X_full_scaled = scaler.transform(df.drop(columns=drop_cols).fillna(0)).astype(np.float32)
    y_full_encoded = df['encoded_labels'].values

    # [PER-CLASS SEQUENCE GENERATION]: Builds CNN spatial inputs and Transformer temporal sequences independently for each class block
    print("Constructing temporal sequences across the full dataset...")
    X_img_list, X_seq_list, y_list = [], [], []

    for cls_val in np.unique(y_full_encoded):
        mask = (y_full_encoded == cls_val)
        if np.sum(mask) > 0:
            img, seq, y_sub = create_sequences(X_full_scaled[mask], y_full_encoded[mask], SEQUENCE_LENGTH)
            if len(img) > 0:
                X_img_list.append(img)
                X_seq_list.append(seq)
                y_list.append(y_sub)

    # Stack all per-class spatial tensors, temporal sequences, and target labels into unified evaluation arrays
    X_img_eval = np.vstack(X_img_list)
    X_seq_eval = np.vstack(X_seq_list)
    y_target = np.concatenate(y_list)

    total_flows = len(y_target)
    print(f"Executing full-dataset inference on {total_flows:,} sequences...")

    # [DUAL-BRANCH MODEL INFERENCE]: Feeds both spatial (6x6) and temporal (10-step) inputs into the CNN-Transformer model and measures latency
    start_time = time.time()
    y_pred_probs = model.predict([X_img_eval, X_seq_eval], batch_size=512, verbose=1)
    end_time = time.time()

    # Convert Softmax probability distributions into discrete predicted class indices
    y_pred = np.argmax(y_pred_probs, axis=1)
    target_names = label_encoder.classes_
    labels_all = np.arange(len(target_names))

    # [GLOBAL & SOC METRICS COMPUTATION]: Calculates accuracy, macro scores, latency per flow, disk footprint, and Normal false-positive rates
    acc = accuracy_score(y_target, y_pred)
    macro_prec = precision_score(y_target, y_pred, average='macro', zero_division=0)
    macro_rec = recall_score(y_target, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_target, y_pred, average='macro', zero_division=0)

    inference_time_ms = ((end_time - start_time) / total_flows) * 1000
    model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)

    cm = confusion_matrix(y_target, y_pred, labels=labels_all)
    normal_idx = list(target_names).index("Normal") if "Normal" in target_names else -1

    fpr_per_1000 = 0.0
    normal_fpr_pct = 0.0
    if normal_idx != -1:
        total_normal = np.sum(cm[normal_idx, :])
        false_alarms = total_normal - cm[normal_idx, normal_idx]
        if total_normal > 0:
            normal_fpr_pct = (false_alarms / total_normal) * 100
            fpr_per_1000 = (false_alarms / total_normal) * 1000

    # [REPORTING]: Prints global benchmark summary and per-class precision, recall, and F1-score breakdown
    print("\n" + "=" * 65)
    print("CYBERSCOPE AI - FULL DATASET (100%) GLOBAL METRICS")
    print("=" * 65)
    print(f"Total Evaluated Sequences : {total_flows:,}")
    print(f"Overall Accuracy          : {acc * 100:.4f}%")
    print(f"Macro Precision           : {macro_prec * 100:.4f}%")
    print(f"Macro Recall              : {macro_rec * 100:.4f}%")
    print(f"Macro F1-Score            : {macro_f1 * 100:.4f}%")
    print(f"Normal False-Positive Rate: {normal_fpr_pct:.4f}%")
    print(f"False Alarms / 1000 Flows : {fpr_per_1000:.2f}")
    print(f"Inference Latency         : {inference_time_ms:.4f} ms per flow")
    print(f"Physical Model Size       : {model_size_mb:.2f} MB")

    print("\n" + "=" * 65)
    print("CYBERSCOPE AI - FULL DATASET CLASSIFICATION REPORT")
    print("=" * 65)
    print(classification_report(y_target, y_pred, labels=labels_all, target_names=target_names, digits=4, zero_division=0))

    # [CONFUSION MATRIX VISUALIZATION]: Plots and saves the 8x8 multi-class confusion matrix as a high-resolution PNG
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('CYBERSCOPE AI - Full Dataset Confusion Matrix (100% Flows)')
    plt.ylabel('Actual Attack Vector')
    plt.xlabel('Predicted Attack Vector')
    plt.tight_layout()

    cm_path = "confusion_matrix_full.png"
    plt.savefig(cm_path, dpi=300)
    print(f"\nSaved full-dataset confusion matrix to: {cm_path}")

# Entry point to run the full-dataset evaluation pipeline
if __name__ == "__main__":
    evaluate_full_dataset()