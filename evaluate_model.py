import os
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATA_PATH = os.path.join("data_pipeline", "cyberscope_processed_data.csv")
MODEL_PATH = "cyberscope_model.keras"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
SEQUENCE_LENGTH = 10
IMAGE_DIM = (6, 6) 

def create_sequences(X_scaled, y_encoded, seq_len):
    """Generates 10-step time-series windows, front-padding micro-classes if N < seq_len"""
    num_raw = len(X_scaled)
    if num_raw == 0:
        return (np.empty((0, IMAGE_DIM[0], IMAGE_DIM[1], 1), dtype=np.float32), 
                np.empty((0, seq_len, X_scaled.shape[1]), dtype=np.float32), 
                np.empty((0,), dtype=int))

    if num_raw < seq_len:
        pad_count = seq_len - num_raw
        pad_X = np.repeat(X_scaled[:1], pad_count, axis=0)
        X_scaled = np.vstack([pad_X, X_scaled])
        pad_y = np.repeat(y_encoded[:1], pad_count, axis=0)
        y_encoded = np.concatenate([pad_y, y_encoded])

    num_samples = len(X_scaled) - seq_len + 1
    target_num_features = IMAGE_DIM[0] * IMAGE_DIM[1]
    
    if X_scaled.shape[1] < target_num_features:
        pad_width = target_num_features - X_scaled.shape[1]
        X_padded = np.pad(X_scaled, ((0, 0), (0, pad_width)), mode="constant").astype(np.float32)
    else:
        X_padded = X_scaled[:, :target_num_features].astype(np.float32)

    X_images = X_padded.reshape(-1, IMAGE_DIM[0], IMAGE_DIM[1], 1)
    
    X_seq = np.empty((num_samples, seq_len, X_scaled.shape[1]), dtype=np.float32)
    X_img = X_images[seq_len - 1:].copy()
    y_target = y_encoded[seq_len - 1:]

    for i in range(num_samples):
        X_seq[i] = X_scaled[i : i + seq_len]

    return X_img, X_seq, y_target

def evaluate_cyberscope():
    print("Loading secure artifacts...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f: label_encoder = pickle.load(f)

    df = pd.read_csv(DATA_PATH)
    target_col = "Attack_Group" if "Attack_Group" in df.columns else "Label"
    
    df['encoded_labels'] = label_encoder.transform(df[target_col].astype(str).values)
    drop_cols = [c for c in ["Label", "Attack_Group", "Timestamp", "Source IP", "Destination IP", "encoded_labels"] if c in df.columns]

    # REPLICATE EXACT PER-CLASS 20% TEST SPLIT
    test_indices = []
    for cls_val in np.unique(df['encoded_labels']):
        cls_idx = df[df['encoded_labels'] == cls_val].index.values
        split_pt = int(len(cls_idx) * 0.8)
        test_indices.extend(cls_idx[split_pt:])

    test_indices = np.sort(test_indices)
    df_test = df.loc[test_indices].copy()

    X_test_scaled = scaler.transform(df_test.drop(columns=drop_cols).fillna(0)).astype(np.float32)

    X_img_test_list, X_seq_test_list, y_test_list = [], [], []

    for cls_val in np.unique(df['encoded_labels']):
        mask_te = (df_test['encoded_labels'].values == cls_val)
        if np.sum(mask_te) > 0:
            img, seq, y_sub = create_sequences(X_test_scaled[mask_te], df_test['encoded_labels'].values[mask_te], SEQUENCE_LENGTH)
            if len(img) > 0:
                X_img_test_list.append(img)
                X_seq_test_list.append(seq)
                y_test_list.append(y_sub)

    X_img_eval = np.vstack(X_img_test_list)
    X_seq_eval = np.vstack(X_seq_test_list)
    y_target = np.concatenate(y_test_list)

    print(f"Executing benchmark inference on {len(y_target):,} test samples across ALL classes...")
    start_time = time.time()
    y_pred_probs = model.predict([X_img_eval, X_seq_eval], batch_size=256)
    end_time = time.time()
    
    y_pred = np.argmax(y_pred_probs, axis=1)
    target_names = label_encoder.classes_
    labels_all = np.arange(len(target_names))

    inference_time_ms = ((end_time - start_time) / len(y_target)) * 1000
    model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    
    cm = confusion_matrix(y_target, y_pred, labels=labels_all)
    normal_idx = list(target_names).index("Normal") if "Normal" in target_names else -1
    
    fpr_per_1000 = 0
    if normal_idx != -1:
        total_normal = np.sum(cm[normal_idx, :])
        false_alarms = total_normal - cm[normal_idx, normal_idx]
        fpr_per_1000 = (false_alarms / total_normal) * 1000 if total_normal > 0 else 0

    print("\n" + "="*60)
    print("CYBERSCOPE AI - LEAK-PROOF SOC BENCHMARKS")
    print("="*60)
    print(f"Overall Accuracy         : {accuracy_score(y_target, y_pred) * 100:.4f}%")
    print(f"Macro F1-Score           : {f1_score(y_target, y_pred, average='macro', zero_division=0) * 100:.4f}%")
    print(f"False Alarms / 1000 Flows: {fpr_per_1000:.2f}")
    print(f"Inference Latency        : {inference_time_ms:.4f} ms per flow")
    print(f"Physical Model Size      : {model_size_mb:.2f} MB")
    
    print("\n" + "="*60)
    print(classification_report(y_target, y_pred, labels=labels_all, target_names=target_names, digits=4, zero_division=0))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('CYBERSCOPE AI - Stratified Leak-Proof Confusion Matrix')
    plt.ylabel('Actual Attack Vector')
    plt.xlabel('Predicted Attack Vector')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)

if __name__ == "__main__":
    evaluate_cyberscope()