"""
CyberScope AI - Unified Data Loading and Sequence Generator
===========================================================
Prepares causal 10-step sequences and 1D spatial flow representations
for the hybrid CyberScope network intrusion detection pipeline.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

DATA_PATH = os.path.join("data_pipeline", "cyberscope_processed_data.csv")
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
SEQUENCE_LENGTH = 10


def create_causal_sequences(X_scaled, y_encoded, seq_len=SEQUENCE_LENGTH):
    """
    Constructs causal 10-step sequence windows and 1D spatial representations.
    
    Each sample i has:
    - X_spatial: shape (num_features, 1) - current packet/flow at time t
    - X_seq: shape (seq_len, num_features) - historical sequence [t - seq_len + 1, ..., t]
    - y_target: integer threat label of the current flow at time t
    """
    num_raw = len(X_scaled)
    num_features = X_scaled.shape[1]
    
    if num_raw < seq_len:
        return (
            np.empty((0, num_features, 1), dtype=np.float32),
            np.empty((0, seq_len, num_features), dtype=np.float32),
            np.empty((0,), dtype=np.int32)
        )
        
    num_samples = num_raw - seq_len + 1
    X_spatial = X_scaled[seq_len - 1:].copy().reshape(-1, num_features, 1).astype(np.float32)
    X_seq = np.empty((num_samples, seq_len, num_features), dtype=np.float32)
    y_target = y_encoded[seq_len - 1:].astype(np.int32)
    
    for i in range(num_samples):
        X_seq[i] = X_scaled[i : i + seq_len]
        
    return X_spatial, X_seq, y_target


def load_cyberscope_dataset(
    data_path=DATA_PATH,
    sample_size=None,
    random_state=42,
    seq_len=SEQUENCE_LENGTH,
    train_ratio=0.70,
    val_ratio=0.10,
    test_ratio=0.20
):
    """
    Loads CyberScope dataset, fits/applies StandardScaler and LabelEncoder,
    and returns train/validation/test partitions for both spatial and temporal branches.
    
    If sample_size is specified, performs stratified sampling while strictly
    preserving 100% of minority class flows (e.g. Infiltration, Botnet).
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
        
    df = pd.read_csv(data_path)
    target_col = "Attack_Group" if "Attack_Group" in df.columns else "Label"
    
    drop_cols = [
        c for c in ["Label", "Attack_Group", "Timestamp", "Source IP", "Destination IP", "encoded_labels"]
        if c in df.columns
    ]
    
    feature_cols = [c for c in df.columns if c not in drop_cols and c != target_col]
    
    label_encoder = LabelEncoder()
    df["encoded_labels"] = label_encoder.fit_transform(df[target_col].astype(str).values)
    num_classes = len(label_encoder.classes_)
    
    # Subsampling if requested (for fast, reproducible ablation benchmarks)
    if sample_size is not None and sample_size < len(df):
        # Always retain all instances of micro-classes (count < 500)
        class_counts = df["encoded_labels"].value_counts()
        rare_classes = class_counts[class_counts < 1000].index.values
        
        rare_indices = df[df["encoded_labels"].isin(rare_classes)].index.values
        common_indices = df[~df["encoded_labels"].isin(rare_classes)].index.values
        
        needed_common = sample_size - len(rare_indices)
        if needed_common > 0 and len(common_indices) > needed_common:
            np.random.seed(random_state)
            sampled_common = np.random.choice(common_indices, size=needed_common, replace=False)
            selected_indices = np.sort(np.concatenate([rare_indices, sampled_common]))
        else:
            selected_indices = np.arange(len(df))
            
        df = df.iloc[selected_indices].reset_index(drop=True)

    # Chronological / Per-Class Partitioning (63% Train, 7% Val, 30% Test or custom)
    train_indices = []
    val_indices = []
    test_indices = []
    
    for cls_val in np.unique(df["encoded_labels"]):
        cls_idx = np.where(df["encoded_labels"] == cls_val)[0]
        n_cls = len(cls_idx)
        
        # When class has very few samples (e.g., Infiltration with 36 samples), ensure all splits get representation
        if n_cls < 10:
            train_indices.extend(cls_idx)
            val_indices.extend(cls_idx)
            test_indices.extend(cls_idx)
        else:
            t_end = int(n_cls * train_ratio)
            v_end = int(n_cls * (train_ratio + val_ratio))
            train_indices.extend(cls_idx[:t_end])
            val_indices.extend(cls_idx[t_end:v_end])
            test_indices.extend(cls_idx[v_end:])
            
    train_indices = np.sort(np.array(train_indices))
    val_indices = np.sort(np.array(val_indices))
    test_indices = np.sort(np.array(test_indices))
    
    df_train = df.iloc[train_indices].copy()
    df_val = df.iloc[val_indices].copy()
    df_test = df.iloc[test_indices].copy()
    
    # Fit StandardScaler strictly on training split
    scaler = StandardScaler()
    X_train_raw = df_train[feature_cols].fillna(0).values.astype(np.float32)
    scaler.fit(X_train_raw)
    
    X_train_scaled = scaler.transform(X_train_raw).astype(np.float32)
    X_val_scaled = scaler.transform(df_val[feature_cols].fillna(0).values).astype(np.float32)
    X_test_scaled = scaler.transform(df_test[feature_cols].fillna(0).values).astype(np.float32)
    
    y_train_lbl = df_train["encoded_labels"].values
    y_val_lbl = df_val["encoded_labels"].values
    y_test_lbl = df_test["encoded_labels"].values
    
    # Generate continuous sequences per partition
    X_sp_train, X_seq_train, y_train = create_causal_sequences(X_train_scaled, y_train_lbl, seq_len)
    X_sp_val, X_seq_val, y_val = create_causal_sequences(X_val_scaled, y_val_lbl, seq_len)
    X_sp_test, X_seq_test, y_test = create_causal_sequences(X_test_scaled, y_test_lbl, seq_len)
    
    dataset_bundle = {
        "X_sp_train": X_sp_train, "X_seq_train": X_seq_train, "y_train": y_train,
        "X_sp_val": X_sp_val, "X_seq_val": X_seq_val, "y_val": y_val,
        "X_sp_test": X_sp_test, "X_seq_test": X_seq_test, "y_test": y_test,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_names": feature_cols,
        "class_names": list(label_encoder.classes_),
        "num_classes": num_classes,
        "num_features": len(feature_cols)
    }
    
    return dataset_bundle
