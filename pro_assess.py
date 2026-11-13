import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, Model
from tqdm.keras import TqdmCallback

DATA_PATH = os.path.join("data_pipeline", "cyberscope_processed_data.csv")
SAVED_MODEL_PATH = "cyberscope_model.keras"
SCALER_PATH = "scaler.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

SEQUENCE_LENGTH = 10
BATCH_SIZE = 128
EPOCHS = 18

def create_continuous_sequences(X_scaled, y_encoded, seq_len):
    """Generates continuous causal sequences globally, preserving true network time."""
    num_raw = len(X_scaled)
    num_features = X_scaled.shape[1]
    
    if num_raw < seq_len:
        return (np.empty((0, num_features, 1), dtype=np.float32), 
                np.empty((0, seq_len, num_features), dtype=np.float32), 
                np.empty((0,), dtype=int))

    num_samples = num_raw - seq_len + 1
    
    # 1D Spatial Representation (Current Packet)
    X_spatial = X_scaled[seq_len - 1:].copy().reshape(-1, num_features, 1).astype(np.float32)
    
    # Pre-allocate temporal sequence array
    X_seq = np.empty((num_samples, seq_len, num_features), dtype=np.float32)
    y_target = y_encoded[seq_len - 1:]

    for i in range(num_samples):
        X_seq[i] = X_scaled[i : i + seq_len]

    return X_spatial, X_seq, y_target

def build_realtime_cyberscope(spatial_shape, seq_shape, num_classes):
    # Branch 1: 1D-CNN for spatial tabular feature extraction
    spatial_input = layers.Input(shape=spatial_shape, name="spatial_input")
    c = layers.Conv1D(32, kernel_size=3, activation="relu", padding="same")(spatial_input)
    c = layers.MaxPooling1D(pool_size=2)(c)
    c = layers.Conv1D(64, kernel_size=3, activation="relu", padding="same")(c)
    c = layers.GlobalAveragePooling1D()(c)

    # Branch 2: UNIDIRECTIONAL GRU (Preserves Real-Time Causality)
    seq_input = layers.Input(shape=seq_shape, name="temporal_input")
    t = layers.GRU(64, return_sequences=True)(seq_input)
    t = layers.GRU(64)(t)

    # Fusion & Classification
    fused = layers.concatenate([c, t])
    dense = layers.Dense(128, activation="relu")(fused)
    dense = layers.Dropout(0.3)(dense)
    output = layers.Dense(num_classes, activation="softmax", name="threat_classification")(dense)

    model = Model(inputs=[spatial_input, seq_input], outputs=output, name="CYBERSCOPE_RealTime")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),
        loss=tf.keras.losses.CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0),
        metrics=["accuracy"]
    )
    return model

def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    target_col = "Attack_Group" if "Attack_Group" in df.columns else "Label"

    label_encoder = LabelEncoder()
    df['encoded_labels'] = label_encoder.fit_transform(df[target_col].astype(str).values)
    num_classes = len(label_encoder.classes_)
    drop_cols = [c for c in ["Label", "Attack_Group", "Timestamp", "Source IP", "Destination IP", "encoded_labels"] if c in df.columns]

    # 1. Fit Scaler STRICTLY on the first 63% of the raw chronological data
    train_raw_end = int(len(df) * 0.63)
    scaler = StandardScaler()
    scaler.fit(df.iloc[:train_raw_end].drop(columns=drop_cols).fillna(0))
    
    # 2. Scale the global dataset
    X_global_scaled = scaler.transform(df.drop(columns=drop_cols).fillna(0)).astype(np.float32)
    y_global_encoded = df['encoded_labels'].values

    # 3. Generate global continuous sequences (True packet sequences)
    print("Generating strictly continuous network sequences...")
    X_sp_global, X_seq_global, y_tgt_global = create_continuous_sequences(X_global_scaled, y_global_encoded, SEQUENCE_LENGTH)

    # 4. Per-Class Chronological Split on Sequences
    print("Executing Per-Class Sequence Splitting (63/7/30)...")
    train_idx, val_idx, test_idx = [], [], []
    
    for cls_val in np.unique(y_tgt_global):
        cls_indices = np.where(y_tgt_global == cls_val)[0]
        n_samples = len(cls_indices)
        
        t_end = int(n_samples * 0.63)
        v_end = int(n_samples * 0.70)
        
        train_idx.extend(cls_indices[:t_end])
        val_idx.extend(cls_indices[t_end:v_end])
        test_idx.extend(cls_indices[v_end:])

    # Sort to maintain chronological order within sets
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)

    X_sp_train, X_seq_train, y_train_lbl = X_sp_global[train_idx], X_seq_global[train_idx], y_tgt_global[train_idx]
    X_sp_val, X_seq_val, y_val_lbl = X_sp_global[val_idx], X_seq_global[val_idx], y_tgt_global[val_idx]

    print(f"Dataset split complete: Train Sequences = {len(train_idx):,}, Val Sequences = {len(val_idx):,}")

    # Calculate dampened class weights (SMOTE removed entirely)
    unique_classes = np.unique(y_train_lbl)
    raw_class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train_lbl)
    smoothed_weights = np.clip(np.sqrt(raw_class_weights), a_min=0.5, a_max=20.0)
    weights_dict = {cls: weight for cls, weight in zip(unique_classes, smoothed_weights)}

    y_train_cat = tf.keras.utils.to_categorical(y_train_lbl, num_classes=num_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val_lbl, num_classes=num_classes)

    # Build and Train
    model = build_realtime_cyberscope(X_sp_train.shape[1:], X_seq_train.shape[1:], num_classes)
    
    print("Initiating Strict Causal Training (1D-CNN + Uni-GRU)...")
    model.fit(
        [X_sp_train, X_seq_train], y_train_cat,
        validation_data=([X_sp_val, X_seq_val], y_val_cat),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        class_weight=weights_dict, verbose=0, 
        callbacks=[TqdmCallback(verbose=1)]
    )

    model.save(SAVED_MODEL_PATH)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)
    with open(LABEL_ENCODER_PATH, "wb") as f: pickle.dump(label_encoder, f)
    print("Methodologically secure artifacts saved successfully!")

if __name__ == "__main__":
    main()