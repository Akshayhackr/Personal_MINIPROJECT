import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
# [CNN IMPORT]: Imports Keras 'layers' (Conv2D, MaxPooling2D, GlobalAveragePooling2D) and 'Model' used to construct the custom CNN
from tensorflow.keras import layers, Model

DATA_PATH = os.path.join("data_pipeline", "cyberscope_processed_data.csv")
SAVED_MODEL_PATH = "cyberscope_model.keras"
SCALER_PATH = "scaler.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
1
SEQUENCE_LENGTH = 10
IMAGE_DIM = (6, 6)
BATCH_SIZE = 128
EPOCHS = 18

def create_sequences(X_scaled, y_encoded, seq_len):
    """Generates 10-step time-series windows, front-padding micro-classes if N < seq_len"""
    num_raw = len(X_scaled)
    if num_raw == 0:
        return (np.empty((0, IMAGE_DIM[0], IMAGE_DIM[1], 1), dtype=np.float32), 
                np.empty((0, seq_len, X_scaled.shape[1]), dtype=np.float32), 
                np.empty((0,), dtype=int))

    # Front-pad if class has fewer samples than sequence length (preserves micro-classes in test split)
    if num_raw < seq_len:
        pad_count = seq_len - num_raw
        pad_X = np.repeat(X_scaled[:1], pad_count, axis=0)
        X_scaled = np.vstack([pad_X, X_scaled])
        pad_y = np.repeat(y_encoded[:1], pad_count, axis=0)
        y_encoded = np.concatenate([pad_y, y_encoded])

    num_samples = len(X_scaled) - seq_len + 1
    target_num_features = IMAGE_DIM[0] * IMAGE_DIM[1]
    
    if X_scaled.shape[1] < target_num_feat++ures:
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

# [CNN BUILDER FUNCTION]: Function where the CNN + Transformer architecture is defined and initialized
def build_cyberscope_model(image_shape, seq_shape, num_classes):
    # [CNN SPATIAL BRANCH]: Initializes the 2D Convolutional Neural Network layers for the 6x6 input matrix
    img_input = layers.Input(shape=image_shape, name="spatial_input")
    c = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(img_input)
    c = layers.MaxPooling2D((2, 2))(c)
    c = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c)
    c = layers.GlobalAveragePooling2D()(c)

    seq_input = layers.Input(shape=seq_shape, name="temporal_input")
    x = layers.MultiHeadAttention(key_dim=64, num_heads=4, dropout=0.1)(seq_input, seq_input)
    x = layers.Dropout(0.1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + seq_input

    x = layers.Conv1D(filters=128, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv1D(filters=seq_input.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    t = x + res
    t = layers.GlobalAveragePooling1D()(t)

    fused = layers.concatenate([c, t])
    dense = layers.Dense(128, activation="relu")(fused)
    dense = layers.Dropout(0.3)(dense)
    output = layers.Dense(num_classes, activation="softmax", name="threat_classification")(dense)

    # [MODEL INSTANTIATION]: Instantiates the Keras Model object combining the CNN and Transformer branches
    model = Model(inputs=[img_input, seq_input], outputs=output, name="CYBERSCOPE_AI")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),
        loss=tf.keras.losses.CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0),
        metrics=["accuracy"]
    )
    return model

def main():
    print("Loading dataset for Per-Class Chronological Splitting...")
    df = pd.read_csv(DATA_PATH)
    target_col = "Attack_Group" if "Attack_Group" in df.columns else "Label"
    
    label_encoder = LabelEncoder()
    df['encoded_labels'] = label_encoder.fit_transform(df[target_col].astype(str).values)
    num_classes = len(label_encoder.classes_)

    drop_cols = [c for c in ["Label", "Attack_Group", "Timestamp", "Source IP", "Destination IP", "encoded_labels"] if c in df.columns]
    
    # PER-CLASS CHRONOLOGICAL SPLIT (80% first, 20% last for EACH attack vector)
    train_indices = []
    test_indices = []

    for cls_val in np.unique(df['encoded_labels']):
        cls_idx = df[df['encoded_labels'] == cls_val].index.values
        split_pt = int(len(cls_idx) * 0.8)
        train_indices.extend(cls_idx[:split_pt])
        test_indices.extend(cls_idx[split_pt:])

    train_indices = np.sort(train_indices)
    test_indices = np.sort(test_indices)

    df_train = df.loc[train_indices].copy()
    df_test = df.loc[test_indices].copy()

    print(f"Dataset split complete: Train samples = {len(df_train):,}, Test samples = {len(df_test):,}")

    # FIT SCALER STRICTLY ON TRAINING DATA
    print("Fitting scaler strictly on training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df_train.drop(columns=drop_cols).fillna(0)).astype(np.float32)
    X_test_scaled = scaler.transform(df_test.drop(columns=drop_cols).fillna(0)).astype(np.float32)

    # GENERATE SEQUENCES INDEPENDENTLY PER CLASS BLOCK
    print("Generating temporal sequences per class block...")
    X_img_train_list, X_seq_train_list, y_train_list = [], [], []
    X_img_test_list, X_seq_test_list, y_test_list = [], [], []

    for cls_val in np.unique(df['encoded_labels']):
        # Train sequences for class
        mask_tr = (df_train['encoded_labels'].values == cls_val)
        if np.sum(mask_tr) > 0:
            img, seq, y_sub = create_sequences(X_train_scaled[mask_tr], df_train['encoded_labels'].values[mask_tr], SEQUENCE_LENGTH)
            if len(img) > 0:
                X_img_train_list.append(img)
                X_seq_train_list.append(seq)
                y_train_list.append(y_sub)

        # Test sequences for class
        mask_te = (df_test['encoded_labels'].values == cls_val)
        if np.sum(mask_te) > 0:
            img, seq, y_sub = create_sequences(X_test_scaled[mask_te], df_test['encoded_labels'].values[mask_te], SEQUENCE_LENGTH)
            if len(img) > 0:
                X_img_test_list.append(img)
                X_seq_test_list.append(seq)
                y_test_list.append(y_sub)

    X_img_train = np.vstack(X_img_train_list)
    X_seq_train = np.vstack(X_seq_train_list)
    y_target_train = np.concatenate(y_train_list)

    X_img_test = np.vstack(X_img_test_list)
    X_seq_test = np.vstack(X_seq_test_list)
    y_target_test = np.concatenate(y_test_list)

    # 5x OVERSAMPLING FOR INFILTRATION (STRICTLY ON TRAIN)
    infil_idx = label_encoder.transform(["Infiltration"])[0] if "Infiltration" in label_encoder.classes_ else -1
    if infil_idx != -1:
        infil_mask = (y_target_train == infil_idx)
        infil_count = np.sum(infil_mask)
        if infil_count > 0:
            print(f"Oversampling {infil_count} Infiltration training sequences by 5x...")
            X_img_train = np.concatenate([X_img_train, np.repeat(X_img_train[infil_mask], 4, axis=0)])
            X_seq_train = np.concatenate([X_seq_train, np.repeat(X_seq_train[infil_mask], 4, axis=0)])
            y_target_train = np.concatenate([y_target_train, np.repeat(y_target_train[infil_mask], 4, axis=0)])

    y_train = tf.keras.utils.to_categorical(y_target_train, num_classes=num_classes)
    y_test = tf.keras.utils.to_categorical(y_target_test, num_classes=num_classes)

    raw_class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_target_train), y=y_target_train)
    smoothed_weights = np.clip(np.sqrt(raw_class_weights), a_min=0.5, a_max=20.0)
    weights_dict = dict(enumerate(smoothed_weights))

    # [CNN INITIALIZATION CALL]: Calls the builder function to initialize the compiled CNN-Transformer model
    model = build_cyberscope_model(X_img_train.shape[1:], X_seq_train.shape[1:], num_classes)
    
    print("Initiating Leak-Proof Training...")
    model.fit(
        [X_img_train, X_seq_train], y_train,
        validation_data=([X_img_test, X_seq_test], y_test),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        class_weight=weights_dict, verbose=1
    )

    model.save(SAVED_MODEL_PATH)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)
    with open(LABEL_ENCODER_PATH, "wb") as f: pickle.dump(label_encoder, f)
    print("Secure artifacts saved successfully!")

if __name__ == "__main__":
    main()