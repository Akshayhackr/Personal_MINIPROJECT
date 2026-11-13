
import os
import json
import pickle
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

from tensorflow.keras import layers, models, callbacks


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "filtered_data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

TARGET = "Attack_Group"

RANDOM_SEED = 42
EPOCHS = 25
BATCH_SIZE = 1024
LEARNING_RATE = 0.001

MODEL_PATH = os.path.join(MODEL_DIR, "cyberscope_mlp.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "mlp_scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "mlp_encoder.pkl")
REPORT_PATH = os.path.join(MODEL_DIR, "mlp_report.csv")
RESULT_PATH = os.path.join(MODEL_DIR, "mlp_results.json")


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

print("=" * 65)
print("CYBERSCOPE AI - MLP BASELINE")
print("=" * 65)


# ============================================================
# LOAD FEATURES
# ============================================================

with open(
    os.path.join(DATA_DIR, "selected_features.json"), "r"
) as f:
    metadata = json.load(f)

FEATURES = metadata["selected_features"]

train_df = pd.read_csv(
    os.path.join(DATA_DIR, "train.csv")
)

val_df = pd.read_csv(
    os.path.join(DATA_DIR, "validation.csv")
)

test_df = pd.read_csv(
    os.path.join(DATA_DIR, "test.csv")
)

for name, frame in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Testing", test_df)
]:
    missing = set(FEATURES + [TARGET]) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


# ============================================================
# FEATURES AND LABELS
# ============================================================

X_train = train_df[FEATURES].to_numpy(dtype=np.float32)
X_val = val_df[FEATURES].to_numpy(dtype=np.float32)
X_test = test_df[FEATURES].to_numpy(dtype=np.float32)

y_train_raw = train_df[TARGET].astype(str).to_numpy()
y_val_raw = val_df[TARGET].astype(str).to_numpy()
y_test_raw = test_df[TARGET].astype(str).to_numpy()

del train_df, val_df, test_df


# ============================================================
# LABEL ENCODING
# ============================================================

encoder = LabelEncoder()

y_train = encoder.fit_transform(y_train_raw)
y_val = encoder.transform(y_val_raw)
y_test = encoder.transform(y_test_raw)

NUM_CLASSES = len(encoder.classes_)
NUM_FEATURES = len(FEATURES)

print("\nClasses:", list(encoder.classes_))


# ============================================================
# SCALING
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_ids = np.unique(y_train)

balanced = compute_class_weight(
    class_weight="balanced",
    classes=class_ids,
    y=y_train
)

weights = np.sqrt(balanced)
weights = weights / np.mean(weights)
weights = np.clip(weights, 0, 10)

class_weights = {
    int(c): float(w)
    for c, w in zip(class_ids, weights)
}

print("\nClass weights:", class_weights)


# ============================================================
# MLP MODEL
# ============================================================

model = models.Sequential([
    layers.Input(shape=(NUM_FEATURES,)),

    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.35),

    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.30),

    layers.Dense(64, activation="relu"),
    layers.Dropout(0.20),

    layers.Dense(NUM_CLASSES, activation="softmax")
], name="CYBERSCOPE_MLP")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

callback_list = [
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),

    callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]


# ============================================================
# TRAINING
# ============================================================

print("\nSTARTING MLP TRAINING")

start = time.perf_counter()

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callback_list,
    shuffle=True,
    verbose=1
)

training_seconds = time.perf_counter() - start

model = tf.keras.models.load_model(MODEL_PATH)


# ============================================================
# SAVE PREPROCESSING
# ============================================================

with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

with open(ENCODER_PATH, "wb") as f:
    pickle.dump(encoder, f)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\nFINAL MLP TEST EVALUATION")

probabilities = model.predict(
    X_test,
    batch_size=4096,
    verbose=1
)

predictions = np.argmax(probabilities, axis=1)

accuracy = accuracy_score(y_test, predictions)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

report = classification_report(
    y_test,
    predictions,
    labels=np.arange(NUM_CLASSES),
    target_names=encoder.classes_,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report).transpose().to_csv(REPORT_PATH)

print("\n", classification_report(
    y_test,
    predictions,
    labels=np.arange(NUM_CLASSES),
    target_names=encoder.classes_,
    zero_division=0
))


# ============================================================
# RESULTS
# ============================================================

results = {
    "model": "MLP",
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "training_seconds": float(training_seconds),
    "epochs_completed": len(history.history["loss"]),
    "classes": encoder.classes_.tolist()
}

with open(RESULT_PATH, "w") as f:
    json.dump(results, f, indent=4)

print("\n" + "=" * 65)
print("MLP BASELINE COMPLETED")
print("=" * 65)

print("Accuracy:", round(accuracy, 6))
print("Macro F1:", round(macro_f1, 6))
print("Weighted F1:", round(weighted_f1, 6))
print("Training time:", round(training_seconds, 2), "seconds")

print("\nSaved:", MODEL_PATH)
print("Saved:", REPORT_PATH)
print("Saved:", RESULT_PATH)
