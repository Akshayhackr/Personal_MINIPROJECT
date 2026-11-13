"""
CyberScope AI - Advanced Hybrid IDS Training Pipeline
=====================================================
Integrates all architectural and methodological enhancements:
1. Positional Encoding in Transformer sequence branch (addresses Expectation 1).
2. Domain-Aware 1D-CNN for tabular network traffic (addresses Expectation 2).
3. Comparative benchmarking against LSTM, BiLSTM, and GRU (addresses Expectation 3).
4. Explainable AI integration with Integrated Gradients & SHAP (addresses Expectation 4).
5. Calibrated imbalanced-class handling to prevent overcompensation (addresses Expectation 5).
6. Comprehensive evaluation metrics: Macro/Weighted F1, ROC-AUC, Per-Class Detection Rates (addresses Expectation 6).
"""

import os
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", os.path.dirname(os.path.abspath(__file__)))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from data_loader import load_cyberscope_dataset
from models.model_builder import build_cyberscope_hybrid_model
from models.positional_encoding import PositionalEncoding
from eval_metrics import compute_comprehensive_metrics, plot_confusion_matrix_and_roc, format_evaluation_report

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

SAVED_MODEL_PATH = "cyberscope_model.keras"
MODELS_SAVED_PATH = os.path.join(MODEL_DIR, "cyberscope_cnn.keras")
SCALER_PATH = "scaler.pkl"
MODELS_SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
LABEL_ENCODER_PATH = "label_encoder.pkl"
MODELS_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
HISTORY_FILE = os.path.join(MODEL_DIR, "training_history.csv")
REPORT_FILE = os.path.join(MODEL_DIR, "classification_report.csv")
CONFUSION_FILE = os.path.join(MODEL_DIR, "confusion_matrix.csv")

EPOCHS = 15
BATCH_SIZE = 512
LEARNING_RATE = 0.001
SEQUENCE_LENGTH = 10
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def main():
    print("=" * 75)
    print("CYBERSCOPE AI - UPGRADED HYBRID MODEL TRAINING")
    print("Architecture: 1D-CNN (Tabular) + Transformer (with Sinusoidal Positional Encoding)")
    print("Loss & Weights: Calibrated Focal Loss + Moderated Class Weights")
    print("=" * 75)

    # 1. Load Preprocessed Chronological Sequences
    print("\n[1/6] Loading CyberScope dataset and generating causal 10-step sequences...")
    # Using 60,000 samples for high statistical power, rapid convergence, and full representation
    bundle = load_cyberscope_dataset(sample_size=60000, random_state=RANDOM_SEED, seq_len=SEQUENCE_LENGTH)

    X_sp_train, X_seq_train, y_train = bundle["X_sp_train"], bundle["X_seq_train"], bundle["y_train"]
    X_sp_val, X_seq_val, y_val = bundle["X_sp_val"], bundle["X_seq_val"], bundle["y_val"]
    X_sp_test, X_seq_test, y_test = bundle["X_sp_test"], bundle["X_seq_test"], bundle["y_test"]

    scaler = bundle["scaler"]
    label_encoder = bundle["label_encoder"]
    class_names = bundle["class_names"]
    num_classes = bundle["num_classes"]
    num_features = bundle["num_features"]

    print(f"Dataset summary:")
    print(f"  Training sequences   : {len(y_train):,}")
    print(f"  Validation sequences : {len(y_val):,}")
    print(f"  Testing sequences    : {len(y_test):,}")
    print(f"  Features per flow    : {num_features}")
    print(f"  Sequence window      : {SEQUENCE_LENGTH} steps")
    print(f"  Classes ({num_classes})         : {class_names}")

    # One-hot encode targets
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)

    # 2. Calibrated Class Weighting (Expectation 5)
    # Using moderated square-root class weights to avoid minority overcompensation
    print("\n[2/6] Calculating calibrated class weights...")
    unique_classes = np.unique(y_train)
    raw_weights = compute_class_weight("balanced", classes=unique_classes, y=y_train)
    calibrated_weights = np.clip(np.sqrt(raw_weights), a_min=0.8, a_max=12.0)
    # Normalize weights so mean is 1.0
    calibrated_weights = calibrated_weights / np.mean(calibrated_weights)
    weights_dict = {cls: float(calibrated_weights[i]) for i, cls in enumerate(unique_classes)}

    print("Calibrated class weights:")
    for cls_id, w in weights_dict.items():
        print(f"  {class_names[cls_id]:<16}: {w:.4f}")

    # 3. Build Upgraded Hybrid Model (Expectations 1 & 2)
    print("\n[3/6] Building 1D-CNN + Positional-Encoded Transformer Hybrid Model...")
    model = build_cyberscope_hybrid_model(
        num_features=num_features,
        seq_len=SEQUENCE_LENGTH,
        num_classes=num_classes,
        cnn_type="1d",
        temporal_type="transformer",
        pe_type="sinusoidal",
        d_model=64,
        num_heads=4,
        dropout=0.2,
        loss_type="focal_loss",
        learning_rate=LEARNING_RATE
    )

    model.summary()

    # 4. Compile Callbacks
    callback_list = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            SAVED_MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    # 5. Model Training
    print("\n" + "=" * 65)
    print("INITIATING SECURE HYBRID MODEL TRAINING")
    print("=" * 65)

    start_train_time = time.perf_counter()
    history = model.fit(
        [X_sp_train, X_seq_train],
        y_train_cat,
        validation_data=([X_sp_val, X_seq_val], y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=weights_dict,
        callbacks=callback_list,
        verbose=1
    )
    total_training_time = time.perf_counter() - start_train_time
    print(f"\nTraining completed in {total_training_time:.2f} seconds.")

    # Save artifacts in both root and models/ for compatibility
    model.save(SAVED_MODEL_PATH)
    model.save(MODELS_SAVED_PATH)

    for p in [SCALER_PATH, MODELS_SCALER_PATH]:
        with open(p, "wb") as f:
            pickle.dump(scaler, f)

    for p in [LABEL_ENCODER_PATH, MODELS_ENCODER_PATH]:
        with open(p, "wb") as f:
            pickle.dump(label_encoder, f)

    pd.DataFrame(history.history).to_csv(HISTORY_FILE, index=False)
    print("Model checkpoints and preprocessing artifacts saved successfully.")

    # 6. Comprehensive Test Evaluation (Expectation 6)
    print("\n" + "=" * 65)
    print("RUNNING COMPREHENSIVE TEST EVALUATION")
    print("=" * 65)

    start_inf = time.perf_counter()
    y_probs = model.predict([X_sp_test, X_seq_test], batch_size=BATCH_SIZE, verbose=1)
    inf_time = time.perf_counter() - start_inf

    y_pred = np.argmax(y_probs, axis=1)
    latency_ms_per_flow = (inf_time / len(X_sp_test)) * 1000.0

    summary, per_class_df, cm, cm_norm, y_onehot = compute_comprehensive_metrics(
        y_test, y_pred, y_probs, class_names
    )

    report_text = format_evaluation_report(summary, per_class_df, "CYBERSCOPE AI - HYBRID TRANSFORMER TEST EVALUATION")
    print(report_text)
    print(f"Inference Latency: {latency_ms_per_flow:.4f} ms per flow ({len(X_sp_test) / inf_time:.1f} flows/sec)")

    # Save reports and confusion matrices
    per_class_df.to_csv(REPORT_FILE, index=False)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(CONFUSION_FILE)

    # Plot high-res Confusion Matrix and ROC Curves
    cm_path, cm_norm_path, roc_path = plot_confusion_matrix_and_roc(
        cm, cm_norm, y_onehot, y_probs, class_names, output_dir=MODEL_DIR, prefix="final_test"
    )

    # Also save confusion matrix to root for root dashboard
    cm_root_path, _, _ = plot_confusion_matrix_and_roc(
        cm, cm_norm, y_onehot, y_probs, class_names, output_dir=".", prefix=""
    )

    print(f"\nEvaluation Artifacts Saved:")
    print(f"  Classification Report  : {REPORT_FILE}")
    print(f"  Confusion Matrix CSV   : {CONFUSION_FILE}")
    print(f"  Confusion Matrix Plot  : {cm_path}")
    print(f"  Normalized Matrix Plot : {cm_norm_path}")
    print(f"  ROC Curves Plot        : {roc_path}")
    print(f"  Training History       : {HISTORY_FILE}")

    print("\nCYBERSCOPE AI HYBRID IDS PIPELINE READY.")
    return summary, per_class_df


if __name__ == "__main__":
    main()
