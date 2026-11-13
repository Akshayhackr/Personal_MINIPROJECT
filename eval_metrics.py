"""
CyberScope AI - Comprehensive Evaluation Metrics Engine
=======================================================
Addresses Expectation 6:
Expands beyond accuracy to monitor:
- Precision (Macro, Weighted, Per-Class)
- Recall / Detection Rate (Macro, Weighted, Per-Class)
- F1-Score (Macro, Weighted, Per-Class)
- Confusion Matrix (Counts & Row-Normalized Percentages)
- Per-Class Detection Rates (Sensitivity for every threat vector)
- ROC-AUC One-vs-Rest (Macro, Weighted, and per-class curves)
- False Alarm Rate per 1,000 Normal flows
- Inference Latency (ms per flow)
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", os.path.dirname(os.path.abspath(__file__)))
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc
)


def compute_comprehensive_metrics(y_true, y_pred, y_probs, class_names, normal_label="Normal"):
    """
    Computes all standard cybersecurity and machine learning evaluation metrics.
    
    Parameters:
    -----------
    y_true: 1D array of ground truth integer labels
    y_pred: 1D array of predicted integer labels
    y_probs: 2D array of predicted class probabilities (shape: n_samples, n_classes)
    class_names: list of string names for classes in label encoder order
    normal_label: string name of the benign / normal class
    
    Returns:
    --------
    dict of metrics, dataframe of per-class performance, confusion matrices
    """
    num_classes = len(class_names)
    labels_all = np.arange(num_classes)
    
    # 1. Global Classification Metrics
    acc = float(accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, labels=labels_all, average="macro", zero_division=0))
    weighted_prec = float(precision_score(y_true, y_pred, labels=labels_all, average="weighted", zero_division=0))
    
    macro_rec = float(recall_score(y_true, y_pred, labels=labels_all, average="macro", zero_division=0))
    weighted_rec = float(recall_score(y_true, y_pred, labels=labels_all, average="weighted", zero_division=0))
    
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels_all, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=labels_all, average="weighted", zero_division=0))

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels_all)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = np.nan_to_num(cm.astype(float) / cm.sum(axis=1)[:, np.newaxis])

    # 3. One-vs-Rest ROC-AUC
    # One-hot encode y_true for multi-class ROC calculation
    y_true_onehot = np.zeros((len(y_true), num_classes), dtype=np.float32)
    for i, val in enumerate(y_true):
        if 0 <= val < num_classes:
            y_true_onehot[i, val] = 1.0

    per_class_roc_auc = {}
    valid_classes_for_auc = []
    
    for c_idx, c_name in enumerate(class_names):
        # Only compute if both positive and negative samples exist in y_true
        if 0 < np.sum(y_true_onehot[:, c_idx]) < len(y_true):
            score = float(roc_auc_score(y_true_onehot[:, c_idx], y_probs[:, c_idx]))
            per_class_roc_auc[c_name] = score
            valid_classes_for_auc.append(c_idx)
        else:
            per_class_roc_auc[c_name] = np.nan

    if valid_classes_for_auc:
        macro_roc_auc = float(np.nanmean(list(per_class_roc_auc.values())))
        # Weighted by class support
        supports = np.array([np.sum(y_true == c_idx) for c_idx in valid_classes_for_auc])
        scores = np.array([per_class_roc_auc[class_names[c_idx]] for c_idx in valid_classes_for_auc])
        weighted_roc_auc = float(np.sum(scores * supports) / np.sum(supports))
    else:
        macro_roc_auc = np.nan
        weighted_roc_auc = np.nan

    # 4. Per-Class Detection Rates and Detailed Breakdown
    per_class_rows = []
    for c_idx, c_name in enumerate(class_names):
        support = int(np.sum(y_true == c_idx))
        tp = int(cm[c_idx, c_idx])
        fn = int(np.sum(cm[c_idx, :]) - tp)
        fp = int(np.sum(cm[:, c_idx]) - tp)
        tn = int(len(y_true) - tp - fn - fp)
        
        # Detection rate is Recall / Sensitivity
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0
        
        per_class_rows.append({
            "Class": c_name,
            "Support": support,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": round(prec, 4),
            "Recall (Detection Rate)": round(rec, 4),
            "F1-Score": round(f1, 4),
            "Specificity": round(specificity, 4),
            "ROC-AUC": round(per_class_roc_auc[c_name], 4) if not np.isnan(per_class_roc_auc[c_name]) else "N/A"
        })

    per_class_df = pd.DataFrame(per_class_rows)

    # 5. False Alarms per 1,000 Normal flows
    fpr_per_1000 = 0.0
    if normal_label in class_names:
        norm_idx = list(class_names).index(normal_label)
        norm_support = np.sum(y_true == norm_idx)
        norm_fp = np.sum(cm[norm_idx, :]) - cm[norm_idx, norm_idx]
        if norm_support > 0:
            fpr_per_1000 = float((norm_fp / norm_support) * 1000.0)

    summary = {
        "accuracy": acc,
        "macro_precision": macro_prec,
        "weighted_precision": weighted_prec,
        "macro_recall": macro_rec,
        "weighted_recall": weighted_rec,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_roc_auc": macro_roc_auc,
        "weighted_roc_auc": weighted_roc_auc,
        "false_alarms_per_1000_normal": fpr_per_1000,
        "per_class_roc_auc": per_class_roc_auc
    }

    return summary, per_class_df, cm, cm_norm, y_true_onehot


def plot_confusion_matrix_and_roc(cm, cm_norm, y_true_onehot, y_probs, class_names, output_dir="models", prefix=""):
    """
    Plots and saves high-resolution Confusion Matrix heatmap and One-vs-Rest ROC Curves.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_prefix = f"{prefix}_" if prefix else ""
    
    # 1. Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax
    )
    ax.set_title(f"CyberScope AI - Confusion Matrix ({prefix.replace('_', ' ').title() if prefix else 'Evaluation'})", fontsize=14, pad=12)
    ax.set_ylabel("Ground Truth Class", fontsize=12)
    ax.set_xlabel("Predicted Class", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, f"{file_prefix}confusion_matrix.png")
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)

    # 2. Normalized Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    sns.heatmap(
        cm_norm * 100.0,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax
    )
    ax.set_title(f"CyberScope AI - Normalized Confusion Matrix (%) ({prefix.replace('_', ' ').title() if prefix else 'Evaluation'})", fontsize=14, pad=12)
    ax.set_ylabel("Ground Truth Class", fontsize=12)
    ax.set_xlabel("Predicted Class", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_norm_path = os.path.join(output_dir, f"{file_prefix}confusion_matrix_normalized.png")
    fig.savefig(cm_norm_path, bbox_inches="tight")
    plt.close(fig)

    # 3. One-vs-Rest ROC Curves
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    
    for i, (c_name, color) in enumerate(zip(class_names, colors)):
        if 0 < np.sum(y_true_onehot[:, i]) < len(y_true_onehot):
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{c_name} (AUC = {roc_auc:.4f})")
    
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Detection Rate / Recall)", fontsize=12)
    ax.set_title(f"CyberScope AI - One-vs-Rest ROC Curves", fontsize=14, pad=12)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, f"{file_prefix}roc_curves.png")
    fig.savefig(roc_path, bbox_inches="tight")
    plt.close(fig)

    return cm_path, cm_norm_path, roc_path


def format_evaluation_report(summary, per_class_df, title="CYBERSCOPE AI EVALUATION REPORT"):
    """
    Formats the evaluation results into a clean, human-readable terminal report.
    """
    lines = [
        "=" * 72,
        f" {title} ",
        "=" * 72,
        f"Overall Accuracy               : {summary['accuracy'] * 100:.4f}%",
        f"Macro F1-Score                 : {summary['macro_f1'] * 100:.4f}%",
        f"Weighted F1-Score              : {summary['weighted_f1'] * 100:.4f}%",
        f"Macro Precision                : {summary['macro_precision'] * 100:.4f}%",
        f"Weighted Precision             : {summary['weighted_precision'] * 100:.4f}%",
        f"Macro Recall (Detection Rate)  : {summary['macro_recall'] * 100:.4f}%",
        f"Weighted Recall                : {summary['weighted_recall'] * 100:.4f}%",
        f"Macro ROC-AUC (One-vs-Rest)    : {summary['macro_roc_auc']:.4f}",
        f"Weighted ROC-AUC               : {summary['weighted_roc_auc']:.4f}",
        f"False Alarms / 1000 Normal     : {summary['false_alarms_per_1000_normal']:.2f}",
        "-" * 72,
        "PER-CLASS DETECTION BREAKDOWN",
        "-" * 72,
        per_class_df.to_string(index=False),
        "=" * 72
    ]
    return "\n".join(lines)
