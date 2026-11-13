
import hashlib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

SEQUENCE_LENGTH = 10
TRAIN_RATIO = 0.63
VALIDATION_END_RATIO = 0.70


def get_file_hash(path):
    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def load_and_prepare_dataframe(path, feature_cols=None):
    df = pd.read_csv(path)

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"],
            format="mixed",
            errors="coerce"
        )

        if df["Timestamp"].isna().any():
            raise ValueError(
                "Timestamp contains invalid or missing values. "
                "Fix them before chronological splitting."
            )

        df = df.sort_values(
            "Timestamp",
            kind="stable"
        ).reset_index(drop=True)

    else:
        print(
            "WARNING: No Timestamp column found. "
            "Assuming CSV row order is chronological."
        )

    target_col = (
        "Attack_Group"
        if "Attack_Group" in df.columns
        else "Label"
    )

    if target_col not in df.columns:
        raise ValueError("No valid target column found.")

    drop_cols = [
    "Label",
    "Attack_Group",
    "Timestamp",
    "Source IP",
    "Destination IP"
]

    if feature_cols is None:
        feature_cols = [
            col for col in df.columns
            if col not in drop_cols
        ]
    else:
        missing = [
            col for col in feature_cols
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing feature columns: {missing}"
            )

    X_df = df[feature_cols].copy()

    for col in feature_cols:
        X_df[col] = pd.to_numeric(
            X_df[col],
            errors="raise"
        )

    X_df = X_df.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    labels = df[target_col].astype(str).values

    return df, X_df, labels, target_col, feature_cols


def chronological_boundaries(total_rows):
    train_end = int(total_rows * TRAIN_RATIO)
    validation_end = int(
        total_rows * VALIDATION_END_RATIO
    )

    return train_end, validation_end


def create_partition_sequences(
    X_scaled,
    y_encoded,
    start,
    end,
    seq_len=SEQUENCE_LENGTH
):
    """
    Each target belongs exclusively to [start, end).

    Earlier rows may be used as historical context.
    No future rows are used.

    The label belongs to the final flow in each window.
    """

    first_target = max(start, seq_len - 1)

    target_indices = np.arange(
        first_target,
        end
    )

    num_features = X_scaled.shape[1]
    num_samples = len(target_indices)

    X_seq = np.empty(
        (num_samples, seq_len, num_features),
        dtype=np.float32
    )

    X_spatial = np.empty(
        (num_samples, num_features, 1),
        dtype=np.float32
    )

    y_target = y_encoded[target_indices]

    for j, target_idx in enumerate(target_indices):
        window_start = target_idx - seq_len + 1

        X_seq[j] = X_scaled[
            window_start:target_idx + 1
        ]

        X_spatial[j, :, 0] = X_scaled[target_idx]

    return X_spatial, X_seq, y_target, target_indices


def print_class_distribution(name, labels, encoder):
    print(f"\n{name} CLASS DISTRIBUTION")
    print("-" * 45)

    counts = np.bincount(
        labels,
        minlength=len(encoder.classes_)
    )

    for class_idx, class_name in enumerate(encoder.classes_):
        print(
            f"{class_name:<25} {counts[class_idx]:>10,}"
        )
