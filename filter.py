
import os
import glob
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "filtered_data")

os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.csv")
VALIDATION_FILE = os.path.join(OUTPUT_DIR, "validation.csv")
TEST_FILE = os.path.join(OUTPUT_DIR, "test.csv")

FEATURES_FILE = os.path.join(OUTPUT_DIR, "selected_features.json")
MI_FILE = os.path.join(OUTPUT_DIR, "feature_importance.csv")
CORRELATION_FILE = os.path.join(
    OUTPUT_DIR, "correlated_features_removed.csv"
)

TARGET = "Attack_Group"

TEST_SIZE = 0.30
VALIDATION_SIZE = 0.10

TOP_K_FEATURES = 30
CORRELATION_THRESHOLD = 0.95

RANDOM_STATE = 42

MI_SAMPLES_PER_CLASS = 20000
CORRELATION_SAMPLE_SIZE = 200000


# ============================================================
# ATTACK GROUP MAPPING
# ============================================================

def normalize_label(label):
    return (
        str(label)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def map_attack_group(label):

    label = normalize_label(label)

    if label == "benign":
        return "Normal"

    if "infiltration" in label:
        return "Infiltration"

    if "bot" in label:
        return "Bot"

    if "portscan" in label or "port scan" in label:
        return "PortScan"

    if "ddos" in label:
        return "DDoS"

    if "dos" in label:
        return "DoS"

    if "web attack" in label:
        return "Web Attack"

    if "patator" in label or "brute force" in label:
        return "Brute Force"

    if "heartbleed" in label:
        return "Heartbleed"

    raise ValueError(f"Unmapped CIC label: {label}")


# ============================================================
# LOAD CIC-IDS2017 FILES
# ============================================================

print("=" * 70)
print("CYBERSCOPE AI - STRATIFIED FEATURE SELECTION")
print("=" * 70)

patterns = [
    "Monday-WorkingHours*.csv",
    "Tuesday-WorkingHours*.csv",
    "Wednesday-workingHours*.csv",
    "Thursday-WorkingHours*.csv",
    "Friday-WorkingHours*.csv"
]

csv_files = []

for pattern in patterns:
    csv_files.extend(
        glob.glob(os.path.join(PROJECT_DIR, pattern))
    )

csv_files = sorted(set(csv_files))

if not csv_files:
    raise FileNotFoundError("No CIC-IDS2017 CSV files found.")

print(f"\nFound {len(csv_files)} CIC-IDS2017 files.")

frames = []

for file_path in csv_files:

    print("Loading:", os.path.basename(file_path))

    temp = pd.read_csv(
        file_path,
        low_memory=False
    )

    temp.columns = (
        temp.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    temp = temp.loc[:, ~temp.columns.duplicated()]

    if "Label" not in temp.columns:
        raise ValueError(
            f"Label column missing in {os.path.basename(file_path)}"
        )

    temp["Label"] = temp["Label"].astype(str).str.strip()

    temp[TARGET] = temp["Label"].apply(map_attack_group)

    frames.append(temp)

df = pd.concat(
    frames,
    ignore_index=True,
    sort=False
)

del frames

print("\nCombined dataset shape:", df.shape)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\nFULL DATASET CLASS DISTRIBUTION")

print(df[TARGET].value_counts().to_string())


# ============================================================
# FEATURE FILTRATION
# ============================================================

DROP_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Label",
    TARGET
]

X = df.drop(
    columns=DROP_COLUMNS,
    errors="ignore"
).copy()

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

X.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

# Use float32 to reduce memory consumption
X = X.astype(np.float32)

invalid_columns = X.columns[
    X.isna().all()
].tolist()

X.drop(
    columns=invalid_columns,
    inplace=True
)

if X.shape[1] == 0:
    raise ValueError("No valid numeric features found.")

y = df[TARGET].copy()

print("\nDATA FILTRATION")
print("Invalid columns removed:", len(invalid_columns))
print("Rows:", len(X))
print("Numeric features:", X.shape[1])


# ============================================================
# STRATIFIED 70:30 TRAIN / TEST SPLIT
# ============================================================

indices = np.arange(len(X))

train_indices, test_indices = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# Split 10% of the training portion for validation
train_fit_indices, validation_indices = train_test_split(
    train_indices,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y.iloc[train_indices]
)

# Preserve original row order within each partition
train_fit_indices = np.sort(train_fit_indices)
validation_indices = np.sort(validation_indices)
test_indices = np.sort(test_indices)

X_train = X.iloc[train_fit_indices].copy()
X_validation = X.iloc[validation_indices].copy()
X_test = X.iloc[test_indices].copy()

y_train = y.iloc[train_fit_indices].copy()
y_validation = y.iloc[validation_indices].copy()
y_test = y.iloc[test_indices].copy()

print("\nSTRATIFIED DATA SPLIT")
print("Training:", len(X_train), f"({len(X_train)/len(X):.2%})")
print("Validation:", len(X_validation),
      f"({len(X_validation)/len(X):.2%})")
print("Testing:", len(X_test), f"({len(X_test)/len(X):.2%})")


# ============================================================
# VERIFY CLASS DISTRIBUTION
# ============================================================

print("\nCLASS DISTRIBUTION BY PARTITION")

for name, labels in [
    ("TRAINING", y_train),
    ("VALIDATION", y_validation),
    ("TESTING", y_test)
]:

    print(f"\n{name}")
    print(labels.value_counts().to_string())

all_classes = set(y.unique())

for name, labels in [
    ("Training", y_train),
    ("Validation", y_validation),
    ("Testing", y_test)
]:

    missing = all_classes - set(labels.unique())

    if missing:
        raise ValueError(
            f"{name} is missing classes: {sorted(missing)}"
        )


# ============================================================
# TRAIN-ONLY MEDIAN IMPUTATION
# ============================================================

medians = X_train.median()

unusable_features = medians[
    medians.isna()
].index.tolist()

if unusable_features:

    X_train.drop(columns=unusable_features, inplace=True)
    X_validation.drop(columns=unusable_features, inplace=True)
    X_test.drop(columns=unusable_features, inplace=True)

    medians.drop(
        index=unusable_features,
        inplace=True
    )

X_train = X_train.fillna(medians).fillna(0)
X_validation = X_validation.fillna(medians).fillna(0)
X_test = X_test.fillna(medians).fillna(0)

print("\nTRAIN-ONLY IMPUTATION COMPLETED")


# ============================================================
# REMOVE CONSTANT FEATURES
# ============================================================

variances = X_train.var()

variance_features = variances[
    variances > 0
].index.tolist()

X_train = X_train[variance_features].copy()
X_validation = X_validation[variance_features].copy()
X_test = X_test[variance_features].copy()

if not variance_features:
    raise ValueError("No features remain after variance filtering.")

print("Features after variance filtering:", len(variance_features))


# ============================================================
# MUTUAL INFORMATION FEATURE RANKING
# ============================================================

print("\nCALCULATING MUTUAL INFORMATION...")

sampled_parts = []

for class_name in y_train.unique():

    class_indices = y_train[
        y_train == class_name
    ].index

    sample_count = min(
        len(class_indices),
        MI_SAMPLES_PER_CLASS
    )

    sampled_indices = pd.Series(class_indices).sample(
        n=sample_count,
        random_state=RANDOM_STATE
    ).values

    sampled_parts.append(sampled_indices)

mi_indices = np.concatenate(sampled_parts)

X_mi = X_train.loc[mi_indices]
y_mi = y_train.loc[mi_indices]

# Encode class labels
y_mi_encoded, class_names = pd.factorize(y_mi)

print("MI samples:", len(X_mi))
print("MI classes:", len(class_names))

mi_scores = mutual_info_classif(
    X_mi,
    y_mi_encoded,
    random_state=RANDOM_STATE,
    n_neighbors=3
)

mi_results = pd.DataFrame({
    "Feature": X_train.columns,
    "MI_Score": mi_scores
}).sort_values(
    "MI_Score",
    ascending=False
).reset_index(drop=True)

mi_lookup = dict(
    zip(
        mi_results["Feature"],
        mi_results["MI_Score"]
    )
)

print("Mutual information completed.")


# ============================================================
# CORRELATION FILTERING
# ============================================================

print("\nCALCULATING CORRELATION MATRIX...")

correlation_count = min(
    len(X_train),
    CORRELATION_SAMPLE_SIZE
)

X_corr = X_train.sample(
    n=correlation_count,
    random_state=RANDOM_STATE
)

correlation_matrix = X_corr.corr(
    method="pearson"
).abs()

ranked_features = mi_results["Feature"].tolist()

retained_features = []
correlation_records = []

for feature in ranked_features:

    remove_feature = False

    for retained in retained_features:

        corr_value = correlation_matrix.loc[
            feature,
            retained
        ]

        if corr_value > CORRELATION_THRESHOLD:

            correlation_records.append({
                "Removed_Feature": feature,
                "Retained_Feature": retained,
                "Absolute_Correlation": float(corr_value),
                "Removed_MI_Score": float(mi_lookup[feature]),
                "Retained_MI_Score": float(mi_lookup[retained])
            })

            remove_feature = True
            break

    if not remove_feature:
        retained_features.append(feature)


correlation_df = pd.DataFrame(
    correlation_records,
    columns=[
        "Removed_Feature",
        "Retained_Feature",
        "Absolute_Correlation",
        "Removed_MI_Score",
        "Retained_MI_Score"
    ]
)

correlation_df.to_csv(
    CORRELATION_FILE,
    index=False
)

print("Correlation threshold:", CORRELATION_THRESHOLD)
print("Before correlation filtering:", len(ranked_features))
print("After correlation filtering:", len(retained_features))


# ============================================================
# TOP 30 FEATURE SELECTION
# ============================================================

selected_features = retained_features[:TOP_K_FEATURES]

if len(selected_features) < TOP_K_FEATURES:
    warnings.warn(
        f"Only {len(selected_features)} features survived filtering."
    )

if not selected_features:
    raise ValueError("No features selected.")

X_train = X_train[selected_features]
X_validation = X_validation[selected_features]
X_test = X_test[selected_features]

print("\nSELECTED FEATURES")

for i, feature in enumerate(selected_features, 1):
    print(
        f"{i:02d}. {feature} "
        f"(MI: {mi_lookup[feature]:.6f})"
    )


# ============================================================
# SAVE TRAINING, VALIDATION AND TEST DATA
# ============================================================

train_output = X_train.copy()
train_output[TARGET] = y_train.values

validation_output = X_validation.copy()
validation_output[TARGET] = y_validation.values

test_output = X_test.copy()
test_output[TARGET] = y_test.values

train_output.to_csv(TRAIN_FILE, index=False)
validation_output.to_csv(VALIDATION_FILE, index=False)
test_output.to_csv(TEST_FILE, index=False)


# ============================================================
# SAVE FEATURE METADATA
# ============================================================

metadata = {
    "target_column": TARGET,
    "selected_features": selected_features,
    "top_k_features": TOP_K_FEATURES,
    "correlation_threshold": CORRELATION_THRESHOLD,
    "random_state": RANDOM_STATE,
    "total_rows": len(X),
    "training_rows": len(X_train),
    "validation_rows": len(X_validation),
    "testing_rows": len(X_test),
    "outer_train_ratio": 0.70,
    "outer_test_ratio": 0.30,
    "validation_fraction_of_training": VALIDATION_SIZE,
    "split_method": "Stratified random split",
    "classes": sorted(all_classes),
    "invalid_columns_removed": invalid_columns,
    "unusable_features_removed": unusable_features,
    "variance_features": variance_features,
    "correlated_features_removed": [
        record["Removed_Feature"]
        for record in correlation_records
    ],
    "training_medians": {
        key: float(value)
        for key, value in medians.items()
    }
}

with open(FEATURES_FILE, "w") as f:
    json.dump(metadata, f, indent=4)

mi_results.to_csv(
    MI_FILE,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("FEATURE SELECTION COMPLETED")
print("=" * 70)

print("Total rows:", len(X))
print("Selected features:", len(selected_features))

print("\nOUTPUT FILES")
print(TRAIN_FILE)
print(VALIDATION_FILE)
print(TEST_FILE)
print(FEATURES_FILE)
print(MI_FILE)
print(CORRELATION_FILE)

print("\nSMOTE: Not applied")
print("Split: Stratified 70:30 outer split")
print("Validation: 10% of training partition")
print("Feature selection: Training partition only")
print("Correlation threshold:", CORRELATION_THRESHOLD)
print("Top K features:", TOP_K_FEATURES)

print("\nCYBERSCOPE AI FEATURE PIPELINE FINISHED")
