"""
CyberScope AI - Tabular Network Feature Representation & 1D-CNN Architecture
=============================================================================
Addresses Expectation 2:
Replacing arbitrary 6x6 grid reshaping with domain-aware 1D-CNN architectures
tailored for tabular network traffic.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# The 30 canonical features from CyberScope CIC-IDS2017 dataset
CANONICAL_FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Bwd Packet Length Max",
    "Bwd Packet Length Min", "Bwd Packet Length Mean", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Max", "Flow IAT Min", "Packet Length Mean", "Packet Length Std",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "Down/Up Ratio", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "min_seg_size_forward", "Active Mean", "Idle Mean"
]

# Domain-specific semantic grouping for network traffic:
# Features within the same group share strong statistical and protocol dependencies.
DOMAIN_FEATURE_GROUPS = {
    "Endpoints_and_Windows": [
        "Destination Port", "min_seg_size_forward",
        "Init_Win_bytes_forward", "Init_Win_bytes_backward"
    ],
    "Flow_Timing_and_Dynamics": [
        "Flow Duration", "Flow IAT Mean", "Flow IAT Max", "Flow IAT Min",
        "Active Mean", "Idle Mean"
    ],
    "Packet_Rates_and_Volumes": [
        "Flow Bytes/s", "Flow Packets/s", "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Down/Up Ratio"
    ],
    "Packet_Size_Distributions": [
        "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
        "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
        "Packet Length Mean", "Packet Length Std"
    ],
    "Protocol_Flags": [
        "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count"
    ]
}

# Ordered list of features according to semantic grouping
DOMAIN_ORDERED_FEATURES = []
for grp, f_list in DOMAIN_FEATURE_GROUPS.items():
    DOMAIN_ORDERED_FEATURES.extend(f_list)


def get_feature_reordering_indices(original_features=None):
    """
    Returns permutation indices to rearrange features from original order
    into domain-grouped semantic order.
    """
    if original_features is None:
        original_features = CANONICAL_FEATURES
    
    orig_to_idx = {f: i for i, f in enumerate(original_features)}
    reorder_indices = []
    for f in DOMAIN_ORDERED_FEATURES:
        if f in orig_to_idx:
            reorder_indices.append(orig_to_idx[f])
    
    # If any features were not in the domain list, append them at the end
    remaining = [i for i in range(len(original_features)) if i not in reorder_indices]
    reorder_indices.extend(remaining)
    return np.array(reorder_indices, dtype=np.int32)


def build_1d_cnn_branch(input_shape=(30, 1), name_prefix="1d_cnn"):
    """
    Domain-specialized 1D-CNN branch for tabular network traffic.
    
    Uses multi-scale 1D convolutions (kernel size 3 and 5) to capture
    both intra-group feature correlations and cross-group interactions,
    avoiding the artificial 2D spatial artifacts of grid reshaping.
    """
    inputs = layers.Input(shape=input_shape, name=f"{name_prefix}_input")
    
    # Block 1: Local feature extraction
    c1 = layers.Conv1D(32, kernel_size=3, padding="same", activation="relu", name=f"{name_prefix}_c1")(inputs)
    c1 = layers.BatchNormalization(name=f"{name_prefix}_bn1")(c1)
    
    # Block 2: Multi-scale receptive field
    c2 = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu", name=f"{name_prefix}_c2")(c1)
    c2 = layers.BatchNormalization(name=f"{name_prefix}_bn2")(c2)
    p2 = layers.MaxPooling1D(pool_size=2, name=f"{name_prefix}_pool1")(c2)
    
    # Block 3: Higher-level pattern synthesis
    c3 = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu", name=f"{name_prefix}_c3")(p2)
    c3 = layers.BatchNormalization(name=f"{name_prefix}_bn3")(c3)
    
    # Global pooling produces compact 128-dim tabular spatial representation
    out = layers.GlobalAveragePooling1D(name=f"{name_prefix}_gap")(c3)
    
    return inputs, out


def build_2d_cnn_branch(input_shape=(6, 6, 1), name_prefix="2d_cnn"):
    """
    Legacy 2D-CNN branch that reshapes 30 padded features into a 6x6 matrix.
    Maintained for baseline comparison and ablation.
    """
    inputs = layers.Input(shape=input_shape, name=f"{name_prefix}_input")
    c1 = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name=f"{name_prefix}_c1")(inputs)
    p1 = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool1")(c1)
    c2 = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name=f"{name_prefix}_c2")(p1)
    out = layers.GlobalAveragePooling2D(name=f"{name_prefix}_gap")(c2)
    return inputs, out
