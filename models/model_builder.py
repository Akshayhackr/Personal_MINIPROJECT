"""
CyberScope AI - Unified Hybrid Model Builder
============================================
Supports:
1. Positional Encoding in Transformer branch (sinusoidal, learnable, none).
2. 1D-CNN (domain tabular) vs 2D-CNN (legacy 6x6 grid) spatial representation.
3. Temporal branch comparison: Transformer vs LSTM vs BiLSTM vs GRU.
4. Flexible loss selection (categorical crossentropy vs focal loss).
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from models.positional_encoding import PositionalEncoding
from models.feature_representation import build_1d_cnn_branch, build_2d_cnn_branch


def build_temporal_branch(seq_input, temporal_type="transformer", pe_type="sinusoidal", d_model=64, num_heads=4, dropout=0.1):
    """
    Constructs the temporal branch for the 10-step flow sequence.
    
    temporal_type: 'transformer', 'lstm', 'bilstm', or 'gru'
    pe_type: 'sinusoidal', 'learnable', or 'none' (used if temporal_type == 'transformer')
    """
    temporal_type = temporal_type.lower()
    seq_len = seq_input.shape[1]
    in_features = seq_input.shape[2]
    
    if temporal_type == "transformer":
        # Project raw features to d_model space for rich multi-head attention
        x = layers.Dense(d_model, activation="relu", name="temporal_projection")(seq_input)
        
        # Inject positional encoding to encode sequence ordering
        x = PositionalEncoding(
            sequence_length=seq_len,
            d_model=d_model,
            mode=pe_type,
            name=f"pos_encoding_{pe_type}"
        )(x)
        
        # Multi-Head Self-Attention
        attn_out = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
            name="multi_head_attention"
        )(x, x)
        attn_out = layers.Dropout(dropout, name="attn_dropout")(attn_out)
        x = layers.LayerNormalization(epsilon=1e-6, name="ln_1")(x + attn_out)
        
        # Feed-Forward Network
        ffn = layers.Conv1D(filters=d_model * 2, kernel_size=1, activation="relu", name="ffn_1")(x)
        ffn = layers.Dropout(dropout, name="ffn_dropout")(ffn)
        ffn = layers.Conv1D(filters=d_model, kernel_size=1, name="ffn_2")(ffn)
        x = layers.LayerNormalization(epsilon=1e-6, name="ln_2")(x + ffn)
        
        # Temporal aggregation
        t_out = layers.GlobalAveragePooling1D(name="transformer_gap")(x)
        return t_out

    elif temporal_type == "lstm":
        x = layers.LSTM(64, return_sequences=True, name="lstm_1")(seq_input)
        x = layers.Dropout(dropout, name="lstm_dropout")(x)
        t_out = layers.LSTM(64, return_sequences=False, name="lstm_2")(x)
        return t_out

    elif temporal_type == "bilstm":
        x = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True), name="bilstm_1"
        )(seq_input)
        x = layers.Dropout(dropout, name="bilstm_dropout")(x)
        t_out = layers.Bidirectional(
            layers.LSTM(32, return_sequences=False), name="bilstm_2"
        )(x)
        return t_out

    elif temporal_type == "gru":
        x = layers.GRU(64, return_sequences=True, name="gru_1")(seq_input)
        x = layers.Dropout(dropout, name="gru_dropout")(x)
        t_out = layers.GRU(64, return_sequences=False, name="gru_2")(x)
        return t_out

    else:
        raise ValueError(
            f"Unsupported temporal architecture: {temporal_type}. "
            "Choose 'transformer', 'lstm', 'bilstm', or 'gru'."
        )


def build_cyberscope_hybrid_model(
    num_features=30,
    seq_len=10,
    num_classes=8,
    cnn_type="1d",
    temporal_type="transformer",
    pe_type="sinusoidal",
    d_model=64,
    num_heads=4,
    dropout=0.2,
    loss_type="categorical_crossentropy",
    learning_rate=0.001
):
    """
    Constructs and compiles the complete CyberScope Hybrid Architecture.
    
    Parameters:
    -----------
    num_features: int, number of tabular network traffic features (default: 30)
    seq_len: int, temporal sequence window length (default: 10)
    num_classes: int, number of output threat categories (default: 8)
    cnn_type: '1d' (domain-specialized tabular 1D-CNN) or '2d' (legacy 6x6 matrix 2D-CNN)
    temporal_type: 'transformer', 'lstm', 'bilstm', or 'gru'
    pe_type: 'sinusoidal', 'learnable', or 'none' (used if temporal_type == 'transformer')
    loss_type: 'categorical_crossentropy' or 'focal_loss'
    learning_rate: float
    """
    cnn_type = cnn_type.lower()
    
    # 1. Spatial Branch
    if cnn_type == "1d":
        spatial_in, spatial_out = build_1d_cnn_branch(input_shape=(num_features, 1), name_prefix="spatial_1d")
    elif cnn_type == "2d":
        spatial_in, spatial_out = build_2d_cnn_branch(input_shape=(6, 6, 1), name_prefix="spatial_2d")
    else:
        raise ValueError(f"Unknown cnn_type: {cnn_type}. Choose '1d' or '2d'.")

    # 2. Temporal Branch
    seq_in = layers.Input(shape=(seq_len, num_features), name="temporal_input")
    temporal_out = build_temporal_branch(
        seq_in,
        temporal_type=temporal_type,
        pe_type=pe_type,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout
    )

    # 3. Fusion & Multi-Class Classification Head
    fused = layers.concatenate([spatial_out, temporal_out], name="feature_fusion")
    dense = layers.Dense(128, activation="relu", name="fusion_dense_1")(fused)
    dense = layers.Dropout(0.3, name="fusion_dropout")(dense)
    dense = layers.Dense(64, activation="relu", name="fusion_dense_2")(dense)
    threat_output = layers.Dense(num_classes, activation="softmax", name="threat_classification")(dense)

    model = Model(
        inputs=[spatial_in, seq_in],
        outputs=threat_output,
        name=f"CYBERSCOPE_{cnn_type.upper()}_{temporal_type.upper()}"
    )

    # 4. Compile with appropriate loss function
    if loss_type == "focal_loss":
        loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0)
    else:
        loss_fn = tf.keras.losses.CategoricalCrossentropy()

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"]
    )

    return model
