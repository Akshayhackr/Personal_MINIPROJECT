"""
CyberScope AI - Positional Encoding for Temporal Transformer Branch
===================================================================
Provides both Sinusoidal and Learnable Positional Encoding layers
for the 10-step network flow sequence in the CyberScope architecture.

Self-attention is permutation-invariant: without positional encoding,
the model cannot distinguish packet arrival order, burst timing, or
temporal progression of attack traffic.
"""

import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="CyberScope")
class SinusoidalPositionalEncoding(layers.Layer):
    """
    Fixed Sinusoidal Positional Encoding (Vaswani et al., 2017).
    
    PE(pos, 2i)   = sin(pos / (10000 ** (2i / d_model)))
    PE(pos, 2i+1) = cos(pos / (10000 ** (2i / d_model)))
    
    Computes deterministic harmonic frequencies for each position in
    the sequence, enabling the Multi-Head Attention layer to incorporate
    temporal packet order without adding trainable parameters.
    """
    def __init__(self, sequence_length=10, d_model=30, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = int(sequence_length)
        self.d_model = int(d_model)
        self.pe_matrix = self._compute_pe_matrix()

    def _compute_pe_matrix(self):
        pe = np.zeros((self.sequence_length, self.d_model), dtype=np.float32)
        position = np.arange(0, self.sequence_length, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, self.d_model, 2, dtype=np.float32) *
            -(math.log(10000.0) / self.d_model)
        )
        
        pe[:, 0::2] = np.sin(position * div_term)
        if self.d_model % 2 == 1:
            pe[:, 1::2] = np.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = np.cos(position * div_term)
            
        pe = tf.constant(pe[np.newaxis, ...], dtype=tf.float32)  # shape: (1, seq_len, d_model)
        return pe

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        """Adds sinusoidal positional encoding to inputs."""
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pe_matrix[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "d_model": self.d_model,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="CyberScope")
class LearnablePositionalEncoding(layers.Layer):
    """
    Learnable Positional Encoding.
    
    Uses a trainable parameter matrix of shape (1, sequence_length, d_model)
    that optimizes positional embeddings during backpropagation specifically
    for network traffic sequences.
    """
    def __init__(self, sequence_length=10, d_model=30, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = int(sequence_length)
        self.d_model = int(d_model)
        self.pos_embedding = None

    def build(self, input_shape):
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, self.sequence_length, self.d_model),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        """Adds learnable positional embeddings to inputs."""
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_embedding[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "d_model": self.d_model,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="CyberScope")
class PositionalEncoding(layers.Layer):
    """
    Unified Positional Encoding Layer supporting both 'sinusoidal' and 'learnable' modes.
    """
    def __init__(self, sequence_length=10, d_model=30, mode="sinusoidal", **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = int(sequence_length)
        self.d_model = int(d_model)
        self.mode = mode.lower()
        
        if self.mode == "sinusoidal":
            self.pe_layer = SinusoidalPositionalEncoding(sequence_length, d_model)
        elif self.mode == "learnable":
            self.pe_layer = LearnablePositionalEncoding(sequence_length, d_model)
        elif self.mode == "none":
            self.pe_layer = None
        else:
            raise ValueError(f"Unknown positional encoding mode: {mode}. Choose 'sinusoidal', 'learnable', or 'none'.")

    def call(self, inputs):
        if self.pe_layer is not None:
            return self.pe_layer(inputs)
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "d_model": self.d_model,
            "mode": self.mode,
        })
        return config
