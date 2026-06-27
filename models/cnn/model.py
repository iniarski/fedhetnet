from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn

N_FEATURES =33



@dataclass
class CNNConfig:
    cnn_dims: list[int]
    cnn_ker_szs: list[int]
    cnn_ker_dils: list[int]
    drop_rate: float
    dtype: str


class CNN(nn.Module):
    config: CNNConfig
    @nn.compact
    def __call__(self, x, training=True, n_classes=7):

        for n_features, kernel_size, kernel_dil in zip(self.config.cnn_dims, self.config.cnn_ker_szs, self.config.cnn_ker_dils):
            x = nn.Conv(features=n_features, kernel_size=(kernel_size,), padding='SAME', kernel_dilation=kernel_dil)(x)
            x = nn.gelu(x)
            x = nn.Dropout(self.config.drop_rate)(x, deterministic=not training)

        y = nn.Conv(features=n_classes, kernel_size=(1,), padding='SAME')(x)

        return y
