import jax
from typing import Callable
import equinox as eqx
import logging

import equinox.nn as nn
import jax.nn as jnn
import jax.numpy as jnp

class ResidualBlock(eqx.Module):
    in_channels: int
    resample: str
    activation: eqx.Module
    dropout: eqx.Module
    residual_layer_1: eqx.Module
    shortcut_layer: eqx.Module
    residual_2_layer: eqx.Module

    def __init__(self, in_channels, resample=None, activation=jnn.relu, dropout=None, first=False, key=None):
        self.in_channels = in_channels
        self.resample = resample
        self.activation = activation
        self.dropout = dropout

        keys = jax.random.split(key, 3)

        self.residual_layer_1 = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, key=keys[0]
        )

        if resample is None:
            self.shortcut_layer = nn.Identity()
            self.residual_2_layer = nn.Conv2d(
                in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, key=keys[2]
            )
        elif resample == "down":
            self.shortcut_layer = nn.Conv2d(
                in_channels=in_channels, out_channels=2 * in_channels, kernel_size=3, stride=2, padding=1, key=keys[1]
            )
            self.residual_2_layer = nn.Conv2d(
                in_channels=in_channels, out_channels=2 * in_channels, kernel_size=3, stride=2, padding=1, key=keys[2]
            )
        elif resample == "up":
            self.shortcut_layer = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=0 if first else 1,
                key=keys[1],
            )
            self.residual_2_layer = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=0 if first else 1,
                key=keys[2],
            )

    def __call__(self, inputs):
        shortcut = self.shortcut_layer(inputs)
        residual_1 = self.activation(inputs)
        residual_1 = self.residual_layer_1(residual_1)
        residual_2 = self.activation(residual_1)
        residual_2 = self.residual_2_layer(residual_2)

        return shortcut + residual_2


class ConvEncoder(eqx.Module):
    context_features: int
    channels_multiplier: int
    activation: Callable
    initial_layer: eqx.Module
    residual_blocks: list
    final_layer: eqx.Module

    def __init__(self, context_features, channels_multiplier, activation=jnn.relu, key=None):
        self.context_features = context_features
        self.channels_multiplier = channels_multiplier
        self.activation = activation

        keys = jax.random.split(key, 8)

        self.initial_layer = nn.Conv2d(1, channels_multiplier, kernel_size=1, key=keys[0])
        self.residual_blocks = [
            ResidualBlock(in_channels=channels_multiplier, key=keys[1]),
            ResidualBlock(in_channels=channels_multiplier, resample="down", key=keys[2]),
            ResidualBlock(in_channels=channels_multiplier * 2, key=keys[3]),
            ResidualBlock(in_channels=channels_multiplier * 2, resample="down", key=keys[4]),
            ResidualBlock(in_channels=channels_multiplier * 4, key=keys[5]),
            ResidualBlock(in_channels=channels_multiplier * 4, resample="down", key=keys[6]),
        ]
        self.final_layer = nn.Linear(
            in_features=(4 * 4 * channels_multiplier * 8), out_features=context_features, key=keys[7]
        )

    def __call__(self, inputs):
        temps = self.initial_layer(inputs)
        for residual_block in self.residual_blocks:
            temps = residual_block(temps)
        temps = self.activation(temps)
        outputs = self.final_layer(temps.reshape(-1, 4 * 4 * self.channels_multiplier * 8).squeeze())
        return outputs


class ConvDecoder(eqx.Module):
    latent_features: int
    channels_multiplier: int
    activation: Callable
    initial_layer: eqx.Module
    residual_blocks: list
    final_layer: eqx.Module

    def __init__(self, latent_features, channels_multiplier, activation=jnn.relu, key=None):
        self.latent_features = latent_features
        self.channels_multiplier = channels_multiplier
        self.activation = activation

        keys = jax.random.split(key, 8)

        self.initial_layer = nn.Linear(
            in_features=latent_features, out_features=(4 * 4 * channels_multiplier * 8), key=keys[0]
        )
        self.residual_blocks = [
            ResidualBlock(in_channels=channels_multiplier * 8, key=keys[1]),
            ResidualBlock(in_channels=channels_multiplier * 8, resample="up", first=True, key=keys[2]),
            ResidualBlock(in_channels=channels_multiplier * 4, key=keys[3]),
            ResidualBlock(in_channels=channels_multiplier * 4, resample="up", key=keys[4]),
            ResidualBlock(in_channels=channels_multiplier * 2, key=keys[5]),
            ResidualBlock(in_channels=channels_multiplier * 2, resample="up", key=keys[6]),
        ]
        self.final_layer = nn.Conv2d(in_channels=channels_multiplier, out_channels=1, kernel_size=1, key=keys[7])
        self.mean_head = nn.Linear(in_features=(4 * 4 * channels_multiplier * 8), out_features=1, key=keys[8])
        self.sigma_head = nn.Linear(in_features=(4 * 4 * channels_multiplier * 8), out_features=1, key=keys[9])

    def __call__(self, inputs):
        temps = self.initial_layer(inputs).reshape(-1, self.channels_multiplier * 8, 4, 4).squeeze()
        for residual_block in self.residual_blocks:
            temps = residual_block(temps)
        temps = self.activation(temps)
        outputs = self.final_layer(temps)
        mean = self.mean_head(
            outputs.reshape(
                self.flat_dim,
            )
        )
        sigma = self.sigma_head(
            outputs.reshape(
                self.flat_dim,
            )
        )
        return mean, sigma


class ModifiedConvEncoder(eqx.Module):
    channels_in: int
    out_features: int
    channels_multiplier: int
    activation: Callable
    initial_layer: eqx.Module
    residual_blocks: list
    flat_dim: int
    mean_head: eqx.Module
    sigma_head: eqx.Module

    def __init__(
        self, h, w, channels_in, out_features, levels=4, channels_multiplier=1, activation=jnn.relu, key=None
    ):
        self.channels_in = channels_in
        self.out_features = out_features
        self.channels_multiplier = channels_multiplier
        self.activation = activation

        keys = jax.random.split(key, levels + 3)

        self.initial_layer = nn.Conv2d(channels_in, channels_in * channels_multiplier, kernel_size=1, key=keys[0])
        blocks = []
        for i in range(levels):
            key1, key2 = jax.random.split(keys[i + 1], 2)
            blocks.append(ResidualBlock(in_channels=channels_in * channels_multiplier * 2**i, key=key1))
            blocks.append(
                ResidualBlock(in_channels=channels_in * channels_multiplier * 2**i, resample="down", key=key2)
            )
        self.residual_blocks = blocks

        self.flat_dim = (h // 2**levels) * (w // 2**levels) * channels_in * channels_multiplier * 2**levels
        if self.channels_in == 1:
            self.flat_dim = 4 * self.flat_dim
        self.mean_head = nn.Linear(in_features=self.flat_dim, out_features=out_features, key=keys[-2])
        self.sigma_head = nn.Linear(in_features=self.flat_dim, out_features=out_features, key=keys[-1])

    def __call__(self, inputs, context=None):
        assert context is None
        temps = self.initial_layer(inputs)
        for residual_block in self.residual_blocks:
            temps = residual_block(temps)
        temps = self.activation(temps)
        mean = self.mean_head(
            temps.reshape(
                self.flat_dim,
            )
        )
        sigma = self.sigma_head(
            temps.reshape(
                self.flat_dim,
            )
        )
        return mean, sigma


class Conv2dSameSize(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, key=None):
        same_padding = kernel_size // 2  # Padding that would keep the spatial dims the same
        super().__init__(in_channels, out_channels, kernel_size, padding=same_padding, key=key)
