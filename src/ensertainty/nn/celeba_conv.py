import jax
from typing import Callable
import equinox as eqx
import logging
from jax import Array
import equinox.nn as nn
import jax.nn as jnn
import jax.numpy as jnp


class ConvCelebAEncoder(eqx.Module):
    in_channels: int
    out_features:int
    activation:Callable
    body: eqx.Module
    mean_head: eqx.Module
    sigma_head: eqx.Module
    
    def __init__(self, h:int,w:int,channels_in:int, out_features:int, key:Array):
        self.in_channels = channels_in
        self.out_features = out_features
        self.activation = jnn.relu

        keys = jax.random.split(key, 7)

        # Dense VAE encoder
        self.body = nn.Sequential(
            [nn.Conv2d(self.in_channels, 32, kernel_size=3,stride=2, padding=1, key=keys[0]),
            eqx.nn.Lambda(self.activation),
            nn.Conv2d(32, 64, kernel_size=3,stride=2, padding=1, key=keys[1]),
            eqx.nn.Lambda(self.activation),
            nn.Conv2d(64, 128, kernel_size=3,stride=2, padding=1, key=keys[2]),
            eqx.nn.Lambda(self.activation),
            nn.Conv2d(128, 256, kernel_size=3,stride=2, padding=1, key=keys[3]),
            eqx.nn.Lambda(self.activation),
            nn.Conv2d(256, 512, kernel_size=3,stride=2, padding=1, key=keys[4]),
            eqx.nn.Lambda(self.activation)
            ]
        )

        self.mean_head = nn.Linear(512*5 * 5, out_features, key=keys[5])
        self.sigma_head = nn.Linear(512*5 * 5, out_features, key=keys[6])

    def __call__(self, x):
        x = jnp.transpose(x, (2,0,1))
        x = self.body(x)
        mean = self.mean_head(x)
        sigma = self.sigma_head(x)
        return mean, sigma
    
class ConvCelebADecoder(eqx.Module):
    in_features:int
    out_features:int
    activation:Callable
    body: eqx.Module
    mean_head: eqx.Module
    sigma_head: eqx.Module
    
    def __init__(self, in_features:int, h:int,w:int,channels_in:int, key:Array):
        self.in_features = in_features
        self.out_features = h*w*channels_in
        self.activation = jnn.relu

        keys = jax.random.split(key, 8)

        self.body = nn.Sequential(
            [
            nn.Linear(self.in_features,512*5 * 5, key=keys[0]),    
            nn.ConvTranspose2d(512, 256, kernel_size=3,stride=2, padding=1,output_padding=1, key=keys[0]),
            eqx.nn.Lambda(self.activation),
            nn.ConvTranspose2d(256, 128, kernel_size=3,stride=2, padding=1, output_padding=1, key=keys[1]),
            eqx.nn.Lambda(self.activation),
            nn.ConvTranspose2d(128, 64, kernel_size=3,stride=2, padding=1, output_padding=1, key=keys[2]),
            eqx.nn.Lambda(self.activation),
            nn.ConvTranspose2d(64, 32, kernel_size=3,stride=2, padding=1,output_padding=1, key=keys[3]),
            eqx.nn.Lambda(self.activation),
            nn.ConvTranspose2d(32, 32, kernel_size=3,stride=2, padding=1, output_padding=1, key=keys[4]),
            eqx.nn.Lambda(self.activation),
            nn.Conv2d(32, 3, kernel_size=3, padding=1, key=keys[5]),
            eqx.nn.Lambda(jnn.sigmoid),
            ]
        )

        self.mean_head = nn.Linear(2, 1, key=keys[6])
        self.sigma_head = nn.Linear(150, 150, key=keys[7])
    
    def __call__(self, x):
        x = self.body(x)
        log_var = self.sigma_head(jnp.ravel(x))
        return x, log_var