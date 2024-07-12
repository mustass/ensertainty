import jax
from typing import Callable
import equinox as eqx
import logging
from jax import Array
import equinox.nn as nn
import jax.nn as jnn
import jax.numpy as jnp

class DenseEncoder(eqx.Module):
    in_features: int
    out_features:int
    activation:Callable
    body: eqx.Module
    mean_head: eqx.Module
    sigma_head: eqx.Module
    
    def __init__(self, h:int,w:int,channels_in:int, out_features:int, key:Array):
        self.in_features = h*w*channels_in
        self.out_features = out_features
        self.activation = jnn.relu

        keys = jax.random.split(key, 6)

        # Dense VAE encoder
        self.body = nn.Sequential(
            [nn.Linear(self.in_features, 512, key=keys[0]),
            eqx.nn.Lambda(self.activation),
            nn.Linear(512, 256, key=keys[1]),
            eqx.nn.Lambda(self.activation),
            nn.Linear(256, 128, key=keys[2]),
            eqx.nn.Lambda(self.activation),
            nn.Linear(128, 64, key=keys[3]),
            eqx.nn.Lambda(self.activation)]
        )

        self.mean_head = nn.Linear(64, out_features, key=keys[4])
        self.sigma_head = nn.Linear(64, out_features, key=keys[5])

    def __call__(self, x):
        x = jnp.ravel(x)
        x = self.body(x)
        mean = self.mean_head(x)
        sigma = self.sigma_head(x)
        return mean, sigma
    
class DenseDecoder(eqx.Module):
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

        keys = jax.random.split(key, 6)

        self.body = nn.Sequential(
            [nn.Linear(in_features, 64, key=keys[0]),
            eqx.nn.Lambda(self.activation),
            nn.Linear(64, 128, key=keys[1]),
            eqx.nn.Lambda(self.activation),
            nn.Linear(128, 256, key=keys[2]),
            eqx.nn.Lambda(self.activation),
            nn.Linear(256, 512, key=keys[3]),
            eqx.nn.Lambda(self.activation)]
        )

        self.mean_head = nn.Linear(512, self.out_features, key=keys[4])
        self.sigma_head = nn.Linear(512, self.out_features, key=keys[5])
    
    def __call__(self, x):
        x = self.body(x)
        mean = self.mean_head(x)
        log_var = self.sigma_head(x)
        return mean, log_var