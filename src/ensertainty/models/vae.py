from typing import Any, List
import equinox as eqx
from jax import Array
import jax.numpy as jnp

import equinox as eqx
import jax
import jax.scipy.stats as stats

class EnsembleVAE(eqx.Module):
        
        encoder: eqx.Module
        decoders: eqx.Module
        num_decoders: int
        latent_dim: int
        kl_weight: float

        def __init__(self, encoder:eqx.Module, decoders: eqx.Module, num_decoders:int, latent_dim:int, kl_weight:float):
            
            self.encoder = encoder
            self.decoders = decoders
            self.num_decoders = num_decoders
            self.latent_dim = latent_dim
            self.kl_weight = kl_weight
            
        def encode(self, x: Array, key: Array) -> List[Array]:
            mu, log_var = self.encoder(x)
            z = jax.random.normal(key, mu.shape) * jnp.exp(log_var * 0.5) + mu
            return z, mu, log_var
        
        
        def sample(self, num_samples: int, key:Array) -> Array:
            z = jax.random.normal(key, (num_samples, self.latent_dim))
            reconstructions = self._decode(z)
            return reconstructions
        
        def __call__(self, x: Array, key:Array) -> List[Array]:
            
            return self._decode(self.encode(x,key))
        

        def _decode(self, z: Array) -> Array:
            @eqx.filter_vmap(in_axes=(eqx.if_array(0), 0))
            def _decode_per_ensamble(model, x):
                return model(x)

            return _decode_per_ensamble(self.decoders, z)
    

        def decode(self, u: Array) -> Any:

            h = jnp.squeeze(u)
            h = jnp.expand_dims(h, 1)
            u = jnp.repeat(h, self.num_decoders, 1)
            u = jnp.swapaxes(u, 0, 1)
            @eqx.filter_vmap(in_axes=(eqx.if_array(0), 0))
            def _decode_per_ensamble(model, x):
                return model(x)[0]
            return _decode_per_ensamble(self.decoders, u)
        
        def log_prob(self, x:Array, mu:Array, cov: Array) -> Array:
            cov = jnp.diag(jnp.exp(cov))
            return stats.multivariate_normal.logpdf(x, mu, cov)