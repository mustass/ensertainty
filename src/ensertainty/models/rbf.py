import equinox as eqx
from jax import Array
import jax
import jax.numpy as jnp

class RBF(eqx.Module):

    W: Array
    centroids: Array
    lambdas: Array
    c: Array

    def __init__(self, W, centroids, lambdas, c):
        self.W = W
        self.centroids = centroids
        self.lambdas = lambdas
        self.c = c

    def v_k(self,lmbda,center,input):
            return jnp.exp(-lmbda*jnp.linalg.norm(input-center)**2)
        
    def rbf(self, input):
        V = jax.vmap(lambda bandwidth, center: self.v_k(bandwidth,center,input))(self.lambdas,self.centroids)
        return self.W @ V + self.c
    
    def __call__(self, input):
        return self.rbf(input)