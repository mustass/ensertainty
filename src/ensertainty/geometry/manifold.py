import equinox as eqx
from jax import jit, vmap, jacfwd, jacrev


class Manifold(eqx.Module):
    jacobian: callable

    def __init__(self, decoder: callable):
        jacobian_forward = jacrev(decoder)

        self.jacobian = vmap(lambda x: jacobian_forward(x[None, ...]).squeeze())

    def metric(self, inp):
        J = self.jacobian(inp)
        return J.transpose(0, 2, 1) @ J

    def ensemble_metric(self, inp):
        J = self.jacobian(inp)
        return J.transpose(0, 1, 3, 2) @ J
