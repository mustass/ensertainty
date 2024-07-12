import jax.numpy as jnp
import jax


def mse(preds, y):
    loss = ((preds - y) ** 2).mean(axis=0).sum()
    return loss


def cross_entropy_loss(preds, y):
    preds = jax.nn.log_softmax(preds, axis=-1)
    return -jnp.mean(jnp.sum(preds * y, axis=-1))
