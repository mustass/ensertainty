import jax
import equinox as eqx

key = jax.random.PRNGKey(0)
keys = jax.random.split(key, 8)


# Create an ensemble of models
@eqx.filter_vmap
def make_ensemble(key):
    return eqx.nn.MLP(2, 2, 2, 2, key=key)


mlp_ensemble = make_ensemble(keys)


# Evaluate each member of the ensemble on the same data
@eqx.filter_vmap(in_axes=(eqx.if_array(0), None))
def evaluate_ensemble(model, x):
    return model(x)


evaluate_ensemble(mlp_ensemble, jax.random.normal(key, (2,)))


# Evaluate each member of the ensemble on different data
@eqx.filter_vmap
def evaluate_per_ensemble(model, x):
    return model(x)


print(mlp_ensemble)
print(evaluate_per_ensemble(mlp_ensemble, jax.random.normal(key, (8, 2))))
print(evaluate_ensemble(mlp_ensemble, jax.random.normal(key, (2,))))

array = jax.random.normal(key, (3,))
print(array)
array = jax.numpy.expand_dims(array, 1)
print(array)
array = jax.numpy.repeat(array, 2, 1)
print(array)
array = jax.numpy.swapaxes(array, 0, 1)
print(array)
