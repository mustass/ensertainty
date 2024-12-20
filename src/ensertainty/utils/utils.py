import importlib
from typing import Any
import logging
import os
import shutil
from jax import tree_flatten
import jax.numpy as jnp
from torch.utils.data import Dataset
import hydra
import jax
import equinox as eqx
import numpy as np
import jax.random as random
from ensertainty.data.utils import select_classes
from itertools import product, combinations, chain
import random as pyrandom


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """
    Extract an object from a given path.
    https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)


def compute_num_params(pytree):
    return sum(x.size if hasattr(x, "size") else 0 for x in jax.tree_util.tree_leaves(pytree))


def save_useful_info(name) -> None:
    logging.info(hydra.utils.get_original_cwd())
    logging.info(os.getcwd())
    shutil.copytree(
        os.path.join(hydra.utils.get_original_cwd(), "src"),
        os.path.join(hydra.utils.get_original_cwd(), f"{os.getcwd()}/code/src"),
    )
    shutil.copytree(
        os.path.join(hydra.utils.get_original_cwd(), "data"),
        os.path.join(hydra.utils.get_original_cwd(), f"{os.getcwd()}/code/data"),
    )
    shutil.copy2(
        os.path.join(f"{hydra.utils.get_original_cwd()}/scripts", name),
        os.path.join(hydra.utils.get_original_cwd(), os.getcwd(), "code"),
    )


def l2_norm(tree):
    """Compute the l2 norm of a pytree of arrays. Useful for weight decay."""
    leaves, _ = tree_flatten(tree)
    return jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))


def max_func(tree):
    leaves, _ = tree_flatten(tree)
    return jnp.max(jnp.concatenate([jnp.asarray(abs(x)).ravel() for x in leaves], axis=None))


def init_decoder_ensamble(cfg, key):
    keys = jax.random.split(key, cfg["model"]["num_decoders"])

    @eqx.filter_vmap
    def make_ensamble(key):
        return load_obj(cfg["decoder"]["class_name"])(
            key=key, **cfg["decoder"]["params"]
        )

    return make_ensamble(keys)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def pick_pairs(dataset: Dataset, n_pairs: int, n_diff: int = None, seed: int = None):
    """
    dataset : Dataset
        Dataset to pick pairs from
    n_pairs : int
        Number of pairs to pick
    n_diff : int
        Number of pairs with different labels
    """
    pyrandom.seed(seed)
    np.random.seed(seed)
    key1, key2 = random.split(random.PRNGKey(seed), 2)
    logging.info(f"Need to pick {n_diff+n_pairs} pairs")
    indecies = np.asarray(random.randint(key1, (n_pairs * 2,), 0, len(dataset)))

    point_pairs = list(zip(indecies[:n_pairs], indecies[-n_pairs:]))

    logging.info(f"Picking following indecies for the {len(indecies)} random geodesic pairs: {indecies}")

    if n_diff is not None:
        classes = np.unique(np.argmax(dataset.targets, axis=1)).tolist()
        n_classes = len(classes)
        n_per_class = n_diff // n_classes

        list_of_lists = []
        for cls in classes:
            _indices = jnp.where(jnp.argmax(dataset.targets, axis=1) == cls)[0]
            _indices = jnp.setdiff1d(_indices, indecies)
            idxs = random.choice(key=key2, a=_indices, shape=(n_per_class,), replace=False)
            list_of_lists.append(np.array(idxs).tolist())

        _pairs = []
        for i, l in enumerate(list_of_lists):
            other_lists = list_of_lists[:i] + list_of_lists[i + 1 :]
            other_lists = list(chain.from_iterable(other_lists))
            c = list(product(l, other_lists))
            _pairs.extend(c)

        _pairs = set([tuple(sorted(pair)) for pair in _pairs])
        _pairs = pyrandom.sample(sorted(_pairs), n_diff)
        point_pairs.extend(_pairs)

    return point_pairs
