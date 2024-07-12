import os
import logging
from omegaconf import DictConfig, OmegaConf
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
import optax
from jax.tree_util import tree_flatten, tree_unflatten
import jax.random as random
from tqdm import tqdm
import equinox as eqx
from ensertainty.utils.utils import compute_num_params, load_obj
import pickle
import yaml
import shutil
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import jax.tree_util as jtu
from ..utils import l2_norm, max_func, COLORS, MARKERS
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ensertainty.geometry import Manifold, Geodesics
from jax import Array
from jax.random import PRNGKey, normal, uniform, choice, split
from sklearn.cluster import KMeans
from ensertainty.models import RBF

class TrainerModule:
    def __init__(self, model: eqx.Module, config: DictConfig, wandb_logger):
        """
        Module for summarizing all training functionalities for classification on CIFAR10.

        Inputs:
            model_name - String of the class name, used for logging and saving
            model_class - Class implementing the neural network
            model_hparams - Hyperparameters of the model, used as input to model constructor
            optimizer_name - String of the optimizer name, supporting ['sgd', 'adam', 'adamw']
            optimizer_hparams - Hyperparameters of the optimizer, including learning rate as 'lr'
            exmp_imgs - Example imgs, used as input to initialize the model
            seed - Seed to use in the model initialization
        """

        super().__init__()
        self.model = model
        self.optim_config = config["optimizer"]
        self.train_config = config["training"]
        self.loss_config = config["loss"]
        self.batch_size = config["datamodule"]["batch_size"]
        self.flat_dim = config["datamodule"]["dim"]
        self.grad_clipping_config = config["grad_clipping"]
        self.scheduler_config = config["scheduler"]
        self.config = config

        self.seed = self.train_config["seed"]
        self.early_stopping = self.train_config["early_stopping_patience"]

        self.logger = wandb_logger

        self.model_checkpoint_path = config["general"]["model_checkpoints_path"]
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model()

        self.best_model_path = None
        self.best_state_path = None

        self.config_saved = False

    def init_model(self):
        self.num_params = compute_num_params(eqx.filter(self.model, eqx.is_array))

        logging.info(f"👉 Number of trainable parameters network: {self.num_params}")

    # @optax.inject_hyperparams
    def init_optimizer(self, num_epochs, num_steps_per_epoch):
        self.n_steps_per_epoch = num_steps_per_epoch

        grad_transformations = []

        if self.grad_clipping_config is not None:
            grad_transformations.append(
                load_obj(self.grad_clipping_config["class_name"])(**self.grad_clipping_config["params"])
            )

        if self.scheduler_config["class_name"] == "optax.warmup_cosine_decay_schedule":
            self.scheduler_config["params"]["decay_steps"] = num_epochs * num_steps_per_epoch
            lr_schedule = load_obj(self.scheduler_config["class_name"])(**self.scheduler_config["params"])

        elif self.scheduler_config["class_name"] == "optax.piecewise_constant_schedule":
            assert len(self.scheduler_config["params"]["boundaries"]) == len(
                self.scheduler_config["params"]["scales"]
            ), "LR scheduler must have same number of boundaries and scales"
            boundaries_and_scales = dict(
                [
                    (num_epochs * num_steps_per_epoch * key, value)
                    for key, value in zip(
                        self.scheduler_config["params"]["boundaries"], self.scheduler_config["params"]["scales"]
                    )
                ]
            )
            lr_schedule = load_obj(self.scheduler_config["class_name"])(
                init_value=self.scheduler_config["params"]["init_value"],
                boundaries_and_scales=boundaries_and_scales,
            )
        elif self.scheduler_config["class_name"] == "optax.constant_schedule":
            lr_schedule = load_obj(self.scheduler_config["class_name"])(**self.scheduler_config["params"])
        elif self.scheduler_config["class_name"] == "optax.sgdr_schedule":
            n_iterations = (self.train_config["max_epochs"] * num_steps_per_epoch) // self.scheduler_config["params"][
                "decay_steps"
            ] + 1
            params = [dict(self.scheduler_config["params"]) for i in range(n_iterations)]
            lr_schedule = load_obj(self.scheduler_config["class_name"])(params)
        else:
            raise NotImplementedError

        grad_transformations.append(
            load_obj(self.optim_config["class_name"])(lr_schedule, **self.optim_config["params"])
        )

        self.optimizer = optax.chain(*grad_transformations)

        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))

    def train_model(self, train_loader, val_loader, random_key, num_epochs=200, logger=None):
        # Train model for defined number of epochs
        # We first need to create optimizer and the scheduler for the given number of epochs

        self.init_optimizer(num_epochs, len(train_loader))

        # Track best eval accuracy
        best_eval = jnp.inf
        best_eval_epoch = 0
        for epoch_idx in tqdm(range(1, num_epochs + 1)):
            random_key, key1, key2 = random.split(random_key, 3)

            self.train_epoch(train_loader, epoch=epoch_idx, random_key=key1)
            if epoch_idx % self.train_config.eval_every == 0:
                eval_loss = self.eval_model(val_loader, epoch_idx, random_key=key2, eval_type="val_set")

                logging.info(f"🧐 Epoch {epoch_idx} eval loss: {eval_loss:.4}")

                if eval_loss <= best_eval:
                    best_eval = eval_loss
                    best_eval_epoch = epoch_idx
                    self.best_model_path, self.best_state_path = self.save_model(
                        identifier="epoch_" + str(epoch_idx), replace=True
                    )

                if (self.early_stopping > 0) and (epoch_idx - best_eval_epoch > self.early_stopping):
                    logging.info(
                        f"😥 Eval loss has not improved in {self.early_stopping} epochs. \n Training stopped at best eval accuracy {best_eval} at epoch {best_eval_epoch}"
                    )

                    break

        self.save_train_run()

    def train_epoch(self, train_loader, epoch, random_key=None):
        # Train model for one epoch, and log avg loss and accuracy

        for i, batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            self.model, self.opt_state, metrics_dict = self.train_step(self.model, self.opt_state, batch, random_key)

            # metrics_dict["Learning rate": self.opt_state.hyperparams['learning_rate']]
            for dict_key, dict_val in metrics_dict.items():
                self.logger.log({"train_" + dict_key + "_batch": dict_val}, step=i + self.n_steps_per_epoch * epoch)

    def eval_model(self, data_loader, epoch=None, random_key=None, eval_type=None):
        # Test model on all images of a data loader and return avg loss
        key, random_key = random.split(random_key)

        eval_losses = []
        for batch in data_loader:
            eval_loss = self.eval_step(self.model, self.opt_state, batch, key)
            eval_losses.append(eval_loss)

        eval_loss = jnp.mean(jnp.stack(eval_losses))
        self.logger.log({eval_type + "_loss" + "_epoch": eval_loss}, step=self.n_steps_per_epoch * (epoch + 1))

        return eval_loss

    def save_model(self, identifier: str, replace=False):
        # Save current model at certain training iteration
        os.makedirs(self.model_checkpoint_path, exist_ok=True)

        if replace and (self.best_model_path is not None and self.best_state_path is not None):
            if os.path.exists(self.best_model_path) and os.path.exists(self.best_state_path):
                os.remove(self.best_model_path)
                os.remove(self.best_state_path)

        model_path = self.model_checkpoint_path + "model_" + identifier + ".eqx"
        state_path = self.model_checkpoint_path + "state_" + identifier + ".pickle"
        eqx.tree_serialise_leaves(model_path, self.model)
        with open(state_path, "wb") as file:
            pickle.dump(self.opt_state, file)

        if not self.config_saved:
            self.config_saved = self.save_config()

        return model_path, state_path

    def save_train_run(self):
        logging.info("👉🏻 Saving the last state of training...")
        # self.save_model(identifier="last", replace=False)

        logging.info("👉🏻 Saving the best state of training...")
        shutil.copy2(self.best_model_path, self.model_checkpoint_path + "model.eqx")
        shutil.copy2(self.best_state_path, self.model_checkpoint_path + "state.pickle")

        if self.best_model_path is not None and self.best_state_path is not None:
            if os.path.exists(self.best_model_path) and os.path.exists(self.best_state_path):
                os.remove(self.best_model_path)
                os.remove(self.best_state_path)

        logging.info(f"👍🏻 Run saved successfully to {self.model_checkpoint_path}")

    def save_config(self):
        logging.info("👉🏻 Saving the config...")
        config_yaml = OmegaConf.to_yaml(self.config, resolve=True)
        with open(self.model_checkpoint_path + "config.yml", "w") as outfile:
            yaml.dump(config_yaml, outfile, default_flow_style=False)
        return True

    def load_model(self, pretrained=False):
        # Load model. We use different checkpoint for pretrained models
        if not pretrained:
            self.model = eqx.tree_deserialise_leaves(self.model_checkpoint_path + "model.eqx", self.model)
        else:
            assert (
                self.train_config["checkpoint"] is not None
            ), "Loading pretrained model requires compatible checkpoint"
            self.model = eqx.tree_deserialise_leaves(
                self.train_config["checkpoint"] + self.train_config["checkpoint_model"], self.model
            )
            with open(self.train_config["checkpoint"] + self.train_config["checkpoint_state"], "rb") as file:
                self.opt_state = pickle.load(file)

    def checkpoint_exists(self):
        # Check whether a pretrained model exist for this
        raise NotImplementedError


class Trainer(TrainerModule):
    def create_functions(self):
        # Function to calculate the classification loss and accuracy for a model
        def calculate_loss(
            model,
            batch,
            key,
        ):
            imgs, _ = batch["image"], batch["label"]
            keys = split(key, imgs.shape[0])
            latents, mus, sigmas = jax.vmap(model.encode)(imgs, keys)
            latents_reshaped = jnp.reshape(latents, (-1,model.latent_dim))
            mus_x, sigmas_x = jax.vmap(model._decode, in_axes=0)(latents_reshaped)
            mus_x = jnp.reshape(mus_x, (self.batch_size, -1))
            sigmas_x = jnp.reshape(sigmas_x, (self.batch_size, -1))
            sigmas_x = jnp.ones_like(sigmas_x)
            imgs = jnp.squeeze(imgs.reshape(self.batch_size, -1))
            log_prob = load_obj(self.loss_config["class_name"])(mus_x, jnp.squeeze(imgs.reshape(self.batch_size, -1)))

            #log_prob = -1*jnp.mean(jax.vmap(model.log_prob, in_axes=(0, 0, 0))(imgs, mus_x, sigmas_x))

            kl_loss = lambda mu, sigma: -0.5 * jnp.sum(1 + sigma - mu ** 2 - jnp.exp(sigma), axis=-1)
            kl = jnp.mean(jax.vmap(kl_loss)(mus,sigmas))

            loss = log_prob +  model.kl_weight * kl
            return loss, (kl, log_prob)


        # Training function
        @eqx.filter_jit
        def train_step(model: eqx.Module, opt_state: PyTree, batch, key:Array):
            loss_fn = lambda params: calculate_loss(
                params,
                batch,
                key,
            )
            # Get loss, gradients for loss, and other outputs of loss function
            out, grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
            updates, opt_state = self.optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            metrics_dict = {
                "loss_value": out[0],
                "recons": out[1][1],
                "kl": out[1][0],
                "grads_norm": l2_norm(grads),
                "grads_max": max_func(grads),
                "updates_norm": l2_norm(updates),
                "updates_max": max_func(updates),
            }
            return model, opt_state, metrics_dict

        # Eval function
        @eqx.filter_jit
        def eval_step(model, opt_state, batch, key):
            # Return the accuracy for a single batch
            loss, (kl, recons) = calculate_loss(
                model,
                batch,
                key,
            )
            return loss

        @eqx.filter_jit
        def reconstruct(model: eqx.Module, batch, mode="mean"):
            reconstructed_images = jax.vmap(model)(batch["image"])
            ## reshape back to image
            original_image_size = batch["image"].shape[1:]

            loss = load_obj(self.loss_config["class_name"])
            loss_val = jax.vmap(loss, in_axes=(0, 0))(
                reconstructed_images, jnp.squeeze(batch["image"].reshape(self.batch_size, -1))
            )
            reconstructed_images = jnp.reshape(reconstructed_images, (-1,) + original_image_size)
            return reconstructed_images, loss_val

        self.train_step = train_step
        self.eval_step = eval_step
        self.reconstruct = reconstruct


class EnsembleTrainer(TrainerModule):

    def create_functions(self):
        # Function to calculate the classification loss and accuracy for a model
        def calculate_loss(
            model,
            batch,
            key,
        ):
            imgs, _ = batch["image"], batch["label"]
            keys = split(key, imgs.shape[0])
            latents, mus, sigmas = jax.vmap(model.encode)(imgs, keys)
            latents_reshaped = jnp.reshape(latents, (-1, model.num_decoders, model.latent_dim))
            mus_x, sigmas_x = jax.vmap(model._decode, in_axes=0)(latents_reshaped)
            mus_x = jnp.reshape(mus_x, (self.batch_size, -1))
            sigmas_x = jnp.reshape(sigmas_x, (self.batch_size, -1))
            sigmas_x = jnp.ones_like(sigmas_x)
            imgs = jnp.squeeze(imgs.reshape(self.batch_size, -1))
            log_prob = load_obj(self.loss_config["class_name"])(mus_x, jnp.squeeze(imgs.reshape(self.batch_size, -1)))

            #log_prob = -1*jnp.mean(jax.vmap(model.log_prob, in_axes=(0, 0, 0))(imgs, mus_x, sigmas_x))

            kl_loss = lambda mu, sigma: -0.5 * jnp.sum(1 + sigma - mu ** 2 - jnp.exp(sigma), axis=-1)
            kl = jnp.mean(jax.vmap(kl_loss)(mus,sigmas))

            loss = log_prob +  model.kl_weight * kl
            return loss, (kl, log_prob)

        # Training function
        @eqx.filter_jit
        def train_step(model: eqx.Module, opt_state: PyTree, batch, key:Array):
            loss_fn = lambda params: calculate_loss(
                params,
                batch,
                key,
            )
            # Get loss, gradients for loss, and other outputs of loss function
            out, grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
            updates, opt_state = self.optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            metrics_dict = {
                "loss_value": out[0],
                "recons": out[1][1],
                "kl": out[1][0],
                "grads_norm": l2_norm(grads),
                "grads_max": max_func(grads),
                "updates_norm": l2_norm(updates),
                "updates_max": max_func(updates),
            }
            return model, opt_state, metrics_dict

        # Eval function
        @eqx.filter_jit
        def eval_step(model, opt_state, batch, key: Array):

            loss, (kl, recons) = calculate_loss(
                model,
                batch,
                key,
            )
            return loss

        @eqx.filter_jit
        def reconstruct(model: eqx.Module, batch, mode="mean"):
            preds = jax.vmap(model, in_axes=0)(batch["image"])
            reconstructed_mean = jnp.mean(preds, axis=1)
            reconstructed_std = jnp.std(preds, axis=1)
            ## reshape back to image
            original_image_size = batch["image"].shape[1:]

            loss = load_obj(self.loss_config["class_name"])
            loss_val = jax.vmap(loss, in_axes=(1, None))(
                preds, jnp.squeeze(batch["image"].reshape(self.batch_size, -1))
            )

            reconstructed_images = {
                "mean": jnp.reshape(reconstructed_mean, (-1,) + original_image_size),
                "std": jnp.reshape(reconstructed_std, (-1,) + original_image_size),
            }[mode]
            return reconstructed_images, loss_val

        self.train_step = train_step
        self.eval_step = eval_step
        self.reconstruct = reconstruct

    
    def estimate_kmeans_and_bandwiths(self, model, train_loader, k, alpha, key: Array):
        """
        For benchmarking against Latent Space Oddity paper
        only for use with one decoder
        """

        # encode the data
        latents = []
        for i, batch in enumerate(train_loader):
            key1, key = split(key, 2)
            keys  = split(key1,batch["image"].shape[0])
            latent = jax.vmap(model.encode)(batch["image"], keys)
            latents.append(latent[0])
        
        latents = np.array(latents)
        latents = np.reshape(latents, (-1, model.latent_dim))

        # perform kmeans
        kmeans = KMeans(n_clusters=k, random_state=0).fit(latents)

        c_k = kmeans.cluster_centers_
        memberships = kmeans.labels_
        
        lambda_k = []
        for cluster in range(k):
            cluster_latents = latents[memberships == cluster]
            diff = cluster_latents-c_k[cluster]
            norms_sq=jax.vmap(lambda x: jnp.dot(x,x))(diff)
            mean = jnp.mean(norms_sq)
            result = 0.5 * (alpha * mean)**(-2)
            lambda_k.append(result)

        return c_k, lambda_k

    def train_rbf(self, model, train_loader, key: Array, k, alpha, epochs):
        
        centroids, lambdas = self.estimate_kmeans_and_bandwiths(model,train_loader, k, alpha, key)
        
        centroids = jnp.array(centroids)
        lambdas = jnp.array(lambdas)

        D = self.flat_dim
        W = jax.nn.softplus(jax.random.normal(key, (D, k)))
        c = jnp.ones(D)*self.config.model.rbf_c

        def v_k(lmbda,center,input):
            return jnp.exp(-lmbda*jnp.linalg.norm(input-center)**2)
        
        def rbf(W,input, lambdas,centroids, c):
            V = jax.vmap(lambda bandwidth, center: v_k(bandwidth,center,input))(lambdas,centroids)
            return W @ V + c
        

        self.rbf_optimizer = load_obj("optax.adam")(self.config.model.rbf_lr)
        opt_state = self.rbf_optimizer.init(W)

        def rbf_loss_fn(W, batch,key):
            W = jax.nn.softplus(W)
            imgs, _ = batch["image"], batch["label"]
            keys = split(key, imgs.shape[0])
            latents, mus, sigmas = jax.vmap(model.encode)(imgs, keys)
            latents_reshaped = jnp.reshape(latents, (-1, model.num_decoders, model.latent_dim))
            mus_x, _ = jax.vmap(model._decode, in_axes=0)(latents_reshaped)
            mus_x = jnp.reshape(mus_x, (self.batch_size, -1))
            precisions = jax.vmap(lambda input: rbf(W,input, lambdas, centroids, c))(jnp.squeeze(latents_reshaped))
            variances = jax.vmap(jax.vmap(lambda x: 1/x))(precisions)
            imgs = jnp.squeeze(imgs.reshape(self.batch_size, -1))
            log_prob = -1*jnp.mean(jax.vmap(model.log_prob, in_axes=(0, 0, 0))(imgs, mus_x, variances))

            kl_loss = lambda mu, sigma: -0.5 * jnp.sum(1 + sigma - mu ** 2 - jnp.exp(sigma), axis=-1)
            kl = jnp.mean(jax.vmap(kl_loss)(mus,sigmas))

            loss = log_prob + kl
            return loss, (kl, log_prob)
        
        @eqx.filter_jit
        def train_step(W: Array, opt_state: PyTree, batch, key:Array):
            loss_fn = lambda W: rbf_loss_fn(
                W,
                batch,
                key
            )
            # Get loss, gradients for loss, and other outputs of loss function
            out, grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(W)
            updates, opt_state = self.rbf_optimizer.update(grads, opt_state, W)
            W = eqx.apply_updates(W, updates)
            metrics_dict = {
                "loss_value": out[0],
                "recons": out[1][1],
                "kl": out[1][0],
                "grads_norm": l2_norm(grads),
                "grads_max": max_func(grads),
                "updates_norm": l2_norm(updates),
                "updates_max": max_func(updates),
            }
            return W, opt_state, metrics_dict
        
        
        best_w = jax.nn.softplus(W)
        best_loss = np.inf
        for epoch in range(epochs):
            for i, batch in enumerate(train_loader):
                W, opt_state, metrics_dict = train_step(W, opt_state, batch, key)
                print(f"Epoch {epoch}, Batch {i}, Loss: {metrics_dict['loss_value']}")
                if metrics_dict['recons'] < best_loss:
                    best_loss = metrics_dict['recons']
                    best_w = jax.nn.softplus(W)


        self.W = best_w
        self.centroids = centroids
        self.lambdas = lambdas
        self.rbf = lambda input: rbf(W,input)
        self.cs = c

        self.rbf = RBF(self.W, self.centroids, self.lambdas, self.cs)

    def save_rbf(self, identifier: str='rbf'):
        # Save current model at certain training iteration
        os.makedirs(self.model_checkpoint_path, exist_ok=True)

        model_path = self.model_checkpoint_path + "model_" + identifier + ".eqx"
        eqx.tree_serialise_leaves(model_path, self.rbf)
        logging.info(f"👉🏻 Saved RBF to {model_path}...")

    def load_rbf(self):
        # Load model. We use different checkpoint for pretrained models

        W = jax.nn.softplus(jax.random.normal(PRNGKey(0), (self.flat_dim, self.config.model.rbf_k)))
        centroids = jax.random.normal(PRNGKey(0), (self.config.model.rbf_k, self.config.model.latent_dim))
        lambdas = jax.random.normal(PRNGKey(0), (self.config.model.rbf_k,))
        c = jnp.ones(self.flat_dim)*0.1
        rbf = RBF(W, centroids, lambdas, c)


        self.rbf = eqx.tree_deserialise_leaves(self.model_checkpoint_path + "model_rbf.eqx", rbf)
        logging.info(f"👉🏻 Loaded RBF from {self.model_checkpoint_path}...")