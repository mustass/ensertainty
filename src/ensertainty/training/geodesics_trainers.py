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
from ensertainty.geometry import Manifold, Geodesics, GeodesicsRBF
from jax import Array
from jax.random import PRNGKey, normal, uniform, choice, split
from sklearn.cluster import KMeans
from ensertainty.models import RBF
from .trainers import TrainerModule
#matplotlib.rc("xtick", labelsize=10)
#matplotlib.rc("ytick", labelsize=10)

plot_fontsize = 10
figsize=(2.1, 2.1)
rc_fonts = {
"font.family": "serif",
"font.size": plot_fontsize,
"figure.figsize": figsize,
"text.usetex": True,
"text.latex.preamble":  r"""
        \usepackage{libertine}
        \usepackage[libertine]{newtxmath}
        """
}
matplotlib.rcParams.update(rc_fonts)


class GeodesicsEval(TrainerModule):
    def __init__(self, model: eqx.Module, config: DictConfig, wandb_logger):
        super().__init__(model, config, wandb_logger)

        self.n_steps = config.inference.geodesics_params.n_steps
        self.n_poly = config.inference.geodesics_params.n_poly
        self.n_t = config.inference.geodesics_params.n_t
        self.n_t_lengths = config.inference.geodesics_params.n_t_lengths
        self.latents = None
        self.labels = None
        self.meshgrid = None
        self.determinant = None

        self.n_ensemble = config.model.get("num_decoders", None)
        self.metric_mode = "ensemble" if (self.n_ensemble is not None) and (self.n_ensemble > 1) else "single"

        self.geodesics_optimizer = config.inference.geodesics_params.optimizer.class_name
        self.geodesics_optimizer_params = config.inference.geodesics_params.optimizer.params
        self.geodesics_lr = config.inference.geodesics_params.lr
        self.geodesics_mode = config.inference.geodesics_params.mode
        self.geodesics_method = config.inference.geodesics_params.method
        self.geodesics_bs = config.inference.geodesics_params.batch_size
        self.geodesics_init_mode = config.inference.geodesics_params.init_mode
        self.geodesics_init_scale = config.inference.geodesics_params.init_scale
        self.warmup_steps = config.inference.geodesics_params.warmup_steps
        self.early_stopping_n = config.inference.geodesics_params.early_stopping_n
        self.early_stopping_delta = config.inference.geodesics_params.early_stopping_delta

    def create_functions(self):
        def calculate_energy(key, t, diff_model, static_model):
            geodesics = eqx.combine(diff_model, static_model)
            energies = geodesics.calculate_energy(
                t,
                key,
                self.geodesics_mode,
                derivative="delta",
                metric_mode=self.metric_mode,
                n_ensemble=self.n_ensemble,
            )

            return jnp.sum(energies, axis=None), energies

        @eqx.filter_jit
        def geodesic_optim_step(g, t, opt_state, key, filter_spec):
            diff_model, static_model = eqx.partition(g, filter_spec)

            loss = lambda d, s: calculate_energy(key, t, d, s)

            (energy_value, energies), grads = jax.value_and_grad(loss, has_aux=True)(diff_model, static_model)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            g = eqx.apply_updates(g, updates)

            return g, energy_value, opt_state, energies

        self.geodesic_step = geodesic_optim_step

    def compute_geodesic(self, batch, key, plot=False):
        key_init, key_encode, key = random.split(key, 3)

        @eqx.filter_jit
        def _encode(batch, key):
            keys = split(key,2)
            return jax.vmap(self.model.encode)(batch, keys)[0]

        @eqx.filter_jit
        def _decode(batch):
            return jax.vmap(self.model.decode)(batch)

        encode_keys = split(key_encode,batch.shape[0])

        point_pairs = jax.vmap(_encode)(
            batch,
            encode_keys,
        )  ## The double vmap will have result (pair,from/to,dim), so 3 pairs, 4 dims will be (3,2,4)
        #point_pairs_decoded = jax.vmap(_decode)(point_pairs)
        euclidean_distances = jax.vmap(lambda x: jnp.linalg.norm(x[0, :] - x[1, :], ord=2))(point_pairs)
        euclidean_in_ambient = jnp.array([0.0]) # jax.vmap(lambda x: jnp.linalg.norm(x[0, :] - x[1, :], ord=2))(point_pairs_decoded)

        geodesic = Geodesics(
            self.model, self.n_poly, point_pairs, key_init, self.geodesics_init_mode, self.geodesics_init_scale
        )
        self.optimizer, opt_state = self.init_optimizer(eqx.filter(geodesic, eqx.is_array))

        filter_spec = jtu.tree_map(lambda _: False, geodesic)
        filter_spec = eqx.tree_at(
            lambda tree: tree.params,
            filter_spec,
            replace=True,
        )
        t = jnp.linspace(0, 1, self.n_t)
        best_energy = jnp.inf
        best_energy_step = 0

        energy_history = []
        for i in (pbar := tqdm(range(self.n_steps), desc="Training geodesic", leave=False)):
            key_geodesic_step, key = random.split(key, 2)
            geodesic, energy, opt_state, energies = self.geodesic_step(
                geodesic, t, opt_state, key_geodesic_step, filter_spec
            )

            energy_history.append(energy.item())

            if best_energy - energy > self.early_stopping_delta:
                best_energy = energy
                best_energies = np.array(energies)
                best_params = geodesic.params
                best_energy_step = i

            pbar.set_postfix(current_energy=energy.item())

            if (i - best_energy_step) > self.early_stopping_n:
                break

        geodesic = eqx.tree_at(lambda g: g.params, geodesic, best_params)

        length_key, key = random.split(key, 2)
        lengths = geodesic.calculate_length(
            jnp.linspace(0, 1, self.n_t_lengths),
            length_key,
            derivative="delta",
            metric_mode=self.metric_mode,
            n_ensemble=self.n_ensemble,
        )

        if plot:
            figure_det = (
                self.plot_deodesic(geodesic, jnp.linspace(0, 1, self.n_t_lengths), point_pairs, "determinant")
                if self.geodesics_mode == "metric"
                else self.plot_deodesic(geodesic, jnp.linspace(0, 1, self.n_t_lengths), point_pairs, "no_background")
            )
            if self.n_ensemble is not None:
                figure_uncertainty = self.plot_deodesic(geodesic, jnp.linspace(0, 1, self.n_t_lengths), point_pairs, "uncertainty")
            else:
                figure_uncertainty = None
            figure_indicatrix = (
                self.plot_deodesic(geodesic, jnp.linspace(0, 1, self.n_t_lengths), point_pairs, "indicatrix") if self.geodesics_mode == "metric" else None
            )
            figures = (figure_det, figure_uncertainty, figure_indicatrix)
        else:
            figures = None
            
        return (
            best_energy.item(),
            lengths,
            best_energies,
            energy_history,
            euclidean_distances,
            euclidean_in_ambient,
            figures,
        )

    def plot_deodesic(self, geodesic, t, point_pair, mode="determinant"):
        geodesic_paths = np.array(geodesic.eval(t))
        euclidean_paths = np.array(jax.vmap(lambda pair: geodesic._eval_line(t, pair))(point_pair))
        assert self.latents is not None, "Latents not computed"

        plt.close()
        #plt.style.use("fast")
        plt.style.use("fast")
        fig, ax = plt.subplots()

        for i, label in enumerate(np.unique(self.labels)):
            idx = np.where(self.labels == label)[0]
            ax.scatter(self.latents[idx, 0], self.latents[idx, 1], c=COLORS[i], marker=MARKERS[i], label=label, s=0.1)

        for i in range(point_pair.shape[0]):
            g_path = geodesic_paths[i]
            e_path = euclidean_paths[i]
            latent_from = point_pair[i, 0, :]
            latent_to = point_pair[i, 1, :]
            plt.plot(
                g_path[0, :],
                g_path[1, :],
                linestyle="-",
                marker="None",
                color="#00FFFF",
                linewidth=1.0,
                label=r"$\gamma$" if i == 0 else None,
            )
            #plt.plot(
            #    e_path[0, :],
            #    e_path[1, :],
            #    linestyle=":",
            #    color="#000000",
            #    linewidth=1,
            #    label=r"$\Delta$" if i == 0 else None,
            #)
            plt.scatter(latent_from[0], latent_from[1], facecolors="none", edgecolors="b", s=0.25)
            plt.scatter(latent_to[0], latent_to[1], facecolors="none", edgecolors="b", s=0.25)
        # add meshgrid and determinant as contour plot
        if mode == "determinant":
            plt.contourf(
                self.meshgrid[:, 0].reshape(100, 100),
                self.meshgrid[:, 1].reshape(100, 100),
                self.determinant.reshape(100, 100),
                zorder=0.0,
                levels=50,
                cmap="viridis",
            )
            plt.colorbar()
            # plt.title("Background: Determinant")
        elif mode == "uncertainty":
            plt.contourf(
                self.meshgrid[:, 0].reshape(100, 100),
                self.meshgrid[:, 1].reshape(100, 100),
                self.uncertainty.reshape(100, 100),
                zorder=0.0,
                levels=50,
                cmap="viridis",
            )
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("right", "5%", pad="25%")
            cbar = plt.colorbar(cax=cax)
            cbar.ax.yaxis.set_ticks_position('left')
            cbar.ax.tick_params(labelsize=int(plot_fontsize*0.5))
            cbar.set_label(r'Uncertainty', loc="center", rotation=270, labelpad=plot_fontsize)

        elif mode == "indicatrix":
            if self.n_ensemble is not None:
                for i in range(self.n_ensemble):
                    fig, ax = self.plot_indicatrices(self.meshgrid, self.indicatrices[i], (fig, ax))
            else:
                fig, ax = self.plot_indicatrices(self.meshgrid, self.indicatrices, (fig, ax))
        else:
            pass
        
        ax.set_xlim(right=3.0)  # xmax is your value
        ax.set_xlim(left=-3.0)  # xmin is your value
        ax.set_ylim(top=3.0)  # ymax is your value
        ax.set_ylim(bottom=-3.0)  # ymin is your value
        
        #box = ax.get_position()
        #ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        ax.axis("equal")
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            fancybox=True,
            shadow=True,
            ncol=len(self.labels)+1,
            markerscale=3.0,
            fontsize = int(plot_fontsize*0.8),
            columnspacing=0.5
        )
        ax.set_title(r"Geodesic path $\gamma$ in the latent space", fontsize = plot_fontsize)
        ax.grid(False)
        ax.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
        plt.tight_layout()
        
        
        return fig

    def latents_data(self, data_set, key: Array):
        logging.info("👉 Computing latents for the dataset...")

        @eqx.filter_jit
        def _encode(batch, key):
            return self.model.encode(batch, key)[0]

        latents = []
        labels = []
        for i, batch in enumerate(data_set):
            _k, key  = split(key,2)
            latents.append(_encode(batch[0],_k))
            labels.append(np.argmax(batch[1]))

        self.latents = np.array(latents)
        self.labels = np.array(labels)

        logging.info("👍 Latents computed successfully")

    def metric_indicatrices(self, metrics):
        logging.info("👉 Computing indicatrices in the latent space...")
        _, eigvecs = jnp.linalg.eigh(metrics)

        @eqx.filter_jit
        def gnorm(metric, eigenvectors):
            norm = jnp.sqrt(jnp.einsum("ij,ij->j", eigenvectors, jnp.dot(metric, eigenvectors)))
            eigenvectors = eigenvectors / norm
            return eigenvectors

        eigvecs = jax.vmap(gnorm, in_axes=(0, 0))(metrics, eigvecs)

        @eqx.filter_jit
        def indicatrix(eigvecs):
            t = jnp.linspace(-jnp.pi, jnp.pi, 100)
            indicatrix = jnp.array([jnp.cos(t), jnp.sin(t)])
            indicatrix = jnp.dot(eigvecs, indicatrix)
            return indicatrix

        indicatrices = jax.vmap(indicatrix, in_axes=0)(eigvecs)
        return indicatrices

    def plot_indicatrices(self, Z, indicatrices, plot=None):
        # Z gives a meshgrid of points
        # indicatrices gives the indicatrices at each of the points in the meshgrid
        selection = np.linspace(10, 90, 8).astype(int)
        # repeat 5 times and add 100 to each of the repeats
        selection = np.concatenate([selection + 500, selection + 3000, selection + 5500, selection + 8000])
        indicatrices = indicatrices[selection, :, :]
        if plot is None:
            plt.close()
            plt.style.use("fast")
            fig, ax = plt.subplots()
        else:
            fig, ax = plot

        # at each point in the meshgrid, plot the indicatrix
        for i, indicatrix in enumerate(indicatrices):
            ax.plot(
                indicatrix[0, :] + Z[selection[i], 0],
                indicatrix[1, :] + Z[selection[i], 1],
                linestyle="--",
                color=COLORS[i],
                linewidth=0.5,
                alpha=0.1 if plot is not None else 0.9,
            )
            # plot the point
            if plot is None:
                ax.scatter(Z[selection[i], 0], Z[selection[i], 1], c=COLORS[i], s=2)
        ax.axis("equal")
        plt.xlim(right=3.0)  # xmax is your value
        plt.xlim(left=-3.0)  # xmin is your value
        plt.ylim(top=3.0)  # ymax is your value
        plt.ylim(bottom=-3.0)  # ymin is your value
        if plot is None:
            plt.grid(True)

        return fig, ax

    def metric_determinants(self):
        logging.info("👉 Computing metric determinant in the latent space...")
        x = np.linspace(-5.0, 5.0, 100)
        y = np.linspace(-5.0, 5.0, 100)
        X, Y = np.meshgrid(x, y)

        # make them a 2D array
        Z = np.array([X.flatten(), Y.flatten()]).T
        Z = jnp.array(Z)

        # batch Zs by 10 for self.manifold.ensemble_metric calculation
        if self.n_ensemble is not None:
            logging.info(f"👉 Batching metric determinant computation due to ensembling ...")
            Z_split = jnp.array_split(Z, 10)
            metrics = []
            for subarray in Z_split:
                metric = self.manifold.ensemble_metric(subarray)
                metrics.append(metric)

            metric = jnp.concatenate(metrics)
            determinants = jax.vmap(jax.vmap(jnp.linalg.det, in_axes=0), in_axes=0)(metric)
            determinant_meaned = jnp.mean(determinants, axis=1)
            indicatrices_per_ensamble = jax.vmap(self.metric_indicatrices, in_axes=1)(metric)
        else:
            metric = self.manifold.metric(Z)
            determinant = jnp.linalg.det(metric)
            indicatrices = self.metric_indicatrices(metric)
        ## combine meshgrid and determinant to return
        self.meshgrid = np.array(Z)
        self.determinant = np.array(determinant) if self.n_ensemble is None else np.array(determinant_meaned)
        self.determinants = np.array(determinants) if self.n_ensemble is not None else None
        self.indicatrices = np.array(indicatrices) if self.n_ensemble is None else np.array(indicatrices_per_ensamble)
        logging.info(
            f"👍 Determinants computed successfully max: {np.max(self.determinant,axis=None)} min: {np.min(self.determinant,axis=None)}"
            if self.n_ensemble is None
            else f"👍 Determinants computed successfully max: {jnp.max(determinants,axis=None).item()} min: {jnp.min(determinants,axis=None).item()}"
        )

        figures_determinants = [self.plot(Z, self.determinant, "Determinant landscape overall")]
        figures_indicatrices = []

        if self.n_ensemble is not None:
            plt.close()
            plt.style.use("fast")
            plots_in = plt.subplots()
            for i in range(self.n_ensemble):
                figures_determinants.append(self.plot(Z, determinants[:, i], f"Determinant landscape decoder {i+1}"))
                plots_in = self.plot_indicatrices(Z, indicatrices_per_ensamble[i], plots_in)
                figures_indicatrices.append(self.plot_indicatrices(Z, indicatrices_per_ensamble[i])[0])
            figures_indicatrices = [plots_in[0]] + figures_indicatrices
        else:
            figures_indicatrices.append(self.plot_indicatrices(Z, self.indicatrices)[0])
        return figures_determinants, figures_indicatrices

    def plot(self, linspace, linspaced_quantity, title):
        plt.close()
        plt.style.use("fast")
        fig, ax = plt.subplots()

        for i, label in enumerate(np.unique(self.labels)):
            idx = np.where(self.labels == label)[0]
            ax.scatter(self.latents[idx, 0], self.latents[idx, 1], c=COLORS[i], marker=MARKERS[i], label=label, s=2)

        # add meshgrid and determinant as contour plot
        plt.contourf(
            linspace[:, 0].reshape(100, 100),
            linspace[:, 1].reshape(100, 100),
            linspaced_quantity.reshape(100, 100),
            zorder=0.0,
            levels=50,
            cmap="viridis",
        )
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "10%", pad="25%")
        cbar = plt.colorbar(cax=cax)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.tick_params(labelsize=int(plot_fontsize*0.5))
        cbar.set_label(r'Uncertainty', loc="center", rotation=270,  labelpad=plot_fontsize)

        plt.xlim(right=3.0)  # xmax is your value
        plt.xlim(left=-3.0)  # xmin is your value
        plt.ylim(top=3.0)  # ymax is your value
        plt.ylim(bottom=-3.0)  # ymin is your value
        #plt.grid(False)
        ax.axis("equal")
        #box = ax.get_position()
        #ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            fancybox=True,
            shadow=True,
            ncol=len(self.labels),
            markerscale=3.0,
        )

        ax.set_title(f"Test set encoded into the latent space")
        ax.grid(False)
        ax.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
        plt.tight_layout()
        return fig

    def compute_uncertainty(self, mode="mean"):
        if self.n_ensemble is None:
            logging.info(f"👎🏼 Cannot compute uncertainties when num_decoders is None. Returning None.")
            return None
        x = np.linspace(-5.0, 5.0, 100)
        y = np.linspace(-5.0, 5.0, 100)
        X, Y = np.meshgrid(x, y)

        # make them a 2D array
        Z = np.array([X.flatten(), Y.flatten()]).T
        Z = jnp.array(Z)

        logging.info(f"👉 Batching uncertanties computation due to ensembling ...")
        Z_split = jnp.array_split(Z, 10)
        decodes_list = []
        for subarray in Z_split:
            decodes = jax.vmap(self.model.decode)(subarray)
            decodes_list.append(decodes)
        decodes = jnp.concatenate(decodes_list)
        decodes = jnp.reshape(decodes, (10000,self.n_ensemble,-1))

        print(f"Shape of decodes: {decodes.shape}")
        uncertainty = {
            "mean": jnp.mean,
            "max": jnp.max,
        }[
            mode
        ](jnp.std(decodes, axis=1), axis=1)

        self.meshgrid = Z
        self.uncertainty = np.array(uncertainty)

        return self.plot(Z, uncertainty, "Uncertainties")

    def init_optimizer(self, params):
        # grad_transformations = [optax.clip_by_global_norm(1.0)]

        # lr_schedule = optax.warmup_cosine_decay_schedule(
        #    init_value=0.0,
        #    peak_value=self.geodesics_lr,
        #    warmup_steps=self.warmup_steps,
        #    decay_steps=self.n_steps,
        #    end_value=self.geodesics_lr*0.1
        # )

        # grad_transformations.append(
        #    load_obj(self.geodesics_optimizer)(self.geodesics_lr, **self.geodesics_optimizer_params)
        # )

        # self.optimizer = optax.chain(*grad_transformations)
        optimizer = load_obj(self.geodesics_optimizer)(self.geodesics_lr, **self.geodesics_optimizer_params)

        opt_state = optimizer.init(params)
        return optimizer, opt_state
    



class GeodesicsEvalRBF(GeodesicsEval):
    def create_functions(self):
        def calculate_energy(key, t, diff_model, static_model):
            geodesics = eqx.combine(diff_model, static_model)
            energies = geodesics.calculate_energy(
                t,
                key,
                self.geodesics_mode,
                derivative="delta",
                metric_mode=self.metric_mode,
                n_ensemble=self.n_ensemble,
            )

            return jnp.sum(energies, axis=None), energies

        @eqx.filter_jit
        def geodesic_optim_step(g, t, opt_state, key, filter_spec):
            diff_model, static_model = eqx.partition(g, filter_spec)

            loss = lambda d, s: calculate_energy(key, t, d, s)

            (energy_value, energies), grads = jax.value_and_grad(loss, has_aux=True)(diff_model, static_model)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            g = eqx.apply_updates(g, updates)

            return g, energy_value, opt_state, energies

        self.geodesic_step = geodesic_optim_step

    def compute_geodesic(self, batch, key, plot=False):
        key_init, key_encode, key = random.split(key, 3)

        @eqx.filter_jit
        def _encode(batch, key):
            keys = split(key,2)
            return jax.vmap(self.model.encode)(batch, keys)[0]

        @eqx.filter_jit
        def _decode(batch):
            return jax.vmap(self.model.decode)(batch)

        encode_keys = split(key_encode,batch.shape[0])

        point_pairs = jax.vmap(_encode)(
            batch,
            encode_keys,
        )  ## The double vmap will have result (pair,from/to,dim), so 3 pairs, 4 dims will be (3,2,4)
        #point_pairs_decoded = jax.vmap(_decode)(point_pairs)
        euclidean_distances = jax.vmap(lambda x: jnp.linalg.norm(x[0, :] - x[1, :], ord=2))(point_pairs)
        euclidean_in_ambient = jnp.array([0.0]) # jax.vmap(lambda x: jnp.linalg.norm(x[0, :] - x[1, :], ord=2))(point_pairs_decoded)

        geodesic = GeodesicsRBF(
            self.model, self.rbf, self.n_poly, point_pairs, key_init, self.geodesics_init_mode, self.geodesics_init_scale
        )
        self.optimizer, opt_state = self.init_optimizer(eqx.filter(geodesic, eqx.is_array))

        filter_spec = jtu.tree_map(lambda _: False, geodesic)
        filter_spec = eqx.tree_at(
            lambda tree: tree.params,
            filter_spec,
            replace=True,
        )
        t = jnp.linspace(0, 1, self.n_t)
        best_energy = jnp.inf
        best_energy_step = 0

        energy_history = []
        for i in (pbar := tqdm(range(self.n_steps), desc="Training geodesic", leave=False)):
            key_geodesic_step, key = random.split(key, 2)
            geodesic, energy, opt_state, energies = self.geodesic_step(
                geodesic, t, opt_state, key_geodesic_step, filter_spec
            )

            energy_history.append(energy.item())

            if best_energy - energy > self.early_stopping_delta:
                best_energy = energy
                best_energies = np.array(energies)
                best_params = geodesic.params
                best_energy_step = i

            pbar.set_postfix(current_energy=energy.item())

            if (i - best_energy_step) > self.early_stopping_n:
                break

        geodesic = eqx.tree_at(lambda g: g.params, geodesic, best_params)

        length_key, key = random.split(key, 2)
        lengths = geodesic.calculate_length(
            jnp.linspace(0, 1, self.n_t_lengths),
            length_key,
            derivative="delta",
            metric_mode=self.metric_mode,
            n_ensemble=self.n_ensemble,
        )

        if plot:
            figure_det = (
                self.plot_deodesic(geodesic, jnp.linspace(0, 1, self.n_t_lengths), point_pairs, "determinant")
                if self.geodesics_mode == "metric"
                else self.plot_deodesic(geodesic, jnp.linspace(0, 1, self.n_t_lengths), point_pairs, "no_background")
            )
            if self.n_ensemble is not None:
                figure_uncertainty = self.plot_deodesic(geodesic, jnp.linspace(0, 1, self.n_t_lengths), point_pairs, "uncertainty")
            else:
                figure_uncertainty = None
            figure_indicatrix = (
                self.plot_deodesic(geodesic, jnp.linspace(0, 1, self.n_t_lengths), point_pairs, "indicatrix") if self.geodesics_mode == "metric" else None
            )
            figures = (figure_det, figure_uncertainty, figure_indicatrix)
        else:
            figures = None

        return (
            best_energy.item(),
            lengths,
            best_energies,
            energy_history,
            euclidean_distances,
            euclidean_in_ambient,
            figures,
        )

    def plot_deodesic(self, geodesic, t, point_pair, mode="determinant"):
        geodesic_paths = np.array(geodesic.eval(t))
        euclidean_paths = np.array(jax.vmap(lambda pair: geodesic._eval_line(t, pair))(point_pair))
        assert self.latents is not None, "Latents not computed"

        plt.close()
        #plt.style.use("fast")
        plt.style.use("fast")
        fig, ax = plt.subplots()

        for i, label in enumerate(np.unique(self.labels)):
            idx = np.where(self.labels == label)[0]
            ax.scatter(self.latents[idx, 0], self.latents[idx, 1], c=COLORS[i], marker=MARKERS[i], label=label, s=0.1)

        for i in range(point_pair.shape[0]):
            g_path = geodesic_paths[i]
            e_path = euclidean_paths[i]
            latent_from = point_pair[i, 0, :]
            latent_to = point_pair[i, 1, :]
            plt.plot(
                g_path[0, :],
                g_path[1, :],
                linestyle="-",
                marker="None",
                color="#00FFFF",
                linewidth=1.0,
                label=r"$\gamma$" if i == 0 else None,
            )
            #plt.plot(
            #    e_path[0, :],
            #    e_path[1, :],
            #    linestyle=":",
            #    color="#000000",
            #    linewidth=1,
            #    label=r"$\Delta$" if i == 0 else None,
            #)
            plt.scatter(latent_from[0], latent_from[1], facecolors="none", edgecolors="b", s=0.25)
            plt.scatter(latent_to[0], latent_to[1], facecolors="none", edgecolors="b", s=0.25)
        # add meshgrid and determinant as contour plot
        if mode == "determinant":
            plt.contourf(
                self.meshgrid[:, 0].reshape(100, 100),
                self.meshgrid[:, 1].reshape(100, 100),
                self.determinant.reshape(100, 100),
                zorder=0.0,
                levels=50,
                cmap="viridis",
            )
            plt.colorbar()
            # plt.title("Background: Determinant")
        elif mode == "uncertainty":
            plt.contourf(
                self.meshgrid[:, 0].reshape(100, 100),
                self.meshgrid[:, 1].reshape(100, 100),
                self.uncertainty.reshape(100, 100),
                zorder=0.0,
                levels=50,
                cmap="viridis",
            )
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("right", "5%", pad="25%")
            cbar = plt.colorbar(cax=cax)
            cbar.ax.yaxis.set_ticks_position('left')
            cbar.ax.tick_params(labelsize=int(plot_fontsize*0.5))
            cbar.set_label(r'Uncertainty', loc="center", rotation=270, labelpad=plot_fontsize)

        elif mode == "indicatrix":
            if self.n_ensemble is not None:
                for i in range(self.n_ensemble):
                    fig, ax = self.plot_indicatrices(self.meshgrid, self.indicatrices[i], (fig, ax))
            else:
                fig, ax = self.plot_indicatrices(self.meshgrid, self.indicatrices, (fig, ax))
        else:
            pass
        
        ax.set_xlim(right=3.0)  # xmax is your value
        ax.set_xlim(left=-3.0)  # xmin is your value
        ax.set_ylim(top=3.0)  # ymax is your value
        ax.set_ylim(bottom=-3.0)  # ymin is your value
        
        #box = ax.get_position()
        #ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        ax.axis("equal")
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            fancybox=True,
            shadow=True,
            ncol=len(self.labels)+1,
            markerscale=3.0,
            fontsize = int(plot_fontsize*0.8),
            columnspacing=0.5
        )
        ax.set_title(r"Geodesic path $\gamma$ in the latent space", fontsize = plot_fontsize)
        ax.grid(False)
        ax.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
        plt.tight_layout()
        
        
        return fig

    def latents_data(self, data_set, key: Array):
        logging.info("👉 Computing latents for the dataset...")

        @eqx.filter_jit
        def _encode(batch, key):
            return self.model.encode(batch, key)[0]

        latents = []
        labels = []
        for i, batch in enumerate(data_set):
            _k, key  = split(key,2)
            latents.append(_encode(batch[0],_k))
            labels.append(np.argmax(batch[1]))

        self.latents = np.array(latents)
        self.labels = np.array(labels)

        logging.info("👍 Latents computed successfully")

    def metric_indicatrices(self, metrics):
        logging.info("👉 Computing indicatrices in the latent space...")
        _, eigvecs = jnp.linalg.eigh(metrics)

        @eqx.filter_jit
        def gnorm(metric, eigenvectors):
            norm = jnp.sqrt(jnp.einsum("ij,ij->j", eigenvectors, jnp.dot(metric, eigenvectors)))
            eigenvectors = eigenvectors / norm
            return eigenvectors

        eigvecs = jax.vmap(gnorm, in_axes=(0, 0))(metrics, eigvecs)

        @eqx.filter_jit
        def indicatrix(eigvecs):
            t = jnp.linspace(-jnp.pi, jnp.pi, 100)
            indicatrix = jnp.array([jnp.cos(t), jnp.sin(t)])
            indicatrix = jnp.dot(eigvecs, indicatrix)
            return indicatrix

        indicatrices = jax.vmap(indicatrix, in_axes=0)(eigvecs)
        return indicatrices

    def plot_indicatrices(self, Z, indicatrices, plot=None):
        # Z gives a meshgrid of points
        # indicatrices gives the indicatrices at each of the points in the meshgrid
        selection = np.linspace(10, 90, 8).astype(int)
        # repeat 5 times and add 100 to each of the repeats
        selection = np.concatenate([selection + 500, selection + 3000, selection + 5500, selection + 8000])
        indicatrices = indicatrices[selection, :, :]
        if plot is None:
            plt.close()
            plt.style.use("fast")
            fig, ax = plt.subplots()
        else:
            fig, ax = plot

        # at each point in the meshgrid, plot the indicatrix
        for i, indicatrix in enumerate(indicatrices):
            ax.plot(
                indicatrix[0, :] + Z[selection[i], 0],
                indicatrix[1, :] + Z[selection[i], 1],
                linestyle="--",
                color=COLORS[i],
                linewidth=0.5,
                alpha=0.1 if plot is not None else 0.9,
            )
            # plot the point
            if plot is None:
                ax.scatter(Z[selection[i], 0], Z[selection[i], 1], c=COLORS[i], s=2)
        ax.axis("equal")
        plt.xlim(right=3.0)  # xmax is your value
        plt.xlim(left=-3.0)  # xmin is your value
        plt.ylim(top=3.0)  # ymax is your value
        plt.ylim(bottom=-3.0)  # ymin is your value
        if plot is None:
            plt.grid(True)

        return fig, ax

    def metric_determinants(self):
        logging.info("👉 Computing metric determinant in the latent space...")
        x = np.linspace(-5.0, 5.0, 100)
        y = np.linspace(-5.0, 5.0, 100)
        X, Y = np.meshgrid(x, y)

        # make them a 2D array
        Z = np.array([X.flatten(), Y.flatten()]).T
        Z = jnp.array(Z)

        # batch Zs by 10 for self.manifold.ensemble_metric calculation
        if self.n_ensemble is not None:
            logging.info(f"👉 Batching metric determinant computation due to ensembling ...")
            Z_split = jnp.array_split(Z, 10)
            metrics = []
            for subarray in Z_split:
                metric = self.manifold.ensemble_metric(subarray)
                metrics.append(metric)

            metric = jnp.concatenate(metrics)
            determinants = jax.vmap(jax.vmap(jnp.linalg.det, in_axes=0), in_axes=0)(metric)
            determinant_meaned = jnp.mean(determinants, axis=1)
            indicatrices_per_ensamble = jax.vmap(self.metric_indicatrices, in_axes=1)(metric)
        else:
            metric = self.manifold.metric(Z)
            determinant = jnp.linalg.det(metric)
            indicatrices = self.metric_indicatrices(metric)
        ## combine meshgrid and determinant to return
        self.meshgrid = np.array(Z)
        self.determinant = np.array(determinant) if self.n_ensemble is None else np.array(determinant_meaned)
        self.determinants = np.array(determinants) if self.n_ensemble is not None else None
        self.indicatrices = np.array(indicatrices) if self.n_ensemble is None else np.array(indicatrices_per_ensamble)
        logging.info(
            f"👍 Determinants computed successfully max: {np.max(self.determinant,axis=None)} min: {np.min(self.determinant,axis=None)}"
            if self.n_ensemble is None
            else f"👍 Determinants computed successfully max: {jnp.max(determinants,axis=None).item()} min: {jnp.min(determinants,axis=None).item()}"
        )

        figures_determinants = [self.plot(Z, self.determinant, "Determinant landscape overall")]
        figures_indicatrices = []

        if self.n_ensemble is not None:
            plt.close()
            plt.style.use("fast")
            plots_in = plt.subplots()
            for i in range(self.n_ensemble):
                figures_determinants.append(self.plot(Z, determinants[:, i], f"Determinant landscape decoder {i+1}"))
                plots_in = self.plot_indicatrices(Z, indicatrices_per_ensamble[i], plots_in)
                figures_indicatrices.append(self.plot_indicatrices(Z, indicatrices_per_ensamble[i])[0])
            figures_indicatrices = [plots_in[0]] + figures_indicatrices
        else:
            figures_indicatrices.append(self.plot_indicatrices(Z, self.indicatrices)[0])
        return figures_determinants, figures_indicatrices

    def plot(self, linspace, linspaced_quantity, title):
        plt.close()
        plt.style.use("fast")
        fig, ax = plt.subplots()

        for i, label in enumerate(np.unique(self.labels)):
            idx = np.where(self.labels == label)[0]
            ax.scatter(self.latents[idx, 0], self.latents[idx, 1], c=COLORS[i], marker=MARKERS[i], label=label, s=2)

        # add meshgrid and determinant as contour plot
        plt.contourf(
            linspace[:, 0].reshape(100, 100),
            linspace[:, 1].reshape(100, 100),
            linspaced_quantity.reshape(100, 100),
            zorder=0.0,
            levels=50,
            cmap="viridis",
        )
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "10%", pad="25%")
        cbar = plt.colorbar(cax=cax)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.tick_params(labelsize=int(plot_fontsize*0.5))
        cbar.set_label(r'Uncertainty', loc="center", rotation=270,  labelpad=plot_fontsize)

        plt.xlim(right=3.0)  # xmax is your value
        plt.xlim(left=-3.0)  # xmin is your value
        plt.ylim(top=3.0)  # ymax is your value
        plt.ylim(bottom=-3.0)  # ymin is your value
        #plt.grid(False)
        ax.axis("equal")
        #box = ax.get_position()
        #ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            fancybox=True,
            shadow=True,
            ncol=len(self.labels),
            markerscale=3.0,
        )

        ax.set_title(f"Test set encoded into the latent space")
        ax.grid(False)
        ax.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
        plt.tight_layout()
        return fig

    def compute_uncertainty(self, mode="mean"):
        x = np.linspace(-5.0, 5.0, 100)
        y = np.linspace(-5.0, 5.0, 100)
        X, Y = np.meshgrid(x, y)

        # make them a 2D array
        Z = np.array([X.flatten(), Y.flatten()]).T
        Z = jnp.array(Z)

        rbf_precisions = jax.vmap(self.rbf)(Z)
        variances = jax.vmap(jax.vmap(lambda x: 1/x))(rbf_precisions)
        print(f"varshape: {variances.shape}")
        uncertainty = {
            "mean": jnp.mean,
            "max": jnp.max,
        }[
            mode
        ](jnp.sqrt(variances), axis=1)

        self.meshgrid = Z
        self.uncertainty = np.array(uncertainty)

        return self.plot(Z, self.uncertainty, "Uncertainties")

    def init_optimizer(self, params):
        # grad_transformations = [optax.clip_by_global_norm(1.0)]

        # lr_schedule = optax.warmup_cosine_decay_schedule(
        #    init_value=0.0,
        #    peak_value=self.geodesics_lr,
        #    warmup_steps=self.warmup_steps,
        #    decay_steps=self.n_steps,
        #    end_value=self.geodesics_lr*0.1
        # )

        # grad_transformations.append(
        #    load_obj(self.geodesics_optimizer)(self.geodesics_lr, **self.geodesics_optimizer_params)
        # )

        # self.optimizer = optax.chain(*grad_transformations)
        optimizer = load_obj(self.geodesics_optimizer)(self.geodesics_lr, **self.geodesics_optimizer_params)

        opt_state = optimizer.init(params)
        return optimizer, opt_state
    
    def load_rbf(self):
        # Load model. We use different checkpoint for pretrained models

        W = jax.nn.softplus(jax.random.normal(PRNGKey(0), (self.flat_dim, self.config.model.rbf_k)))
        centroids = jax.random.normal(PRNGKey(0), (self.config.model.rbf_k, self.config.model.latent_dim))
        lambdas = jax.random.normal(PRNGKey(0), (self.config.model.rbf_k,))
        c = jnp.ones(self.flat_dim)*0.1
        rbf = RBF(W, centroids, lambdas, c)


        self.rbf = eqx.tree_deserialise_leaves(self.model_checkpoint_path + "model_rbf.eqx", rbf)
        logging.info(f"👉🏻 Loaded RBF from {self.model_checkpoint_path}...")
