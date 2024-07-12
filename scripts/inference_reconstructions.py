import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf, open_dict
import os
import jax.numpy as jnp
from jax import config
import yaml
import matplotlib.pyplot as plt
import warnings
import numpy as np
import jax

warnings.filterwarnings("ignore", module="matplotlib\..*")

config.update("jax_debug_nans", True)

import wandb


from ensertainty.utils import set_seed, load_obj, save_useful_info, init_outer_transform, init_decoder_ensamble
from ensertainty.data import get_dataloaders
from jax import random
import pandas as pd
import logging


def run_experiment(cfg: DictConfig, wandb_logger, test_set, test_loader, iters=16):
    set_seed(cfg["training"]["seed"])
    random_key = random.PRNGKey(cfg["training"]["seed"])

    key, key1, key2, key3, key4, random_key = random.split(random_key, 6)

    if cfg["model"]["class_name"].split(".")[-1] == "ManifoldAE":
        model = load_obj(cfg["model"]["class_name"])(
            cfg["model"]["data_dim"],
            cfg["model"]["latent_dim"],
            init_outer_transform(cfg, key3),
        )
    elif cfg["model"]["class_name"].split(".")[-1] == "EncoderManifoldAE":
        encoder = load_obj(cfg["model"]["encoder_class_name"])(**cfg["model"]["encoder_params"], key=key4)
        model = load_obj(cfg["model"]["class_name"])(
            cfg["model"]["data_dim"], cfg["model"]["latent_dim"], init_outer_transform(cfg, key3), encoder
        )
    elif cfg["model"]["class_name"].split(".")[-1] == "EnsambleManifoldAE":
        encoder = load_obj(cfg["model"]["encoder_class_name"])(**cfg["model"]["encoder_params"], key=key4)
        decoders = init_decoder_ensamble(cfg, key3)
        model = load_obj(cfg["model"]["class_name"])(
            cfg["model"]["data_dim"], cfg["model"]["latent_dim"], cfg["model"]["n_ensemble"], decoders, encoder
        )
    else:
        raise ValueError("Model class name not recognized")

    trainer = load_obj(cfg["inference"]["class_name"])(model, cfg, wandb_logger)

    trainer.load_model(True)
    outputs = []
    for iter, batch in enumerate(test_loader):
        normfunc = lambda image1, image2: jnp.linalg.norm(jnp.ravel(image1) - jnp.ravel(image2))
        b1, b2 = batch["image"][: batch["image"].shape[0] // 2], batch["image"][-batch["image"].shape[0] // 2 :]
        logging.info(f"b1 shape: {b1.shape} and b2 shape {b2.shape}")
        norms = jax.vmap(normfunc, in_axes=(0, 0))(b1, b2)
        logging.info(f"\n NORMS in AMBIENT are:\n {norms}\n")
        reconstrunctions, losses = trainer.reconstruct(trainer.model, batch, "mean")
        print(reconstrunctions.shape)
        logging.info(f"losses: {losses}")
        outputs.append((batch["image"], reconstrunctions, batch["label"]))
        if iter == iters:
            break
    #if cfg["model"]["class_name"].split(".")[-1] == "EnsambleManifoldAE":
    #    trainer.latents_data(test_set)
    #    for mode in ["mean", "max"]:
    #        figure = trainer.plot_latent_uncertainty(mode)
    #        wandb.log(
    #            {f"{(cfg.training.checkpoint).split('/')[-2]}_latent_space_uncertainty_{mode}": wandb.Image(figure)}
    #        )

    return outputs


def main(cfg: DictConfig):
    wandb_logger = wandb.init(
        project=cfg["general"]["project_name"],
        name=cfg["general"]["run_name"],
        entity=cfg["general"]["workspace"],
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        handlers=[logging.FileHandler(f"{wandb_logger.dir}/pythonlog.txt"), logging.StreamHandler()],
    )

    experiments = list(
        zip(cfg.inference.checkpoints, cfg.inference.checkpoints_models, cfg.inference.checkpoints_states)
    )
    outputs = []
    for experiment in experiments:
        config = OmegaConf.load(experiment[0] + "config.yml")
        config.training.checkpoint = experiment[0]
        config.training.checkpoint_model = experiment[1]
        config.training.checkpoint_state = experiment[2]
        with open_dict(config):
            config.inference = cfg.inference

        config["datamodule"]["batch_size"] = 16

        test_loader, test_set = get_dataloaders(
            config["datamodule"],
            19821,
            inference_mode=True,
        )

        outputs.append(run_experiment(config, wandb_logger, test_set, test_loader))

    outputs_processed = []

    for i, output in enumerate(outputs):
        processed = [
            (experiments[i], plot_images((jnp.squeeze(res[0]), jnp.squeeze(res[1])), res[2])) for res in output
        ]
        outputs_processed.append(processed)

    os.makedirs(cfg["general"]["model_checkpoints_path"], exist_ok=True)

    for i, output in enumerate(outputs_processed):
        for j, (experiment, fig) in enumerate(output):
            fig.savefig(f'{cfg["general"]["model_checkpoints_path"]}{experiment[0].split("/")[-2]}_batch_{j}.png')
            wandb.log({f"{experiment[0].split('/')[-2]}_batch_{j}": wandb.Image(fig)})

    wandb.finish()


def plot_images(input, labels):
    """
    Plot the input images and the reconstructed images

    Args:
        input: Tuple of batches of images, the first one is the batch of input imags and the second one is the reconstructed images

    Returns:
        A figure with the input and reconstructed images
    """
    batch_len = len(input[0])
    plt.close()
    fig, axs = plt.subplots(2, batch_len, figsize=(batch_len, 2))
    for i in range(batch_len):
        axs[0, i].imshow(np.rot90(np.transpose(input[0][i],(1,2,0)),k=3))
        axs[0, i].axis("off")
        axs[1, i].imshow(np.rot90(np.transpose(input[1][i],(1,2,0)), k=3))
        axs[1, i].axis("off")
    # add labels to the images
    for i in range(batch_len):
        axs[0, i].set_title(f"Label: {np.argmax(labels[i])}")
        axs[1, i].set_title(f"Label: {np.argmax(labels[i])}")

    ## add headers too:
    axs[0, batch_len // 2].set_title("Input")
    axs[1, batch_len // 2].set_title("Reconstructed")

    return fig


@hydra.main(config_path="../configs", config_name="config_reconstructions_inference")
def launch(cfg: DictConfig):
    os.makedirs("logs", exist_ok=True)
    print(OmegaConf.to_yaml(cfg))
    if cfg.general.log_code:
        save_useful_info(os.path.basename(__file__))

    main(cfg)


if __name__ == "__main__":
    launch()
