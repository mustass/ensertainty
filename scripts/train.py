import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import os
from jax import config
import yaml

#config.update("jax_debug_nans", True)
#config.update("jax_disable_jit", True)
# config.update("jax_enable_x64", True)
import wandb

from ensertainty.utils import set_seed, load_obj, save_useful_info, init_decoder_ensamble
from ensertainty.data import get_dataloaders
from jax import random

import logging


def main(cfg: DictConfig, pretrained=False):
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

    set_seed(cfg["training"]["seed"])
    random_key = random.PRNGKey(cfg["training"]["seed"])

    key, key1, key2, key3, key4, random_key = random.split(random_key, 6)
    train_loader, val_loader, test_loader = get_dataloaders(
        cfg["datamodule"],
        19821,
    )

    if cfg["model"]["class_name"].split(".")[-1] == "EnsembleVAE":
        encoder = load_obj(cfg["encoder"]["class_name"])(**cfg["encoder"]["params"], key=key4)
        decoders = init_decoder_ensamble(cfg, key3)
        model = load_obj(cfg["model"]["class_name"])(
             encoder, decoders, cfg["model"]["num_decoders"],  cfg["model"]["latent_dim"],  cfg["model"]["kl_weight"]
        )
    else:
        raise ValueError("Model class name not recognized")

    trainer = load_obj(cfg["training"]["class_name"])(model, cfg, wandb_logger)

    if pretrained:
        trainer.load_model(pretrained)

    key, key1, key2, key3, random_key = random.split(random_key, 5)

    if not (pretrained and cfg.model.train_rbf):
        trainer.train_model(train_loader, val_loader, random_key=key, num_epochs=cfg["training"]["max_epochs"])
        trainer.load_model()

        val_acc = trainer.eval_model(val_loader, random_key=key1, epoch=cfg["training"]["max_epochs"], eval_type="val_set")
        test_acc = trainer.eval_model(
        test_loader, random_key=key2, epoch=cfg["training"]["max_epochs"], eval_type="test_set"
        )
        results = {"val": val_acc, "test": test_acc}
        print(results)

    if cfg.model.train_rbf:
        trainer.train_rbf(trainer.model, train_loader, key3, k = cfg.model.rbf_k, alpha = cfg.model.rbf_alpha, epochs=cfg.model.rbf_epochs)
        trainer.save_rbf()
        trainer.load_rbf()

    wandb.finish()


@hydra.main(config_path="../configs", config_name="config")
def launch(cfg: DictConfig):
    os.makedirs("logs", exist_ok=True)
    logging.info(OmegaConf.to_yaml(cfg))
    if cfg.general.log_code:
        save_useful_info(os.path.basename(__file__))

    cfg.general.model_checkpoints_path = os.path.join(get_original_cwd(), cfg.general.model_checkpoints_path)
    pretrained = False

    if cfg.training.checkpoint is not None:
        logging.info(f"Resuming training from checkpoint {cfg.training.checkpoint}")
        config = OmegaConf.load(cfg.training.checkpoint + "config.yml")
        config.training = cfg.training
        cfg = config
        pretrained = True
        logging.info(f"Running with new config:")
        logging.info(cfg)

    main(cfg, pretrained)


if __name__ == "__main__":
    launch()
