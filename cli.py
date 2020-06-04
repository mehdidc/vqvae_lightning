from argparse import Namespace
import yaml
import os
from clize import run
import pytorch_lightning as pl
from vqvae import Model as VQVAE
from generator import Model as Generator

def train_vqvae(hparams_path):
    hparams = load_hparams(hparams_path)
    os.makedirs(hparams.folder, exist_ok=True)
    model = VQVAE(hparams)
    logger = pl.loggers.TensorBoardLogger(save_dir=hparams.folder, name="logs")
    trainer = pl.Trainer(
        default_root=hparams.folder,
        max_epochs=hparams.epochs,
        show_progress_bar=False,
        gpus=hparams.gpus,
        logger=logger,
    )
    trainer.fit(model)


def load_hparams(path):
    return Namespace(**yaml.load(open(path).read()))


def train_transformer_generator(hparams_path):
    hparams = load_hparams(hparams_path)
    model = Generator(hparams)
    logger = pl.loggers.TensorBoardLogger(save_dir=hparams.folder, name="logs")
    trainer = pl.Trainer(
        default_root=hparams.folder,
        max_epochs=hparams.epochs,
        show_progress_bar=False,
        gpus=hparams.gpus,
        logger=logger,
    )
    trainer.fit(model)

if __name__ == "__main__":
    run([train_vqvae, train_transformer_generator])
