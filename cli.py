from argparse import Namespace
import yaml
import os
from clize import run


import torch
import torchvision
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


@torch.no_grad()
def generate(
    generator_model_path, *, device="cpu", nb_examples=1, out="out.png", temperature=1.
):
    gen = Generator.load_from_checkpoint(generator_model_path, load_dataset=False,)
    gen = gen.to(device)
    vqvae = VQVAE.load_from_checkpoint(gen.hparams.vqvae_model_path)
    vqvae = vqvae.to(device)
    gen.eval()
    vqvae.eval()
    codes = gen.generate(nb_examples, do_sample=True, temperature=temperature,)
    images = vqvae.model.reconstruct_from_code(codes)
    torchvision.utils.save_image(images, out)


if __name__ == "__main__":
    run([train_vqvae, train_transformer_generator, generate])
