import math
from argparse import Namespace
import yaml
import os
from clize import run

import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateLogger

from vqvae import Model as VQVAE

from ae_generator import Model as AeGenerator
from transformer_generator import Model as TransformerGenerator


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


def train_transformer_generator(hparams_path, *, checkpoint=None):
    hparams = load_hparams(hparams_path)
    model = TransformerGenerator(hparams)
    trainer = pl.Trainer()
    if hparams.lr == 0:
        lr_finder = trainer.lr_find(model, num_training=20)
        new_lr = lr_finder.suggestion()
        print("Best LR: ", new_lr)
        model.hparams.lr = new_lr
    logger = pl.loggers.TensorBoardLogger(save_dir=hparams.folder, name="logs")
    trainer = pl.Trainer(
        default_root=hparams.folder,
        max_epochs=hparams.epochs,
        show_progress_bar=False,
        gpus=hparams.gpus,
        logger=logger,
        resume_from_checkpoint=checkpoint,
        callbacks=[LearningRateLogger()],
    )
    trainer.fit(model)

def train_ae_generator(hparams_path, *, checkpoint=None):
    hparams = load_hparams(hparams_path)
    model = AeGenerator(hparams)
    trainer = pl.Trainer()
    logger = pl.loggers.TensorBoardLogger(save_dir=hparams.folder, name="logs")
    trainer = pl.Trainer(
        default_root=hparams.folder,
        max_epochs=hparams.epochs,
        show_progress_bar=False,
        gpus=hparams.gpus,
        logger=logger,
        resume_from_checkpoint=checkpoint,
        callbacks=[LearningRateLogger()],
    )
    trainer.fit(model)


@torch.no_grad()
def transformer_generate(
    generator_model_path, *, device="cpu", nb_examples=1, out="out.png", temperature=1.
):
    gen = TransformerGenerator.load_from_checkpoint(generator_model_path, load_dataset=False,)
    gen = gen.to(device)
    vqvae = VQVAE.load_from_checkpoint(gen.hparams.vqvae_model_path)
    vqvae = vqvae.to(device)
    gen.eval()
    vqvae.eval()
    codes = gen.generate(
        nb_examples, 
        do_sample=True, 
        temperature=temperature,
        top_k=0,
    )
    images = vqvae.model.reconstruct_from_code(codes)
    nrow = int(math.sqrt(len(codes)))
    if (nrow**2) != len(codes):
        nrow = 8
    torchvision.utils.save_image(images, out, nrow=nrow)

@torch.no_grad()
def ae_generate(
    generator_model_path, *, device="cpu", nb_examples=1, nb_iter=10, out="out.png", init_from="random",
):
    load_dataset = True if init_from == "dataset" else False
    gen = AeGenerator.load_from_checkpoint(
        generator_model_path, 
        load_dataset=load_dataset,
        nb_examples=nb_examples,
    )
    gen = gen.to(device)
    vqvae = VQVAE.load_from_checkpoint(gen.hparams.vqvae_model_path)
    vqvae = vqvae.to(device)
    gen.eval()
    vqvae.eval()
    codes = gen.generate(nb_examples, nb_iter, init_from=init_from)
    images = vqvae.model.reconstruct_from_code(codes)
    nrow = int(math.sqrt(len(codes)))
    if (nrow**2) != len(codes):
        nrow = 8
    torchvision.utils.save_image(images, out, nrow=nrow)

@torch.no_grad()
def reconstruct(
    model_path, *, device="cpu", nb_examples=1, out="out.png", temperature=1.
):
    vqvae = VQVAE.load_from_checkpoint(model_path)
    vqvae = vqvae.to(device)
    vqvae.eval()
    X, _ = next(iter(vqvae.train_dataloader()))
    X = X.to(vqvae.device)
    commit_loss, XR, perplexity = vqvae.model(X)
    X = X.data.cpu()
    XR = XR.data.cpu()
    nrow = int(math.sqrt(len(X)))
    X_grid = torchvision.utils.make_grid(X, nrow=nrow)
    XR_grid = torchvision.utils.make_grid(XR, nrow=nrow)
    grid = torch.cat((X_grid, XR_grid), dim=2)
    torchvision.utils.save_image(grid, out)

def load_hparams(path):
    return Namespace(**yaml.load(open(path).read()))


if __name__ == "__main__":
    run([train_vqvae, train_transformer_generator, transformer_generate, reconstruct, train_ae_generator, ae_generate])
