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

from transformer_generator import Model as TransformerGenerator

def train_vqvae(hparams_path, *, checkpoint=None):
    hparams = load_hparams(hparams_path)
    os.makedirs(hparams.folder, exist_ok=True)
    model = VQVAE(hparams)
    logger = pl.loggers.TensorBoardLogger(save_dir=hparams.folder, name="logs")
    trainer = pl.Trainer(
        default_root_dir=hparams.folder,
        max_epochs=hparams.epochs,
        show_progress_bar=False,
        gpus=hparams.gpus,
        distributed_backend=hparams.distributed_backend,
        logger=logger,
        resume_from_checkpoint=checkpoint,
    )
    model.trainer = trainer
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
        default_root_dir=hparams.folder,
        max_epochs=hparams.epochs,
        show_progress_bar=False,
        gpus=hparams.gpus,
        distributed_backend=hparams.distributed_backend,
        logger=logger,
        resume_from_checkpoint=checkpoint,
        callbacks=[LearningRateLogger()],
    )
    model.trainer = trainer
    trainer.fit(model)


@torch.no_grad()
def transformer_generate(
    generator_model_path, 
    *, 
    device="cpu", 
    nb_examples=1,
    temperature=1.0,
    top_k=0,
    top_p=None,
    folder=None,
    single_cond=False,
    custom_cond=None,
):
    gen = TransformerGenerator.load_from_checkpoint(
        generator_model_path,
        load_dataset=True,
    )
    gen = gen.to(device)
    gen.eval()
    gen.vqvae.eval()
    if folder is None:
        folder = gen.hparams.folder
    X, Y = next(iter(gen.train_dataloader(shuffle=True)))
    print(X.shape, Y.shape)
    if single_cond:
        X = X[0:1].repeat(nb_examples, 1)
        if custom_cond:
            Y = gen.encode_cond([custom_cond])
        Y = Y[0:1].repeat(nb_examples, 1)
    else:
        X = X[:nb_examples]
        Y = Y[:nb_examples]
    for y in gen.tokenizer.batch_decode(Y.tolist()):
        print(y)
    X = X[:,1:]#remove start token
    X = X.contiguous()
    X = X.view(len(X), gen.hparams.height, gen.hparams.width)
    X = X.to(gen.device)
    images = gen.vqvae.model.reconstruct_from_code(X)
    nrow = int(math.sqrt(len(X)))
    if (nrow ** 2) != len(X):
        nrow = 8
    out = os.path.join(folder, "generate_true.png")
    torchvision.utils.save_image(images, out, nrow=nrow)

    print("Generating..")
    Y = Y.to(gen.device)
    codes = gen.generate(
        Y,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    print(codes.min(), codes.max())
    print("Reconstructing...")
    images = gen.vqvae.model.reconstruct_from_code(codes)
    nrow = int(math.sqrt(len(codes)))
    if (nrow ** 2) != len(codes):
        nrow = 8
    out = os.path.join(folder, "generate_gen.png")
    torchvision.utils.save_image(images, out, nrow=nrow)



@torch.no_grad()
def reconstruct(
    model_path, 
    *, 
    device="cpu", 
    nb_examples=1, 
    folder=None,
    label=None,
):
    vqvae = VQVAE.load_from_checkpoint(model_path)
    vqvae = vqvae.to(device)
    vqvae.eval()
    if folder is None:
        folder = vqvae.hparams.folder
    out = os.path.join(folder, "rec.png")
    nb = 0
    xs = []
    for X, Y in vqvae.train_dataloader(shuffle=True):
        if label is not None:
            inds = [i for i, y in enumerate(Y) if y == label]
            xs.append(X[inds])
            nb += len(inds)
        else:
            print(Y)
            xs.append(X)
            nb += len(X)
        print(nb)
        if nb >= nb_examples:
            break
    X = torch.cat(xs)
    X = X[:nb_examples]
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
    run([
        train_vqvae, 
        train_transformer_generator, 
        transformer_generate, 
        reconstruct
    ])
