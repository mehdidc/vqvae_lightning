import math
from argparse import Namespace
import yaml
import os
from functools import partial
from clize import run

import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.callbacks.base import Callback

from vqvae import Model as VQVAE
from transformer_generator import Model as TransformerGenerator

import torch
torch.backends.cudnn.benchmark = True


def train_vqvae(hparams_path, *, checkpoint=None):
    if checkpoint is not None and not os.path.exists(checkpoint):
        checkpoint = None
        print("provided checkpoint does not exist, starting from scratch")
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
        callbacks=[LearningRateLogger()],
    )
    model.trainer = trainer
    trainer.fit(model)

def transformer_generator_build_decoder(hparams_path):
    hparams = load_hparams(hparams_path)
    model = TransformerGenerator(hparams, build_model=False, encoder_use_cache=False)
    model.build_decoder(hparams, model.tokenizer)


def train_transformer_generator(hparams_path, *, checkpoint=None):
    if checkpoint is not None and not os.path.exists(checkpoint):
        checkpoint = None
        print("provided checkpoint does not exist, starting from scratch")
    hparams = load_hparams(hparams_path)
    model = TransformerGenerator(hparams)
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
        gradient_clip_val=hparams.gradient_clip_val,
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
    batch_size=16,
    top_k=0,
    top_p:float=None,
    folder=None,
    single_cond=False,
    custom_cond=None,
):
    gen = TransformerGenerator.load_from_checkpoint(
        generator_model_path,
        load_dataset=True,
        encoder_use_cache=False,
    )
    gen = gen.to(device)
    gen.eval()
    gen.vqvae.eval()
    if folder is None:
        folder = gen.hparams.folder
    os.makedirs(folder, exist_ok=True)
    xs = []
    ys = []
    nb = 0
    for inds, X, Y in gen.train_dataloader(shuffle=True):
        xs.append(X)
        ys.append(Y)
        nb += len(X)
        if nb >= nb_examples:
            break
    X = torch.cat(xs)
    Y = torch.cat(ys)
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
    images = _batch_run(gen.vqvae.model.reconstruct_from_code, X, batch_size)
    nrow = int(math.sqrt(len(X)))
    if (nrow ** 2) != len(X):
        nrow = 8
    out = os.path.join(folder, "generate_true.png")
    torchvision.utils.save_image(images, out, nrow=nrow)
    print("Generating..")
    Y = Y.to(gen.device)
    gen_func = partial(
        gen.generate,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    codes = _batch_run(gen_func, Y, batch_size)
    print(codes.min(), codes.max())
    print("Reconstructing...")
    images = _batch_run(gen.vqvae.model.reconstruct_from_code, codes, batch_size, device=gen.device)
    nrow = int(math.sqrt(len(codes)))
    if (nrow ** 2) != len(codes):
        nrow = 8
    out = os.path.join(folder, "generate_gen.png")
    torchvision.utils.save_image(images, out, nrow=nrow)


@torch.no_grad()
def transformer_generate_from_text(
    generator_model_path, 
    textfile,
    *, 
    device="cpu", 
    nb_examples=1,
    temperature=1.0,
    batch_size=16,
    top_k=0,
    top_p:float=None,
    folder=None,
):
    gen = TransformerGenerator.load_from_checkpoint(
        generator_model_path,
        load_dataset=True,
        encoder_use_cache=False,
    )
    gen = gen.to(device)
    gen.eval()
    gen.vqvae.eval()
    if folder is None:
        folder = os.path.join(gen.hparams.folder, "gen")
    os.makedirs(folder, exist_ok=True)
    for custom_cond in open(textfile).readlines():
        custom_cond = custom_cond.replace("\n", "")
        Y = gen.encode_cond([custom_cond])
        Y = Y[0:1].repeat(nb_examples, 1)
        print(f"Generating from {custom_cond}..")
        Y = Y.to(gen.device)
        gen_func = partial(
            gen.generate,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        codes = _batch_run(gen_func, Y, batch_size)
        print(codes.min(), codes.max())
        print("Reconstructing...")
        images = _batch_run(
            gen.vqvae.model.reconstruct_from_code, 
            codes, 
            batch_size, 
            device=gen.device
        )
        nrow = int(math.sqrt(len(codes)))
        if (nrow ** 2) != len(codes):
            nrow = 8
        name = custom_cond.replace(",", "_")
        name = name.replace(" ", "_")
        out = os.path.join(folder, name+".png")
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

def _batch_run(func, X, batch_size, device=None):
    ys = []
    for i in range(0, len(X), batch_size):
        x = X[i:i+batch_size]
        if device is not None:
            x = x.to(device)
        y = func(x)
        y = y.data.cpu()
        ys.append(y)
    Y = torch.cat(ys)
    return Y

def load_hparams(path):
    return Namespace(**yaml.load(open(path).read()))


if __name__ == "__main__":
    run([
        train_vqvae, 
        train_transformer_generator, 
        transformer_generator_build_decoder, 
        transformer_generate, 
        transformer_generate_from_text,
        reconstruct
    ])
