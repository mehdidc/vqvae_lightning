import yaml
import os
from functools import partial
from clize import run
from argparse import Namespace
import numpy as np
from imageio import imsave
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torchvision

from data import Shuffle
from data import load_dataset
from data import SubSet
from model import VQAE

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule


class Model(LightningModule):
    
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.dataset = self.load_dataset(hparams)
        self.model = self.build_model(hparams)

    def load_dataset(self, hparams):
        dataset = load_dataset(
            hparams.dataset_folder, 
            image_size=hparams.image_size, 
            nb_channels=hparams.nb_channels
        )
        if hparams.nb_examples is not None:
            dataset = Shuffle(dataset)
            dataset = SubSet(dataset, hparams.nb_examples)
        return dataset

    def forward(self, x):
        return self.model(x)

    def build_model(self, hparams):
        return VQAE(
            num_hiddens=hparams.num_hiddens,
            num_residual_layers=hparams.num_residual_layers,
            num_residual_hiddens=hparams.num_residual_hiddens,
            num_embeddings=hparams.num_embeddings,
            embedding_dim=hparams.embedding_dim,
            commitment_cost=hparams.commitment_cost,
            decay=hparams.decay,
            nb_channels=hparams.nb_channels,
            nb_blocks=int(np.log2(hparams.image_size//hparams.stride)),
        )
    
    def training_step(self, batch, batch_idx):
        self.save_grids("train")
        X, _ = batch
        commit_loss, XR, perplexity = self.model(X)
        recons_loss = F.mse_loss(X, XR)
        loss = recons_loss + commit_loss
        output = OrderedDict({"loss": loss, "log": {"loss": loss},})
        return output

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )
    
    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = lr_scheduler.ExponentialLR(
            optimizer, gamma=self.hparams.scheduler_gamma
        )
        return [optimizer], [scheduler]
    
    def on_epoch_end(self):
        self.save_grids("train")
    
    def save_grids(self, split):
        if split == "train":
            loader = self.train_dataloader()
        elif split == "valid":
            loader = self.valid_dataloader()
        else:
            raise ValueError(split)

        X, _ = next(iter(loader))
        X = X.to(self.device)
        commit_loss, XR, perplexity = self.model(X)
        X = X.data.cpu()
        XR = XR.data.cpu()
        
        X_grid = torchvision.utils.make_grid(X)
        XR_grid = torchvision.utils.make_grid(XR)
        self.logger.experiment.add_image('inputs', X_grid, self.current_epoch)
        self.logger.experiment.add_image('reconstructions', XR_grid, self.current_epoch)

        X_grid = X_grid.permute(1,2,0).numpy()
        XR_grid = XR_grid.permute(1,2,0).numpy()
        grid = np.concatenate((X_grid, XR_grid), axis=1)
        grid = (grid*255).astype("uint8")
        imsave(os.path.join(self.hparams.folder, f"{split}_rec.png"), grid)


def train(hparams_path):
    hparams = load_hparams(hparams_path)
    os.makedirs(hparams.folder, exist_ok=True)
    model = Model(hparams)
    logger = pl.loggers.TensorBoardLogger(
        save_dir=hparams.folder,
        name='logs'
    )
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

if __name__ == "__main__":
    run([train])