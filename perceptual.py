import torch.nn as nn
from functools import reduce
from collections import OrderedDict
from copy import deepcopy
from torchvision.models import resnet18
from unittest.mock import patch
import os
import math

import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
import torch
from torch import optim
import torchvision

from vqvae import Model as VQVAE

class Model(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        vqvae = VQVAE.load_from_checkpoint(hparams.vqvae_model_path)
        self.vqvae = vqvae
        self.clf = Perceptual()
    
    def training_step(self, batch, batch_idx):
        X, Y = batch
        with torch.no_grad():
            commit_loss, XR, perplexity = self.vqvae(X)
        yt, yf, loss = self.forward(X, XR)
        acc = (((yt>0.5).float().mean()) + (yf<0.5).float().mean()) * 0.5
        output = OrderedDict({"loss": loss, "log": {"loss": loss, "acc": acc},})
        return output
    
    def forward(self, x, y):
        fx = self.clf(x)
        fy = self.clf(y)
        lx = ((fx - 1)**2).mean()
        ly = ((fy - 0)**2).mean()
        return fx, fy, lx + ly

    def train_dataloader(self, shuffle=True):
        return self.vqvae.train_dataloader(shuffle=shuffle)

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
        torch.save(self.clf, os.path.join(self.hparams.folder, "model.th"))

class Perceptual(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.base = resnet18(pretrained=True)
        self.base.fc = nn.Linear(512, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.base(x)

    def loss(self, x, xr):
        fxr = self(xr)
        return ((fxr - 1)**2).mean()
