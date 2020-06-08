import math
from collections import OrderedDict

import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset
from torch import optim

from vqvae import Encoder, Decoder
from vqvae import Model as VQVAE
import pytorch_lightning as pl

class AE(nn.Module):
    def __init__(
        self,
        num_hiddens=128,
        num_residual_layers=2,
        num_residual_hiddens=32,
        embedding_dim=64,
        nb_channels=512,
        nb_blocks=2,
        upsample_method="convtranspose2d",
        noise_proba=0.3,
        noise_type="dropout",
        noise_level=0.1,
    ):
        super().__init__()
        self._encoder = Encoder(
            nb_channels,
            num_hiddens,
            num_residual_layers,
            num_residual_hiddens,
            nb_downsample_blocks=nb_blocks,
        )
        self.embed = nn.Conv2d(num_hiddens, embedding_dim, (1,1), stride=1)
        self._decoder = Decoder(
            embedding_dim,
            num_hiddens,
            num_residual_layers,
            num_residual_hiddens,
            out_channels=nb_channels,
            nb_upsample_blocks=nb_blocks,
            upsample_method=upsample_method,
        )
        self.nb_channels = nb_channels
        self.noise_proba = noise_proba
        self.noise_type = noise_type
        self.noise_level = noise_level

    def forward(self, x):
        z = self._encoder(x)
        z = self.embed(z)
        x = self._decoder(z)
        return x

    def loss(self, x):
        with torch.no_grad():
            #nb,h,w,nb_channels
            x_onehot = torch.nn.functional.one_hot(x, num_classes=self.nb_channels)
            x_onehot = x_onehot.float()
            while torch.rand(1).item() <= self.noise_proba:
                self.noise_(x_onehot)
            #nb,nb_channels,h,w
            x_onehot = x_onehot.permute(0,3,1,2).contiguous()
        xr = self.forward(x_onehot)
        #nb,h,w,nb_channels
        xr = xr.permute(0,2,3,1)
        xr = xr.contiguous()
        #nb*h*w,nb_channels
        xr = xr.view(-1,self.nb_channels)
        #nb*h*w
        x = x.view(-1)
        return nn.functional.cross_entropy(xr, x)
    
    def noise_(self, x_onehot):
        #shape: (nb,h,w,self.nb_channels)
        if self.noise_type == "dropout":
            shape = x_onehot.shape
            x_onehot = x_onehot.view(-1, self.nb_channels)
            drop_mask = (torch.rand(x_onehot.shape[0]) <= self.noise_level).to(x_onehot.device)
            x_onehot[drop_mask] = 0
            x_onehot = x_onehot.view(shape)
            return x_onehot
        elif self.noise_type == "random_replace":
            shape = x_onehot.shape
            x_onehot = x_onehot.view(-1, self.nb_channels)
            drop_mask = (torch.rand(x_onehot.shape[0]) <= self.noise_level).to(x_onehot.device)
            rand = torch.randint(0, self.nb_channels, size=(x_onehot.shape[0],))
            rand = torch.nn.functional.one_hot(rand, num_classes=self.nb_channels)
            rand = rand.float()
            rand = rand.to(x_onehot.device)
            x_onehot[drop_mask] = rand[drop_mask]
            x_onehot = x_onehot.view(shape)
            return x_onehot
        else:
            raise ValueError(self.noise_type)

class Model(pl.LightningModule):
    def __init__(self, hparams, load_dataset=True):
        super().__init__()
        if load_dataset:
            self.dataset = self.load_dataset(hparams)
            hparams.nb_channels = self.dataset.nb_channels
        self.model = self.build_model(hparams)
        self.hparams = hparams

    def load_dataset(self, hparams):
        print("Loading the dataset of codes into memory...")
        device = "cuda" if hparams.gpus else "cpu"
        vqvae = VQVAE.load_from_checkpoint(hparams.vqvae_model_path)
        vqvae = vqvae.to(device)
        codes = []
        nb = 0
        for X, _ in vqvae.train_dataloader():
            X = X.to(device)
            zinds = vqvae.encode(X)
            codes.append(zinds.data.cpu())
            nb += len(zinds)
            if hparams.nb_examples and nb >= hparams.nb_examples:
                break
        codes = torch.cat(codes)
        if hparams.nb_examples and len(codes) >= hparams.nb_examples:
            codes = codes[:hparams.nb_examples]
        dataset = TensorDataset(codes)
        dataset.nb_channels = vqvae.model.num_embeddings
        print("Done loading dataset")
        return dataset

    def forward(self, x):
        return self.model(x)

    def build_model(self, hparams):
        return AE(
            num_hiddens=hparams.num_hiddens,
            num_residual_layers=hparams.num_residual_layers,
            num_residual_hiddens=hparams.num_residual_hiddens,
            embedding_dim=hparams.embedding_dim,
            nb_channels=hparams.nb_channels,
            nb_blocks=int(math.log2(hparams.stride)),
            upsample_method=hparams.upsample_method,
            noise_type=hparams.noise_type,
            noise_level=hparams.noise_level,
        )

    def training_step(self, batch, batch_idx):
        (X,) = batch
        loss = self.model.loss(X)
        output = OrderedDict({"loss": loss, "log": {"loss": loss,},})
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

if __name__ == "__main__":
    model = Model(hparams)
