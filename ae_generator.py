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
        num_layers=2,
        embedding_dim=64,
        nb_channels=512,
        nb_blocks=2,
        noise_proba=0.3,
        noise_type="dropout",
        noise_level=0.1,
    ):
        super().__init__()
        self._encoder = conv_layers(
            in_channels=nb_channels,
            feature_map=num_hiddens,
            out_channels=num_hiddens,
            nb_layers=num_layers,
            cls=nn.Conv2d,
            kernel_size=5,
            # feature_map_mult=0.5,
        )
        self.embed = nn.Conv2d(num_hiddens, embedding_dim, (1,1), stride=1)
        self._decoder = conv_layers(
            in_channels=embedding_dim,
            out_channels=nb_channels,
            feature_map=num_hiddens,
            nb_layers=num_layers,
            cls=nn.ConvTranspose2d,
            kernel_size=5,
            # feature_map_mult=2,
        )
        print(self._encoder)
        print(self._decoder)
        self.nb_channels = nb_channels
        self.noise_proba = noise_proba
        self.noise_type = noise_type
        self.noise_level = noise_level

    def forward(self, x):
        z = self._encoder(x)
        z = self.embed(z)
        z = spatial_sparsity(z)
        x = self._decoder(z)
        return x
    
    def noise_and_forward(self, x):
        #x:nb,h,w
        with torch.no_grad():
            #nb,h,w,nb_channels
            x_onehot = torch.nn.functional.one_hot(x, num_classes=self.nb_channels)
            x_onehot = x_onehot.float()
            x_onehot = self.noise_(x_onehot)
            #nb,nb_channels,h,w
            x_onehot = x_onehot.permute(0,3,1,2).contiguous()
        xr = self.forward(x_onehot)
        return xr

    def loss(self, x):
        #x:nb,h,w
        #nb,nb_channels,h,w
        xr = self.noise_and_forward(x)
        #nb,h,w,nb_channels
        xr = xr.permute(0,2,3,1)
        xr = xr.contiguous()
        #nb*h*w,nb_channels
        xr = xr.view(-1,self.nb_channels)
        #nb*h*w
        x = x.view(-1)
        return nn.functional.cross_entropy(xr, x), xr, x
    
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
        elif self.noise_type == None:
            return x_onehot
        else:
            raise ValueError(self.noise_type)



def conv_layers(
    in_channels=3,
    out_channels=128,
    feature_map=64,
    feature_map_mult=1,
    norm_layer=None,
    act=nn.ReLU(True),
    nb_layers=1,
    kernel_size=5,
    padding=0,
    cls=nn.Conv2d,
):
    fms = (
        [in_channels]
        + [int(feature_map * (feature_map_mult ** i)) for i in range(nb_layers - 1)]
        + [out_channels]
    )
    layers = []
    bias = False if norm_layer == nn.BatchNorm2d else True
    for i, (prev, cur) in enumerate(zip(fms[0:-1], fms[1:])):
        layer = cls(prev, cur, kernel_size=kernel_size, padding=padding, bias=bias)
        layers.append(layer)
        if i< len(fms) - 2:
            if norm_layer:
                layers.append(norm_layer(cur))
            layers.append(act)
    return nn.Sequential(*layers)


class Model(pl.LightningModule):
    def __init__(self, hparams, load_dataset=True, nb_examples=None):
        super().__init__()
        if load_dataset:
            if nb_examples is not None:
                hparams.nb_examples = nb_examples
            self.dataset = self.load_dataset(hparams)
            hparams.nb_channels = self.dataset.nb_channels
            hparams.height = self.dataset.height
            hparams.width = self.dataset.width
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
        dataset.height = codes.shape[1]
        dataset.width = codes.shape[2]
        print("Done loading dataset")
        return dataset

    def forward(self, x):
        return self.model(x)

    def build_model(self, hparams):
        return AE(
            num_hiddens=hparams.num_hiddens,
            num_layers=hparams.num_layers,
            embedding_dim=hparams.embedding_dim,
            nb_channels=hparams.nb_channels,
            noise_proba=hparams.noise_proba,
            noise_type=hparams.noise_type,
            noise_level=hparams.noise_level,
        )

    def training_step(self, batch, batch_idx):
        (X,) = batch
        loss, xr_flat, x_flat = self.model.loss(X)
        _, xr_ind = xr_flat.max(dim=1)
        m = x_flat!=-100
        acc = (x_flat[m]==xr_ind[m]).float().mean()
        output = OrderedDict({"loss": loss, "acc": acc, "log": {"loss": loss, "acc": acc},})
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
    
    @torch.no_grad()
    def generate(self, nb_examples, nb_iter=10, init_from="random"):
        if init_from == "random":
            codes = torch.randint(0, self.hparams.nb_channels, size=(nb_examples, self.hparams.height, self.hparams.width))
        elif init_from == "dataset":
            dataloader = self.train_dataloader()
            codes, = next(iter(dataloader))
            codes = codes[:nb_examples]
        else:
            raise ValueError(init_from)
        iters = [codes]
        for _ in range(nb_iter):
            codes = self.model.noise_and_forward(codes)
            _, codes = codes.max(dim=1)
            iters.append(codes)
        codes = torch.cat(iters)
        return codes


def spatial_sparsity(x):
    xf = x.view(x.size(0), x.size(1), -1)
    m, _ = xf.max(2)
    m = m.view(m.size(0), m.size(1), 1)
    m = m.repeat(1, 1, xf.size(2))
    xf = xf * (xf == m).float()
    xf = xf.view(x.size())
    return xf


def channel_sparsity(x):
    xf = x.view(x.size(0), x.size(1), -1)
    m, _ = xf.max(1)
    m = m.view(m.size(0), 1, m.size(1))
    m = m.repeat(1, xf.size(1), 1)
    xf = xf * (xf == m).float()
    xf = xf.view(x.size())
    return xf



if __name__ == "__main__":
    model = Model(hparams)
