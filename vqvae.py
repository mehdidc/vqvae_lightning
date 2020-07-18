# Credits to:
# - https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
# - the above notebook is itself is based on SONNET from deepmind
from collections import namedtuple
import yaml
import os
from functools import partial
from argparse import Namespace
from collections import OrderedDict
import math
from clize import run

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
from torch.utils.data import TensorDataset

from data import Shuffle
from data import load_dataset
from data import SubSet

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from srgan import SRResNet

class VectorQuantizerEMA(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5
    ):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        batch_size, height, width, channels = input_shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )

            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        encoding_indices = encoding_indices.view(batch_size, height, width)
        # convert quantized from BHWC -> BCHW
        return (
            loss,
            quantized.permute(0, 3, 1, 2).contiguous(),
            perplexity,
            encoding_indices,
        )


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_residual_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.InstanceNorm2d(num_residual_hiddens),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=num_residual_hiddens,
                out_channels=num_hiddens,
                kernel_size=1,
                stride=1,
            ),
            nn.InstanceNorm2d(num_residual_hiddens),
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(
        self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens
    ):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [
                Residual(in_channels, num_hiddens, num_residual_hiddens)
                for _ in range(self._num_residual_layers)
            ]
        )

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
        nb_downsample_blocks=2,
    ):
        super().__init__()
        layers = []
        for i in range(nb_downsample_blocks):
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels if i == 0 else num_hiddens,
                    out_channels=num_hiddens,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            layers.append(nn.InstanceNorm2d(num_hiddens))
            layers.append(nn.ReLU(True))
        self.downsample = nn.Sequential(*layers)
        self._conv = nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

    def forward(self, inputs):
        x = self.downsample(inputs)
        x = self._conv(x)
        return self._residual_stack(x)


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
        nb_upsample_blocks=2,
        out_channels=3,
        upsample_method="convtranspose2d",
    ):
        super().__init__()
        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )
        layers = []
        for i in range(nb_upsample_blocks):
            last = i == nb_upsample_blocks - 1
            out = out_channels if last else num_hiddens
            if upsample_method == "convtranspose2d":
                layers.append(
                    nn.ConvTranspose2d(
                        in_channels=num_hiddens,
                        out_channels=out,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    )
                )
            elif upsample_method in ("bilinear", "nearest"):
                # avoid checkerboard artifacts
                layers.append(nn.Upsample(scale_factor=2, mode=upsample_method,))
                layers.append(
                    nn.Conv2d(
                        in_channels=num_hiddens,
                        out_channels=out,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
            else:
                raise ValueError(upsample_method)
            if not last:
                layers.append(nn.InstanceNorm2d(out))
                layers.append(nn.ReLU(True))
        self.upsample = nn.Sequential(*layers)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._residual_stack(x)
        return self.upsample(x)


class VQVAE(nn.Module):
    def __init__(
        self,
        num_hiddens=128,
        num_residual_layers=2,
        num_residual_hiddens=32,
        num_embeddings=51,
        embedding_dim=64,
        commitment_cost=0.25,
        decay=0.99,
        nb_channels=3,
        nb_blocks=2,
        upsample_method="convtranspose2d",
    ):
        super().__init__()
        self._encoder = Encoder(
            nb_channels,
            num_hiddens,
            num_residual_layers,
            num_residual_hiddens,
            nb_downsample_blocks=nb_blocks,
        )
        self._pre_vq_conv = nn.Conv2d(
            in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1
        )
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(
                num_embeddings, embedding_dim, commitment_cost, decay
            )
        else:
            self._vq_vae = VectorQuantizer(
                num_embeddings, embedding_dim, commitment_cost
            )
        self._decoder = SRResNet(in_channels=embedding_dim, scaling_factor=2**nb_blocks)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        return loss, x_recon, perplexity

    def encode(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, encoding_indices = self._vq_vae(z)
        return encoding_indices

    def reconstruct_from_code(self, encoding_indices):
        batch_size, height, width = encoding_indices.shape
        encoding_indices = encoding_indices.view(-1)
        encodings = torch.nn.functional.one_hot(
            encoding_indices, num_classes=self._vq_vae._num_embeddings
        )
        encodings = encodings.float()
        quantized = torch.matmul(encodings, self._vq_vae._embedding.weight)
        quantized = quantized.view(
            batch_size, height, width, self._vq_vae._embedding_dim
        )
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        x_recon = self._decoder(quantized)
        return x_recon

    @property
    def num_embeddings(self):
        return self._vq_vae._num_embeddings


class Model(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.dataset = self.load_dataset(hparams)
        self.model = self.build_model(hparams)
        self.perceptual_loss = (
            Vgg(hparams.perceptual_loss) if hparams.perceptual_loss else None
        )

    def load_dataset(self, hparams):
        dataset = load_dataset(
            hparams.train_dataset_folder,
            image_size=hparams.image_size,
            nb_channels=hparams.nb_channels,
            dataset_type=hparams.dataset_type,
        )
        if hparams.nb_examples is not None:
            dataset = Shuffle(dataset)
            dataset = SubSet(dataset, hparams.nb_examples)
        return dataset

    def forward(self, x):
        return self.model(x)

    def encode(self, x):
        return self.model.encode(x)

    def reconstruct_from_code(self, code):
        return self.model.reconstruct_from_code(code)

    def build_model(self, hparams):
        return VQVAE(
            num_hiddens=hparams.num_hiddens,
            num_residual_layers=hparams.num_residual_layers,
            num_residual_hiddens=hparams.num_residual_hiddens,
            num_embeddings=hparams.num_embeddings,
            embedding_dim=hparams.embedding_dim,
            commitment_cost=hparams.commitment_cost,
            decay=hparams.decay,
            nb_channels=hparams.nb_channels,
            nb_blocks=int(math.log2(hparams.stride)),
            upsample_method=hparams.upsample_method,
        )

    def training_step(self, batch, batch_idx):
        X, _ = batch
        commit_loss, XR, perplexity = self.model(X)
        recons_loss = F.mse_loss(X, XR)
        if self.perceptual_loss:
            recons_loss += self.perceptual_loss(X, XR)
        loss = recons_loss + commit_loss
        output = OrderedDict(
            {
                "loss": loss,
                "log": {
                    "loss": loss,
                    "commit_loss": commit_loss,
                    "recons_loss": recons_loss,
                },
            }
        )
        return output

    def train_dataloader(self, shuffle=True):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
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

    @rank_zero_only
    def on_epoch_end(self):
        print("Epoch Finish")
        folder = self.hparams.folder
        if self.trainer.current_epoch % self.hparams.save_every == 0:
            self.trainer.save_checkpoint(os.path.join(folder, "model.th"))
            loader = self.train_dataloader(shuffle=False)
            self.save_grids(loader, f"train_rec_epoch_{self.trainer.current_epoch:05d}.png")

    def save_grids(self, loader, out):
        X, Y = next(iter(loader))
        X = X.to(self.device)
        commit_loss, XR, perplexity = self.model(X)
        X = X.data.cpu()
        XR = XR.data.cpu()
        nrow = int(math.sqrt(len(X)))
        X_grid = torchvision.utils.make_grid(X, nrow=nrow)
        XR_grid = torchvision.utils.make_grid(XR, nrow=nrow)
        grid = torch.cat((X_grid, XR_grid), dim=2)
        torchvision.utils.save_image(
            grid, os.path.join(self.hparams.folder, out)
        )

class Vgg(torch.nn.Module):
    # From https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/vgg.py
    def __init__(self, name="vgg16"):
        super().__init__()
        cls = getattr(torchvision.models, name)
        vgg_pretrained_features = cls(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for param in self.parameters():
            param.requires_grad = False
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def features(self, X):
        X = (X - self.mean) / self.std
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
        )
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

    def forward(self, x, y):
        fx = self.features(x)
        fy = self.features(y)
        return F.mse_loss(fx.relu1_2, fy.relu1_2)
