from functools import reduce
from collections import OrderedDict
from transformers import GPT2LMHeadModel, GPT2Config
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset
from torch import optim

from vqvae import Model as VQVAE

class Model(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.dataset = self.load_dataset(hparams)
        self.model = self.build_model(hparams)

    def load_dataset(self, hparams):
        vqvae = VQVAE.load_from_checkpoint(hparams.vqvae_model_path)
        codes = []
        for X, _ in vqvae.train_dataloader():
            X = X.to(vqvae.device)
            zinds = vqvae.encode(X)
            codes.append(zinds.data.cpu())
        codes = torch.cat(codes)
        vocab_size = vqvae.model.num_embeddings
        dataset = TensorDataset(codes)
        dataset.vocab_size = vocab_size
        dataset.shape = codes.shape
        dataset.length = reduce(lambda x,y:x*y, codes.shape[1:])
        return dataset

    def forward(self, x):
        return self.model(x)

    def build_model(self, hparams):
        config = GPT2Config(
            vocab_size=self.dataset.vocab_size,
            n_positions=self.dataset.length,
            n_ctx=self.dataset.length,
            n_embd=512,
            n_layer=1,
            n_head=1,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
        )
        model = GPT2LMHeadModel(config)
        return model

    def generate(self, nb_examples):
        input_ids = torch.zeros(nb_examples).long().to(self.device)
        input_ids[:] = start
        result = self.model.generate(
            input_ids,
            do_sample=True,
            max_length=self.length,
        )
        height, width = self.dataset.shape[1:]
        result = result.view(nb_examples, height, width)
        return result


    def training_step(self, batch, batch_idx):
        X, = batch
        X = X.view(len(X), -1)
        loss, *rest = self.model(X, labels=X)
        output = OrderedDict(
            {
                "loss": loss,
                "log": {
                    "loss": loss,
                },
            }
        )
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
