from functools import reduce
from collections import OrderedDict
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import ReformerModelWithLMHead, ReformerConfig

import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset
from torch import optim

from vqvae import Model as VQVAE


class Model(pl.LightningModule):
    def __init__(self, hparams, load_dataset=True):
        super().__init__()
        if load_dataset:
            self.dataset = self.load_dataset(hparams)
            hparams.vocab_size = self.dataset.vocab_size
            hparams.height, hparams.width = self.dataset.shape[1:]
            hparams.max_length = self.dataset.length
            hparams.start_token = self.dataset.start_token
            hparams.pad_token = self.dataset.pad_token
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
        vocab_size = vqvae.model.num_embeddings + 2
        start_token = vqvae.model.num_embeddings
        pad_token = vqvae.model.num_embeddings+1
        codes_ = codes.view(len(codes), -1)
        codes_ = torch.cat(
            [(torch.ones(len(codes_), 1) * start_token).long(), codes_,], dim=1
        )
        length = codes_.shape[1]
        if hparams.model_type == "reformer":
            total_length = (codes_.shape[1]//64+1)*64
            pad_length = total_length - codes_.shape[1]
            codes_ = torch.cat(
                [codes_, (torch.ones(len(codes_), pad_length) * pad_token).long()], dim=1
            )
        dataset = TensorDataset(codes_)
        dataset.vocab_size = vocab_size
        dataset.shape = codes.shape
        dataset.length = length
        dataset.start_token = start_token
        dataset.pad_token = pad_token
        print("Done loading dataset")
        return dataset

    def forward(self, x):
        return self.model(x)

    def build_model(self, hparams):
        if hparams.model_type == "gpt2":
            config = GPT2Config(
                vocab_size=hparams.vocab_size,
                n_positions=hparams.max_length,
                n_ctx=hparams.max_length,
                n_embd=512,
                n_layer=4,
                n_head=1,
                resid_pdrop=0,
                embd_pdrop=0,
                attn_pdrop=0,
            )
            return GPT2LMHeadModel(config)
        elif hparams.model_type == "reformer":
            total_length = (hparams.max_length//64+1)*64
            config = ReformerConfig(
                vocab_size=hparams.vocab_size,
                num_attention_heads=1,
                max_position_embeddings=total_length,
                attention_head_size=64,
                feed_forward_size=512,
                pad_token_id=hparams.pad_token,
                axial_pos_shape=(64,total_length//64),
            )
            return ReformerModelWithLMHead(config)
        else:
            raise ValueError(hparams.mode_type)

    def generate(self, nb_examples=1, **kwargs):
        input_ids = torch.zeros(nb_examples, 1).long().to(self.device)
        input_ids[:] = self.hparams.vocab_size - 1
        result = self.model.generate(
            input_ids, max_length=self.hparams.max_length, **kwargs,
        )
        result = result[:, 1:]
        result = result.contiguous()
        result = result.view(nb_examples, self.hparams.height, self.hparams.width)
        return result

    def training_step(self, batch, batch_idx):
        (X,) = batch
        loss, *rest = self.model(X, labels=X)
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
