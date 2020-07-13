from functools import reduce
from collections import OrderedDict
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Model, GPT2Tokenizer
from unittest.mock import patch
import os
import math

import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset
from torch import optim
import torchvision

from vqvae import Model as VQVAE


class Model(pl.LightningModule):
    def __init__(self, hparams, load_dataset=True):
        super().__init__()
        self.tokenizer = self.build_tokenizer()
        if load_dataset:
            self.dataset = self.load_dataset(hparams)
            hparams.vocab_size = self.dataset.vocab_size
            hparams.height, hparams.width = self.dataset.shape[1:]
            hparams.max_length = self.dataset.length
            hparams.start_token = self.dataset.start_token
            hparams.encoder_max_length = self.dataset.encoder_max_length
        self.model = self.build_model(hparams, self.tokenizer)
        self.encoder = self.build_encoder_pretrained(hparams, self.tokenizer)
        self.hparams = hparams

    def load_dataset(self, hparams):
        print("Loading the dataset of codes into memory...")
        device = "cuda" if hparams.gpus else "cpu"
        vqvae = VQVAE.load_from_checkpoint(hparams.vqvae_model_path)
        vqvae = vqvae.to(device)
        self.vqvae = vqvae
        nb = 0
        max_length = 0
        for X, Y in vqvae.train_dataloader(shuffle=False):
            Y = list(Y)
            Y = self.tokenizer.batch_encode_plus(Y)
            Y = Y["input_ids"]
            max_length = max(max_length, max(map(len, Y)))
            if hparams.nb_examples and nb >= hparams.nb_examples:
                break
            nb += len(Y)
        nb = 0
        print("MAX LENGTH:", max_length)
        conds = []
        codes = []
        for X, Y in vqvae.train_dataloader(shuffle=False):
            X = X.to(device)
            Y = list(Y)
            Y = self.tokenizer.batch_encode_plus(
                Y, 
                padding=True, 
                pad_to_multiple_of=max_length, 
                max_length=max_length
            )
            Y = Y["input_ids"]
            Y = torch.Tensor(Y).long()
            print(Y.shape)
            conds.append(Y)
            zinds = vqvae.encode(X)
            codes.append(zinds.data.cpu())
            nb += len(zinds)
            if hparams.nb_examples and nb >= hparams.nb_examples:
                break
        conds = torch.cat(conds)
        codes = torch.cat(codes)
        if hparams.nb_examples and len(codes) >= hparams.nb_examples:
            codes = codes[: hparams.nb_examples]
            conds = conds[: hparams.nb_examples]
        vocab_size = vqvae.model.num_embeddings + 1
        start_token = vqvae.model.num_embeddings
        codes_ = codes.view(len(codes), -1)
        codes_ = torch.cat(
            [(torch.ones(len(codes_), 1) * start_token).long(), codes_,], dim=1
        )
        length = codes_.shape[1]
        print(codes_.shape, conds.shape)
        dataset = TensorDataset(codes_, conds)
        dataset.vocab_size = vocab_size
        dataset.shape = codes.shape
        dataset.length = length
        dataset.start_token = start_token
        dataset.encoder_max_length = max_length
        print("Done loading dataset")
        return dataset

    def forward(self, x):
        return self.model(x)
    
    def build_tokenizer(self):
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.add_special_tokens({"cls_token": "<CLS>", "pad_token": "<PAD>", "eos_token": "<EOS>"})
        return tokenizer

    def build_encoder_pretrained(self, hparams, tokenizer):
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.resize_token_embeddings(tokenizer.vocab_size + 3)
        model.tokenizer = tokenizer
        return model
    
    def build_encoder(self, hparams, tokenizer):
        config = GPT2Config(
            vocab_size=tokenizer.vocab_size+3,
            n_positions=hparams.encoder_max_length,
            n_ctx=hparams.encoder_max_length,
            n_embd=hparams.n_embd if hasattr(hparams, "n_embd") else 512,
            n_layer=hparams.n_layer if hasattr(hparams, "n_layer") else 4,
            n_head=hparams.n_head if hasattr(hparams, "n_head") else 1,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
            summary_first_dropout=0,
        )
        model = GPT2LMHeadModel(config)
        model.tokenizer = tokenizer
        return model


    def build_model(self, hparams, tokenizer):
        config = GPT2Config(
            vocab_size=hparams.vocab_size,
            n_positions=hparams.encoder_max_length+hparams.max_length,
            n_ctx=hparams.encoder_max_length+hparams.max_length,
            n_embd=hparams.n_embd if hasattr(hparams, "n_embd") else 512,
            n_layer=hparams.n_layer if hasattr(hparams, "n_layer") else 4,
            n_head=hparams.n_head if hasattr(hparams, "n_head") else 1,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
            summary_first_dropout=0,
        )
        return GPT2LMHeadModel(config)
    
    def encode_cond(self, cond):
        Y = self.tokenizer.batch_encode_plus(
            cond, 
            padding=True, 
            pad_to_multiple_of=self.hparams.encoder_max_length, 
            max_length=self.hparams.encoder_max_length
        )
        Y = Y["input_ids"]
        Y = torch.Tensor(Y).long()
        return Y

    def generate(self, cond):
        result = generate(
            encoder=self.encoder,
            model=self.model, 
            cond=cond,
            max_length=self.hparams.max_length,
            start_token=self.hparams.start_token,
            forbid=[self.hparams.start_token],
        )
        # input_ids = torch.zeros(cond.shape[0], 1).long().to(cond.device)
        # input_ids[:] = self.hparams.start_token
        # result = generate_old(
            # self.model,
            # forbid=[self.hparams.start_token],
            # input_ids=input_ids,
            # max_length=self.hparams.max_length,
            # temperature=1.0,
            # do_sample=True,
            # top_k=0,
            # encoder_outputs=cond,
        # )
        result = result[:, 1:]
        result = result.contiguous()
        result = result.view(result.shape[0], self.hparams.height, self.hparams.width)
        return result

    def training_step(self, batch, batch_idx):
        X, Y = batch
        _, context_features = self.encoder(Y)
        loss, *rest = self.model(
            input_ids=X, 
            past=context_features, 
            labels=X
        )
        output = OrderedDict({"loss": loss, "log": {"loss": loss,},})
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
    
    def on_epoch_end(self):
        if self.current_epoch % self.hparams.save_every == 0:
            folder = self.hparams.folder
            self.trainer.save_checkpoint(os.path.join(folder, "model.th"))
            X, Y = next(iter(self.train_dataloader(shuffle=False)))
            nb = 9
            X = X[:nb]
            Y = Y[:nb]
            
            X = X[:,1:]
            X = X.contiguous()
            X = X.view(len(X), self.hparams.height, self.hparams.width)
            X = X.to(self.device)
            images = self.vqvae.model.reconstruct_from_code(X)
            nrow = int(math.sqrt(len(X)))
            if (nrow ** 2) != len(X):
                nrow = 8
            out = os.path.join(self.hparams.folder, "true.png")
            torchvision.utils.save_image(images, out, nrow=nrow)

                
            print("Generating..")
            Y = Y.to(self.device)
            codes = self.generate(Y)
            print(codes.min(), codes.max())
            print("Reconstructing...")
            images = self.vqvae.model.reconstruct_from_code(codes)
            nrow = int(math.sqrt(len(codes)))
            if (nrow ** 2) != len(codes):
                nrow = 8
            out = os.path.join(self.hparams.folder, "gen.png")
            torchvision.utils.save_image(images, out, nrow=nrow)

@torch.no_grad()
def generate(
    encoder,
    model,
    cond,
    forbid=None,
    max_length=10, 
    start_token=0,
    temperature=1.0,
):
    device = cond.device
    nb, source_length = cond.shape
    target = torch.empty(nb, max_length).long().to(device)
    target[:, 0] = start_token 
    _, context_features = encoder(cond)
    for t in range(1, max_length):
        outputs = model(
            input_ids=target[:, 0:t],
            past=context_features,
        )
        target_pred = outputs[0]
        target_pred = target_pred[:, -1]
        target_pred /= temperature
        if forbid is not None:
            for ind in forbid:
                target_pred[:,ind] = -1e7
        target_pred = target_pred.softmax(dim=1)
        target_sample = torch.multinomial(target_pred, 1)[:, 0]
        target[:, t] = target_sample
    return target

def generate_old(model, forbid, *args, **kwargs):
    orig = model.forward
    def fwd(*args, **kwargs):
        y, *rest = orig(*args, **kwargs)
        for ind in forbid:
            y[:,:,ind] = -1e7
        return (y,) + tuple(rest)
    model.forward = fwd
    y = model.generate(*args, **kwargs)
    model.forward = orig
    return y

