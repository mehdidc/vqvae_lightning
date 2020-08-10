from functools import reduce
from collections import OrderedDict
from copy import deepcopy
import time
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
from data import DatasetWithIndices

from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel, BertLMHeadModel, BertTokenizer
# from transformers import RobertaConfig, EncoderDecoderConfig, EncoderDecoderModel, RobertaLMHeadModel, RobertaTokenizer

from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.parsing import AttributeDict


class Model(pl.LightningModule):
    def __init__(self, hparams, load_dataset=True, build_model=True, force_load_dataset=False, encoder_use_cache=True):
        super().__init__()
        if type(hparams) == dict:
            hparams = AttributeDict(hparams)
        self.tokenizer = self.build_tokenizer()
        if load_dataset:
            self.dataset = self.load_dataset(hparams, force=force_load_dataset)
            hparams.vocab_size = self.dataset.vocab_size
            hparams.height, hparams.width = self.dataset.shape[1:]
            hparams.max_length = self.dataset.length
            hparams.start_token = self.dataset.start_token
            hparams.encoder_max_length = self.dataset.encoder_max_length
        self.hparams = hparams
        if build_model:
            self.model = self.build_model()
        if hparams.fixed_encoder and encoder_use_cache == True:
            print("Caching encoder outputs")
            conds = self.dataset.dataset.tensors[1]
            outputs = None
            # bs = hparams.batch_size
            bs = 128
            device = "cuda" if hparams.gpus else "cpu"
            enc = self.model.encoder.to(device)
            for start in range(0, len(conds), bs):
                cond = conds[start:start+bs].to(device)
                with torch.no_grad():
                    encoder_outputs = enc(cond)
                if outputs is None:
                    outputs = [[o.data.cpu()] for o in encoder_outputs]
                else:
                    for i, o in enumerate(encoder_outputs):
                        outputs[i].append(o.data.cpu())
                print(start, "/", len(conds))
            outputs = [torch.cat(o) for o in outputs]
            self.encoder_outputs = outputs
            print("Finished caching encoder outputs")
                
    def load_dataset(self, hparams, force=False):
        print("Loading the dataset of codes into memory...")
        device = "cuda" if hparams.gpus else "cpu"
        vqvae = VQVAE.load_from_checkpoint(hparams.vqvae_model_path)
        vqvae = vqvae.to(device)
        self.vqvae = vqvae
        nb = 0
        path = os.path.join(os.path.dirname(hparams.folder), "code_dataset.th")
        print(path)
        if os.path.exists(path) and not force:
            conds, codes, max_length = torch.load(path)
            if hparams.nb_examples and len(codes) >= hparams.nb_examples:
                codes = codes[: hparams.nb_examples]
                conds = conds[: hparams.nb_examples]
            print("Loaded dataset from cache")
        else:
            max_length = 0
            print("Finding max conditional length")
            # vqvae.hparams.batch_size =  256
            # vqvae.hparams.num_workers = 16
            size = len(vqvae.dataset)
            start = time.time()
            for X, Y in vqvae.train_dataloader(shuffle=False):
                if isinstance(Y, torch.Tensor):
                    Y = Y.tolist()
                    Y = list(map(str, Y))
                Y = list(Y)
                Y = self.tokenizer.batch_encode_plus(Y)
                Y = Y["input_ids"]
                max_length = max(max_length, max(map(len, Y)))
                if hparams.nb_examples and nb >= hparams.nb_examples:
                    break
                nb += len(Y)
                if nb % 100 == 0:
                    imps = (time.time() - start) / nb
                    remaining = imps * (size-nb)
                    print(nb, imps, remaining) 
            nb = 0
            print(Y)
            print("MAX LENGTH:", max_length)
            conds = []
            codes = []
            start = time.time()
            for X, Y in vqvae.train_dataloader(shuffle=False):
                if isinstance(Y, torch.Tensor):
                    Y = Y.tolist()
                    Y = list(map(str, Y))
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
                conds.append(Y)
                zinds = vqvae.encode(X)
                codes.append(zinds.data.cpu())
                nb += len(zinds)
                if hparams.nb_examples and nb >= hparams.nb_examples:
                    break
                if nb % 100 == 0:
                    imps = (time.time() - start) / nb
                    remaining = imps * (size-nb)
                    print(nb, imps, remaining) 
            conds = torch.cat(conds)
            codes = torch.cat(codes)
            if hparams.nb_examples and len(codes) >= hparams.nb_examples:
                codes = codes[: hparams.nb_examples]
                conds = conds[: hparams.nb_examples]
            if rank_zero_only.rank == 0:
                torch.save((conds, codes, max_length), path)

        vocab_size = vqvae.model.num_embeddings + 1#added one because of start token
        start_token = vqvae.model.num_embeddings
        codes_ = codes.view(len(codes), -1)
        codes_ = torch.cat(
            [(torch.ones(len(codes_), 1) * start_token).long(), codes_,], dim=1
        )
        length = codes_.shape[1]
        print(codes_.shape, conds.shape)
        dataset = DatasetWithIndices(TensorDataset(codes_, conds))
        dataset.vocab_size = vocab_size
        dataset.shape = codes.shape
        dataset.length = length
        dataset.start_token = start_token
        dataset.encoder_max_length = max_length
        print("Done loading dataset")
        return dataset

    def forward(self, x):
        return self.model(x)
    
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
    
    def build_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        return tokenizer
    
    def build_decoder(self, hparams, tokenizer):
        path = os.path.join(self.hparams.folder, "decoder")
        decoder_config = BertConfig(
            is_decoder=True,
            vocab_size=hparams.vocab_size,
            hidden_size=hparams.hidden_size,
            num_hidden_layers=hparams.num_hidden_layers,
            num_attention_heads=hparams.num_attention_heads,
            intermediate_size=hparams.intermediate_size,
            hidden_dropout_prob=hparams.hidden_dropout_prob,
            attention_dropout_prob=hparams.attention_dropout_prob,
            max_position_embeddings=hparams.max_length,
        )
        decoder = BertLMHeadModel(decoder_config)
        decoder.save_pretrained(path)
        return decoder

    def build_model(self):
        path = os.path.join(self.hparams.folder, "decoder")
        model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", path)
        return model

    def generate(self, cond, temperature=1.0, do_sample=True, top_k=0, top_p=None):
        result = generate_with_constraints(
            deepcopy(self.model).to(cond.device), # necesary for multigpu setting otherwise breaks,
            forbid=[self.hparams.start_token],
            input_ids=cond, 
            decoder_start_token_id=self.hparams.start_token,
            temperature=temperature,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            max_length=self.hparams.max_length,
        )
        result = result[:, 1:]
        result = result.contiguous()
        result = result.view(result.shape[0], self.hparams.height, self.hparams.width)
        return result

    def training_step(self, batch, batch_idx):
        inds, X, Y = batch
        if self.hparams.fixed_encoder:
            encoder_outputs = tuple([o[inds].to(X.device) for o in self.encoder_outputs])
        else:
            encoder_outputs = self.model.encoder(Y)
        loss, *rest = self.model(
            decoder_input_ids=X,
            labels=X,
            encoder_outputs=encoder_outputs,
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
        # we handle scheduling in optimizer_step, it's easier
        scheduler = lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=1.0,
        )
        return [optimizer], [scheduler]

    def optimizer_step(
            self, 
            current_epoch, 
            batch_idx, 
            optimizer,
            optimizer_idx, 
            second_order_closure, 
            using_native_amp
    ):
        # warm up lr
        total_steps = self.hparams.epochs * self.trainer.num_training_batches
        if self.hparams.warmup_ratio is not None:
            warm = self.hparams.warmup_ratio * total_steps
        else:
            warm = self.hparams.warmup_iter
        if self.trainer.global_step < warm:
            lr_scale = min(1., float(self.trainer.global_step + 1) / warm)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.lr
        else:
            if self.trainer.global_step == warm:
                print("Finished warmup")
                print("Schedule now: ", self.hparams.learning_rate_scheduler)
            # polynomial decay with power 2
            if self.hparams.learning_rate_scheduler == "poly":
                decay = (1 - ((self.trainer.global_step - warm)/ total_steps)) ** 2
                for pg in optimizer.param_groups:
                    pg['lr'] = self.hparams.lr * decay
            else:
                pass
        # update params
        optimizer.step()
        optimizer.zero_grad() 
        
    @rank_zero_only
    def on_epoch_end(self):
        print("epoch end")
        if self.current_epoch % self.hparams.save_every == 0:
            folder = self.hparams.folder
            self.trainer.save_checkpoint(os.path.join(folder, "model.th"))
            if not self.hparams.generate:
                return
            inds, X, Y = next(iter(self.train_dataloader(shuffle=True)))
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
            out = os.path.join(self.hparams.folder, f"true_epoch_{self.trainer.current_epoch:05d}.png")
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
            out = os.path.join(self.hparams.folder, f"gen_epoch_{self.trainer.current_epoch:05d}.png")
            torchvision.utils.save_image(images, out, nrow=nrow)

def generate_with_constraints(model, forbid, *args, **kwargs):
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
