import random
import os
import numpy as np
from skimage.transform import resize
from glob import glob
from PIL import Image
import io

import torch.nn as nn
import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision.datasets import ImageFolder, MNIST
import torchvision.transforms as transforms

from torchvision.datasets.folder import default_loader


class CacheDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = {}

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        else:
            point = self.dataset[idx]
            self.cache[idx] = point
            return point

    def __len__(self):
        return len(self.dataset)


def load_dataset(
    dataset_name,
    split="train",
    image_size=32,
    crop_size=None,
    random_crop_size=None,
    nb_channels=3,
    invert=False,
    dataset_type="image_folder",
):
    if dataset_name == "mnist":
        dataset = MNIST(
            root=("mnist"),
            download=True,
            transform=transforms.Compose(
                [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
            ),
        )
        return dataset
    elif "quickdraw" in dataset_name:
        dataset = QuickDrawDataset(dataset_name)
        return dataset
    else:
        if dataset_type == "image_folder":
            cls = ImageFolder
        elif dataset_type == "lmdb":
            cls = LMDB
        else:
            raise ValueError(dataset_type)
        tfs = []
        if crop_size is not None:
            tfs.append(transforms.CenterCrop(crop_size))
        tfs.append(transforms.Resize((image_size, image_size)))
        if random_crop_size is not None:
            tfs.append(transforms.RandomCrop(random_crop_size))
        tfs.append(transforms.ToTensor())
        if nb_channels == 1:
            tfs.append(Gray())
        if invert:
            tfs.append(Invert())
        dataset = cls(root=dataset_name, transform=transforms.Compose(tfs))
        return dataset

class LMDB:

    
    def __init__(self, root, transform=None, target_transform=None):
        import lmdb
        #https://github.com/pytorch/vision/blob/master/torchvision/datasets/lsun.py
        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.nb = int(txn.get("nb".encode()))
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        with self.env.begin(write=False) as txn:
            xkey = f"image-{i}".encode()
            ykey = f"labelstr-{i}".encode()
            target = txn.get(ykey)
            target = target.decode()
            # assert target, (ykey, target)
            imgbuf = txn.get(xkey)
        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return self.nb

class QuickDrawDataset:
    def __init__(self, path):
        self.data = np.load(path)

    def __getitem__(self, i):
        x = self.data[i]
        x = x.reshape((28, 28))
        x = resize(x, (32, 32), preserve_range=True)
        x = x / 255.0
        x = torch.from_numpy(x)
        x = x.unsqueeze(0)
        x = x.float()
        y = 0
        return x, y

    def __len__(self):
        return len(self.data)


class Invert:
    def __call__(self, x):
        return 1 - x


class Gray:
    def __call__(self, x):
        return x[0:1]


class SubSet:
    def __init__(self, dataset, nb):
        self.dataset = dataset
        self.nb = nb

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return self.nb


class Shuffle:
    def __init__(self, dataset, seed=42):
        self.dataset = dataset
        if hasattr(dataset, "classes"):
            self.classes = dataset.classes
        rng = np.random.RandomState(seed)
        inds = np.arange(len(dataset))
        rng.shuffle(inds)
        self.inds = inds

    def __getitem__(self, ind):
        return self.dataset[self.inds[ind]]

    def __len__(self):
        return len(self.dataset)


class DatasetWithIndices:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return (idx,) + self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, path
