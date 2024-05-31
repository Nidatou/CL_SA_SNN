import os

import numpy as np
import torch
from torch.utils.data import Dataset
from spikingjelly.datasets.n_mnist import NMNIST


class ReducedDataset(Dataset):
    """A Dataset that only samples correspoinding to provided indices.
    That can be used for splitting a dataset into a training and validation set"""

    def __init__(self, origin_ds, indices):
        super(ReducedDataset, self).__init__()
        self.dataset = origin_ds
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


class SubDataset(Dataset):
    """a sub-sampled dataset, that takes only samples with label in [sub_labels].
    And after the selection, it is also possible to transform(shift) the target-labels,
    which brings convenience for some kinds of continual learning"""

    def __init__(self, origin_ds, sub_labels, target_transform=None):
        super(SubDataset, self).__init__()
        self.dataset = origin_ds
        self.sub_indices = []
        for index in range(len(self.dataset)):
            if hasattr(origin_ds, "targets"):
                if self.dataset.target_transform is None:
                    label = self.dataset.targets[index]
                else:
                    label = self.dataset.target_transform(self.dataset.targets[index])
            else:
                label = self.dataset[index][1]
            if label in sub_labels:
                self.sub_indices.append(index)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sub_indices)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indices[index]]
        if self.target_transform is not None:
            target = self.target_transform(sample[1])
            sample = (sample[0], target)
        return sample


class TransformedDataset(Dataset):
    """to modify an existing dataset with a transform
    This is useful for creating different permutations"""

    def __init__(self, original_dataset, transform=None, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        (input_x, target) = self.dataset[index]
        if self.transform:
            input_x = self.transform(input_x)
        if self.target_transform:
            target = self.target_transform(target)
        return (input_x, target)


class EmbeddingDataset(Dataset):
    def __init__(self, data_root, dataset_suffix, transform=None, target_transform=None, train=True):
        self.dataset_suffix = dataset_suffix
        train_or_test = 'train' if train else 'test'
        self.data_pth = os.path.join(data_root, self.dataset_suffix + train_or_test + '.pt')
        self.embeddings, self.labels = torch.load(self.data_pth)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        if self.transform:
            embedding = self.transform(embedding)
        if self.target_transform:
            label = self.target_transform(label)

        return embedding, label


# class Fixed_NMNIST(Dataset):
#     def __init__(self, root, train, download=True, transform=None, target_transform=None, T=16):
#         super(Fixed_NMNIST, self).__init__()
#         self.ori_dataset = NMNIST(
#             root, train=train, data_type='frame', frames_number=T,
#             split_by='number', transform=transform, target_transform=target_transform
#         )
#
#     def __len__(self):
#         return len(self.ori_dataset)
#
#     def __getitem__(self, index):
#         (input_data, target) = self.ori_dataset[index]
#         from scipy.ndimage import zoom
#         new_data = []
#         for t in range(input_data.shape[0]):
#             new_data.append(zoom(input_data[t, ...], (1, 0.5, 0.5), order=0))
#         new_data = np.stack(new_data, axis=0)
#
#         return new_data, target


# class Spike_transformDataset(Dataset):
#     def __init__(self, original_dataset, transform=None):
#         super(Spike_transformDataset, self).__init__()
#         self.dataset = original_dataset
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, index):
#         (input_data, target) = self.dataset[index]
#         new_data = []
#         if self.transform:
#             for t in range(input_data.shape[0]):
#                 new_data.append(self.transform(input_data[t, ...]))
#         new_data = torch.stack(new_data, dim=0)
#         return new_data, target


# this is used for Artificial CNN
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """Denormalize image, either single image (C,H,W) or image batch (N,C,H,W)"""
        batch = (len(tensor.size()) == 4)
        for t, m, s in zip(tensor.permute(1,0,2,3) if batch else tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def permutate_image_pixels(image, permutation):
    if permutation is None:
        return image
    else:
        c, h, w = image.size()
        image = image.view(c, -1)
        image = image[:, permutation]
        image = image.view(c, h, w)
        return image
