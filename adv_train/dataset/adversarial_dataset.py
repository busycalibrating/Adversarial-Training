from torch.utils.data import Dataset, DataLoader
import torch
import random
import numpy as np
from adv_train.utils import plot


# TODO: There might be a lot of rework to do with this class but should work fine for now.
# - Might not handle large dataset since it creates several copies in memory !
# - Doesn't support easy evaluation of all elements of the adversarial dataset when n_adv > 1


class AdversarialDataset(Dataset):
    def __init__(self, dataset, n_adv=1):
        super().__init__()

        self.dataset = dataset

        self.n_adv = n_adv

        # Maybe need to change this for larger dataset, might be slow.
        self.adv = []
        for x in DataLoader(dataset, batch_size=len(dataset)):
            self.adv.append(x[0])
        self.adv = torch.cat(self.adv, 0)
        self.adv = self.adv.unsqueeze(1).repeat_interleave(n_adv, dim=1)

    def update_adv(self, x, index):
        self.adv[index[0], index[1]] = x.detach().cpu()

    def __getitem__(self, index_1, index_2=None):
        if index_2 is None:
            index_2 = random.randrange(self.n_adv)
        return self.dataset[index_1] + (self.adv[index_1, index_2], (index_1, index_2))

    def __len__(self):
        return len(self.dataset)

    def checkpoint(self, filename):
        torch.save(self.adv, filename)

    def load(self, filename):
        self.adv = torch.load(filename)

    def plot(self, idx=None, n_adv=100, nrow=None, scale_noise=1, save=None, fig=None):
        if idx is None:
            idx = np.random.choice(self.__len__(), size=n_adv, replace=False)
        idx2 = np.random.choice(self.n_adv, size=n_adv)

        noise = self.compute_noise(idx, idx2)
        img = self.adv[idx, idx2] + scale_noise * noise

        fig = plot(img, n_adv, nrow, scale_noise, save, fig)
        return fig

    def compute_noise(self, idx, idx2):
        # A bit hacky but not sure how to use this otherwise
        dataset = torch.utils.data.Subset(self.dataset, idx)
        dataset = next(iter(DataLoader(dataset, batch_size=len(dataset))))[0]

        noise = dataset - self.adv[idx, idx2]
        return noise

    def plot_noise(
        self, idx=None, n_adv=100, nrow=None, scale_noise=1, save=None, fig=None
    ):
        if idx is None:
            idx = np.random.choice(self.__len__(), size=n_adv, replace=False)
        idx2 = np.random.choice(self.n_adv, size=n_adv)

        noise = self.compute_noise(idx, idx2)
        img = scale_noise * noise

        fig = plot(img, n_adv, nrow, scale_noise, save, fig)
        return fig
