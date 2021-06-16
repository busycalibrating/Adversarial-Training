from torch.utils.data import Dataset, DataLoader
import torch


# TODO: There might be a lot of rework to do with this class but should work fine for now.

class AdversarialDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        
        self.dataset = dataset

        # Maybe need to change this for larger dataset, might be slow.
        self.adv = []
        for x in DataLoader(dataset, batch_size=len(dataset)):
            self.adv.append(x[0])   
        self.adv = torch.cat(self.adv, 0) 

    def update_adv(self, x, index):
        self.adv[index] = x.detach().cpu()

    def __getitem__(self, index):
        return self.dataset[index] + (self.adv[index], index)

    def __len__(self):
        return len(self.dataset)

    def checkpoint(self, filename):
        torch.save(self.adv, filename)
