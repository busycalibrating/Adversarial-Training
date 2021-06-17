from adv_train.launcher import Launcher
from adv_train.model import build_model_mnist
from adv_train.dynamic import Langevin
from adv_train.dataset import AdversarialDataset 
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import tqdm

# Script to train a robust classifier.
# Should support different ways of doing adversarial training.

class AdversarialTraining(Launcher):
    def __init__(self, lr=0.01, step=0.1, n_lan=100, batch_size=100):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        dataset = MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
        self.dataset = AdversarialDataset(dataset)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.model = build_model_mnist(device=self.device)
        self.opt = optim.SGD(self.model.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()

        self.langevin = Langevin(self.forward, n_lan=n_lan, lr=step)

    def forward(self, x, y, return_pred=False):
        pred = self.model(x)
        loss = self.loss(pred, y)
        if return_pred:
            return loss, pred
        return loss

    def epoch_adversarial_lan(self):
        """Adversarial training/evaluation epoch over the dataset"""
        total_loss, total_err = 0.,0.
        for x, y, x_adv, idx in tqdm.tqdm(self.dataloader):
            x, x_adv, y = x.to(self.device), x_adv.to(self.device), y.to(self.device).long()
            x_adv.requires_grad_()

            x_adv = self.langevin.step(x_adv, y, x)
            
            self.opt.zero_grad()
            loss, pred = self.forward(x_adv, y, return_pred=True)
            loss.backward()
            self.opt.step()

            total_err += (pred.max(dim=1)[1] != y).sum().item()
            total_loss += loss.item()
            
            self.dataset.update_adv(x_adv, idx)
        return total_err / len(self.dataset), total_loss / len(self.dataset)

    def launch(self, n_epochs=10, save_model=False):
        for _ in range(n_epochs):
            train_err, train_loss = self.epoch_adversarial_lan()
            print("Train error: %.2f,  Train Loss: %.4f"%(train_err, train_loss))  # TODO: Replace this with a Logger interface
        
        if save_model:
            torch.save(self.model.state_dict(), "model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--step', default=0.1, type=int)
    parser.add_argument('--n_lan', default=100, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--save_model', action='store_true')

    args = parser.parse_args()

    adv_train = AdversarialTraining(args.lr, args.step, args.n_lan, args.batch_size)
    adv_train.launch(args.n_epochs, args.save_model)