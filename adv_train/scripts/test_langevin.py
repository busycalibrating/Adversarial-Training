# Should test if the langevin actually produces adversarial against a pretrained classifier
from adv_train.launcher import Launcher
from adv_train.model.mnist import load_mnist_classifier, MnistModel, load_mnist_dataset
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


class LangevinAttack(Launcher):
    def __init__(self, name, model_dir, step=0.1, n_lan=10, batch_size=100):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        dataset = load_mnist_dataset()
        self.dataset = AdversarialDataset(dataset)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.model = load_mnist_classifier(MnistModel.MODEL_A, name=name, model_dir=model_dir, device=self.device, eval=False)
        self.loss = nn.CrossEntropyLoss()

        self.n_lan = n_lan
        self.langevin = Langevin(self.forward, n_lan=n_lan, lr=step)

    def forward(self, x, y, return_pred=False):
        pred = self.model(x)
        loss = self.loss(pred, y)
        if return_pred:
            return loss, pred
        return loss

    def epoch_langevin(self):
        """Adversarial training/evaluation epoch over the dataset"""
        total_loss, total_err = 0.,0.
        for x, y, x_adv, idx in tqdm.tqdm(self.dataloader):
            x, x_adv, y = x.to(self.device), x_adv.to(self.device), y.to(self.device).long()
            x_adv.requires_grad_()

            x_adv = self.langevin.step(x_adv, y, x)
            dist = abs(x_adv - x).view(len(x_adv), -1).max(1)[0]
            if (dist > 0.3 + 1e-6).any():
                print(dist)
                raise ValueError()
            
            loss, pred = self.forward(x_adv, y, return_pred=True)

            total_err += (pred.max(dim=1)[1] != y).sum().item()
            total_loss += loss.item()
            
            self.dataset.update_adv(x_adv, idx)
        return total_err / len(self.dataset), total_loss / len(self.dataset)

    def launch(self, n_epochs=10):
        for _ in range(n_epochs):
            train_err, train_loss = self.epoch_langevin()
            print("Train error: %.2f%%,  Train Loss: %.4f"%(train_err*100, train_loss)) #TODO: Replace this with a Logger interface


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--step', default=0.1, type=float)
    parser.add_argument('--n_lan', default=10, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--name', default="train_0", type=str)
    parser.add_argument('--model_dir', default="/checkpoint/hberard/OnlineAttack/pretained_models", type=str)

    args = parser.parse_args()

    torch.manual_seed(1234)
    attack = LangevinAttack(args.name, args.model_dir, args.step, args.n_lan, args.batch_size)
    attack.launch(args.n_epochs)