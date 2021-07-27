from adv_train.launcher import Launcher
from adv_train.model.mnist import load_mnist_classifier, MnistModel, load_mnist_dataset
from adv_train.dynamic import Attacker
from adv_train.dataset import AdversarialDataset
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
import os
import copy
import subprocess

# Script to train a robust classifier.
# Should support different ways of doing adversarial training.

#TODO: Would be nice to change the args parsing, so that the argument is automatically added to class !


class AdversarialTraining(Launcher):
    @staticmethod
    def add_argument(parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()

        parser.add_argument('--attacker', default=Attacker.LANGEVIN, type=Attacker, choices=Attacker)
        parser.add_argument('--lr', default=0.1, type=float)
        parser.add_argument('--n_epochs', default=10, type=int)
        parser.add_argument('--batch_size', default=100, type=int)
        parser.add_argument('--save_model', default=None, type=str)
        parser.add_argument('--model_dir', default="/checkpoint/hberard/OnlineAttack/pretained_models", type=str)
        parser.add_argument('--type', default=MnistModel.MODEL_A, type=MnistModel, choices=MnistModel)
        parser.add_argument('--eval_name', default=None, type=str)
        parser.add_argument('--eval_clean_flag', action="store_true")
        parser.add_argument('--eval_adv', default=None, type=Attacker, choices=Attacker)
        parser.add_argument('--dest', default=None, type=str)
        parser.add_argument('--train_on_clean', action="store_true")
        parser.add_argument('--n_adv', default=1, type=int)
        parser.add_argument('--restart', action="store_true")

        return parser
    
    def __init__(self, args):
        torch.manual_seed(1234)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        dataset = load_mnist_dataset()
        self.dataset = AdversarialDataset(dataset, n_adv=args.n_adv)
        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        test_dataset = load_mnist_dataset(train=False)
        self.test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        self.model = load_mnist_classifier(args.type, device=self.device, eval=False)

        self.model_eval = None
        if args.eval_name is not None:
            self.model_eval = load_mnist_classifier(args.type, name=args.eval_name, model_dir=args.model_dir, device=self.device, eval=True)

        self.opt = optim.SGD(self.model.parameters(), lr=args.lr)
        self.loss_fn = nn.CrossEntropyLoss()

        self.attacker = Attacker.load_attacker(self.model, args)

        self.save_model = args.save_model
        self.n_epochs = args.n_epochs
        self.eval_clean_flag = args.eval_clean_flag

        self.eval_adv = args.eval_adv
        if self.eval_adv is not None:
            # TODO: This is kinda hacky, would be nice to have a better interface for this !
            attacker_args = copy.deepcopy(args)
            attacker_args.attacker_type = self.eval_adv
            attacker_args.nb_iter = 10
            self.eval_adv = Attacker.load_attacker(self.model, attacker_args)

        self.dest = args.dest
        self.train_on_clean = args.train_on_clean
        self.restart = args.restart

    def forward(self, x, y, model=None, return_pred=False):
        if model is None:
            pred = self.model(x)
        else:
            pred = model(x)

        loss = self.loss_fn(pred, y)
        if return_pred:
            return loss, pred
        return loss

    def epoch_adversarial_lan(self):
        """Adversarial training/evaluation epoch over the dataset"""
        total_loss, total_err = 0.,0.
        for x, y, x_adv, idx in tqdm.tqdm(self.dataloader):
            x, x_adv, y = x.to(self.device), x_adv.to(self.device), y.to(self.device).long()

            x_adv = self.attacker.perturb(x_adv, y)
            x_adv = self.attacker.projection(x_adv, x)

            dist = abs(x_adv - x).view(len(x_adv), -1).max(1)[0]
            if (dist > 0.3 + 1e-6).any():
                print(dist)
                raise ValueError()
            
            self.opt.zero_grad()
            loss, pred = self.forward(x_adv, y, return_pred=True)

            if self.train_on_clean:
                loss += self.forward(x, y, return_pred=False)

            loss.backward()
            self.opt.step()

            total_err += (pred.max(dim=1)[1] != y).sum().item()
            total_loss += loss.item()
            
            if self.restart:
                continue
            self.dataset.update_adv(x_adv, idx)
        return total_err / len(self.dataset)*100, total_loss / len(self.dataset)

    def _eval(self, x, y, model=None, attacker=None):
        x, y = x.to(self.device), y.to(self.device).long()

        if attacker is not None:
            x = attacker.perturb(x, y)
            
        loss, pred = self.forward(x, y, model, return_pred=True)
        acc = (pred.max(dim=1)[1] != y).sum()

        return loss.item(), acc.item()

    def eval(self, attacker=None):
        total_loss, total_err = 0., 0.
        n_samples = 0
        for x, y in self.test_dataloader:
            loss, acc = self._eval(x, y, attacker=attacker)

            total_err += acc
            total_loss += loss
            n_samples += len(x)
            
        return total_err / n_samples*100, total_loss / n_samples

    def eval_attacker(self):
        total_loss, total_err = 0., 0.
        for _, y, x, _ in self.dataloader:               
            loss, acc = self._eval(x, y, self.model_eval)

            total_err += acc
            total_loss += loss
            
        return total_err / len(self.dataset)*100, total_loss / len(self.dataset)

    def launch(self):
        p = None
        for epoch in range(self.n_epochs):
            train_err, train_loss = self.epoch_adversarial_lan()
            attacker_err = 0.
            if self.model_eval is not None:
                attacker_err, _ = self.eval_attacker()

            clean_err = 0.
            if self.eval_clean_flag:
                clean_err, _ = self.eval()
                
            adv_err = 0.
            if self.eval_adv is not None:
                adv_err, _ = self.eval(attacker=self.eval_adv)
            
            print("Iter: %i, Train error: %.2f%%,  Train Loss: %.4f, Attacker error: %.2f%%, Clean error: %.2f%%, Adversarial error: %.2f%%"%(
                epoch, train_err, train_loss, attacker_err, clean_err, adv_err))  # TODO: Replace this with a Logger interface
            
            if self.save_model is not None:
                os.makedirs(os.path.dirname(self.save_model), exist_ok=True)
                torch.save(self.model.state_dict(), self.save_model)
                if self.dest is not None:
                    if p is None:
                        p = subprocess.Popen(["scp", "-r", self.save_model, self.dest], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    elif p.poll() is not None:
                        p = None
                    

if __name__ == "__main__":
    parser = Attacker.add_arguments()
    parser = AdversarialTraining.add_argument(parser)

    args = parser.parse_args()

    adv_train = AdversarialTraining(args)
    adv_train.launch()
