from adv_train.launcher import Launcher
from adv_train.model import (
    DatasetType,
    MnistModel,
    CifarModel,
    load_dataset,
    load_classifier,
)
from adv_train.dynamic import Attacker
from adv_train.dataset import AdversarialDataset
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
import os
import copy
import subprocess
from adv_train.utils.logger import Database, RecordState

# Script to train a robust classifier.
# Should support different ways of doing adversarial training.

# TODO: Would be nice to change the args parsing,
# so that the argument is automatically added to class !


class AdversarialTraining(Launcher):
    @classmethod
    def add_argument(cls, parser=None):
        parser = super().add_arguments(parser)

        parser.add_argument(
            "--dataset",
            default=DatasetType.MNIST,
            type=DatasetType,
            choices=DatasetType,
        )

        args, _ = parser.parse_known_args()

        if args.dataset == DatasetType.MNIST:
            parser.add_argument(
                "--type",
                default=MnistModel.MODEL_A,
                type=MnistModel,
                choices=MnistModel,
            )

        elif args.dataset == DatasetType.CIFAR:
            parser.add_argument(
                "--type",
                default=CifarModel.RESNET_18,
                type=CifarModel,
                choices=CifarModel,
            )

        parser.add_argument("--lr", default=0.1, type=float)
        parser.add_argument("--n_epochs", default=10, type=int)
        parser.add_argument("--batch_size", default=100, type=int)
        parser.add_argument("--save_model", default=None, type=str)
        parser.add_argument(
            "--model_dir",
            default="/checkpoint/hberard/OnlineAttack/pretained_models",
            type=str,
        )
        parser.add_argument("--eval_name", default=None, type=str)
        parser.add_argument("--eval_clean_flag", action="store_true")
        parser.add_argument("--eval_adv", default=None, type=Attacker, choices=Attacker)
        parser.add_argument("--dest", default=None, type=str)
        parser.add_argument("--train_on_clean", action="store_true")
        parser.add_argument("--n_adv", default=1, type=int)
        parser.add_argument("--restart", action="store_true")
        parser.add_argument("--log_dir", default="./logs", type=str)

        return parser

    def __init__(self, args):
        super().__init__(args)

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
        total_loss, total_err = 0.0, 0.0
        for x, y, x_adv, idx in tqdm.tqdm(self.dataloader):
            x, x_adv, y = (
                x.to(self.device),
                x_adv.to(self.device),
                y.to(self.device).long(),
            )

            x_adv = self.attacker.perturb(x_adv, y)
            x_adv = self.attacker.projection(x_adv, x)

            self.num_grad += self.attacker.nb_iter*len(x)

            if not self.attacker.projection.is_valid(x, x_adv):
                raise ValueError()

            self.opt.zero_grad()
            loss, pred = self.forward(x_adv, y, return_pred=True)

            if self.train_on_clean:
                loss += self.forward(x, y, return_pred=False)

            loss.backward()
            self.opt.step()

            self.num_grad += len(x)

            total_err += (pred.max(dim=1)[1] != y).sum().item()
            total_loss += loss.item()

            if self.restart:
                continue
            self._dataset.update_adv(x_adv, idx)
        return total_err / len(self._dataset) * 100, total_loss / len(self._dataset)

    def _eval(self, x, y, model=None, attacker=None):
        x, y = x.to(self.device), y.to(self.device).long()

        if attacker is not None:
            x = attacker.perturb(x, y)

        loss, pred = self.forward(x, y, model, return_pred=True)
        acc = (pred.max(dim=1)[1] != y).sum()

        return loss.item(), acc.item()

    def eval(self, attacker=None):
        self.model.eval()
        total_loss, total_err = 0.0, 0.0
        n_samples = 0
        for x, y in self.test_dataloader:
            loss, acc = self._eval(x, y, attacker=attacker)

            total_err += acc
            total_loss += loss
            n_samples += len(x)
        self.model.train()

        return total_err / n_samples * 100, total_loss / n_samples

    def eval_attacker(self):
        total_loss, total_err = 0.0, 0.0
        for _, y, x, _ in self.dataloader:
            loss, acc = self._eval(x, y, self.model_eval)

            total_err += acc
            total_loss += loss

        return total_err / len(self._dataset) * 100, total_loss / len(self._dataset)

    def launch(self):
        db = Database(self.log_dir)
        self.record = db.create_record()
        self.record.save_hparams(self.args)

        try:
            self.num_grad = 0
            torch.manual_seed(1234)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            dataset = load_dataset(self.dataset, train=True)
            self._dataset = AdversarialDataset(dataset, n_adv=self.n_adv)
            self.dataloader = DataLoader(
                self._dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
            )

            test_dataset = load_dataset(self.dataset, train=False)
            self.test_dataloader = DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
            )

            self.model = load_classifier(
                self.dataset, self.type, device=self.device, eval=False
            )

            self.model_eval = None
            if self.eval_name is not None:
                self.model_eval = load_classifier(
                    self.dataset,
                    self.type,
                    name=self.eval_name,
                    model_dir=self.model_dir,
                    device=self.device,
                    eval=True,
                )

            self.opt = optim.SGD(self.model.parameters(), lr=self.lr)
            self.loss_fn = nn.CrossEntropyLoss()

            self.attacker = Attacker.load_attacker(self.model, self.args)

            if self.eval_adv is not None:
                # TODO: This is kinda hacky,
                # would be nice to have a better interface for this !
                attacker_args = copy.deepcopy(self.args)
                attacker_args.attacker_type = self.eval_adv
                attacker_args.eps_iter = 0.01
                self.eval_adv = Attacker.load_attacker(self.model, attacker_args)

            p = None
            for epoch in range(self.n_epochs):
                train_err, train_loss = self.epoch_adversarial_lan()
                attacker_err = 0.0
                if self.model_eval is not None:
                    attacker_err, _ = self.eval_attacker()

                clean_err = 0.0
                if self.eval_clean_flag:
                    clean_err, _ = self.eval()

                adv_err = 0.0
                if self.eval_adv is not None:
                    adv_err, _ = self.eval(attacker=self.eval_adv)

                print(
                    "Iter: %i, Train error: %.2f%%,  Train Loss: %.4f, Attacker error: %.2f%%, Clean error: %.2f%%, Adversarial error: %.2f%%"
                    % (epoch, train_err, train_loss, attacker_err, clean_err, adv_err)
                )  # TODO: Replace this with a Logger interface

                self.record.add({"epoch": epoch, "train_err": train_err, "train_loss": train_loss, "attacker_err": attacker_err,
                                 "clean_err": clean_err, "adv_err": adv_err, "num_grad": self.num_grad})

                if self.save_model is not None:
                    os.makedirs(os.path.dirname(self.save_model), exist_ok=True)
                    torch.save(self.model.state_dict(), self.save_model)
                    if self.dest is not None:
                        if p is None:
                            p = subprocess.Popen(
                                ["scp", "-r", self.save_model, self.dest],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                            )
                        elif p.poll() is not None:
                            p = None
                else:
                    self.record.save_model(self.model)
            self.record.close()
        
        except:
            self.record.fail()
            raise


if __name__ == "__main__":
    parser = Attacker.add_arguments()
    parser = AdversarialTraining.add_argument(parser)

    args = parser.parse_args()

    adv_train = AdversarialTraining(args)
    adv_train.run()
