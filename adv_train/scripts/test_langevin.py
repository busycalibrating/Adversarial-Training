# Should test if the langevin actually produces adversarial against a pretrained classifier
from adv_train.launcher import Launcher
from adv_train.model import (
    DatasetType,
    MnistModel,
    CifarModel,
    load_classifier,
    load_dataset,
)
from adv_train.dynamic import Attacker
from adv_train.dataset import AdversarialDataset
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm


class LangevinAttack(Launcher):
    @staticmethod
    def add_arguments(parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()

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

        parser.add_argument("--n_epochs", default=1, type=int)
        parser.add_argument("--batch_size", default=100, type=int)
        parser.add_argument("--model_path", default=None, type=str)
        parser.add_argument("--name", default="train_0", type=str)
        parser.add_argument(
            "--model_dir",
            default="/checkpoint/hberard/OnlineAttack/pretained_models",
            type=str,
        )
        parser.add_argument("--n_adv", default=1, type=int)

        return parser

    def __init__(self, args):
        torch.manual_seed(1234)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        dataset = load_dataset(args.dataset, train=False)
        self.dataset = AdversarialDataset(dataset, n_adv=args.n_adv)
        self.dataloader = DataLoader(
            self.dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
        )

        self.model = load_classifier(
            args.dataset,
            args.type,
            model_path=args.model_path,
            name=args.name,
            model_dir=args.model_dir,
            device=self.device,
            eval=True,
        )
        self.loss_fn = nn.CrossEntropyLoss()

        self.attacker = Attacker.load_attacker(self.model, args)

        self.n_epochs = args.n_epochs

    def forward(self, x, y, return_pred=False):
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        if return_pred:
            return loss, pred
        return loss

    def epoch_langevin(self):
        total_loss, total_err = 0.0, 0.0
        for x, y, x_adv, idx in tqdm.tqdm(self.dataloader):
            x, x_adv, y = (
                x.to(self.device),
                x_adv.to(self.device),
                y.to(self.device).long(),
            )

            x_adv = self.attacker.perturb(x_adv, y)
            x_adv = self.attacker.projection(x_adv, x)

            if not self.attacker.projection.is_valid(x, x_adv):
                raise ValueError()

            loss, pred = self.forward(x_adv, y, return_pred=True)

            total_err += (pred.max(dim=1)[1] != y).sum().item()
            total_loss += loss.item()

            self.dataset.update_adv(x_adv, idx)
        return total_err / len(self.dataset), total_loss / len(self.dataset)

    def launch(self):
        for _ in range(self.n_epochs):
            train_err, train_loss = self.epoch_langevin()
        
        print(
            "Train error: %.2f%%,  Train Loss: %.4f" % (train_err * 100, train_loss)
        )  # TODO: Replace this with a Logger interface


if __name__ == "__main__":
    parser = Attacker.add_arguments()
    parser = LangevinAttack.add_arguments(parser)

    args = parser.parse_args()

    torch.manual_seed(1234)
    attack = LangevinAttack(args)
    attack.launch()
