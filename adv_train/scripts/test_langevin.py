# Should test if the langevin actually produces adversarial against a pretrained classifier
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
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
from adv_train.utils.logger import Database, RecordState


class LangevinAttack(Launcher):
    @classmethod
    def add_arguments(cls, parser=None):
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
        super().__init__(args)
        self._attacker = [None]

    def forward(self, x, y, return_pred=False):
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        if return_pred:
            return loss, pred
        return loss

    def epoch_langevin(self, attacker):
        total_loss, total_err = 0.0, 0.0
        for x, y, x_adv, idx in tqdm.tqdm(self.dataloader):
            x, x_adv, y = (
                x.to(self.device),
                x_adv.to(self.device),
                y.to(self.device).long(),
            )

            x_adv = attacker.perturb(x_adv, y)
            x_adv = attacker.projection(x_adv, x)

            if not attacker.projection.is_valid(x, x_adv):
                raise ValueError()

            loss, pred = self.forward(x_adv, y, return_pred=True)

            total_err += (pred.max(dim=1)[1] != y).sum().item()
            total_loss += loss.item()

            self._dataset.update_adv(x_adv, idx)
        return total_err / len(self._dataset), total_loss / len(self._dataset)

    def set_attacker(self, attacker):
        self._attacker = attacker

    def launch(self, attacker=None):
        if self.record is not None:
            self.record.set_state(RecordState.EVAL_RUNNING)
        try:
            torch.manual_seed(1234)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            dataset = load_dataset(self.dataset, train=False)
            self._dataset = AdversarialDataset(dataset, n_adv=self.n_adv)
            self.dataloader = DataLoader(
                self._dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
            )

            if self.record is not None:
                self.model = self.record.load_model(device=self.device, eval=True)
            else:
                self.model = load_classifier(
                    self.dataset,
                    self.type,
                    model_path=self.model_path,
                    name=self.name,
                    model_dir=self.model_dir,
                    device=self.device,
                    eval=True,
                )

            self.loss_fn = nn.CrossEntropyLoss()

            self.attacker = []
            for attacker in self._attacker:
                self.attacker.append(Attacker.load_attacker(self.model, self.args, attacker_type=attacker))

            results = {}
            for attacker in self.attacker:
                for _ in range(self.n_epochs):
                    train_err, train_loss = self.epoch_langevin(attacker)
                
                print(
                    "(%s) error: %.2f%%,  Loss: %.4f" % (attacker.name, train_err * 100, train_loss)
                )  # TODO: Replace this with a Logger interface
                
                results[attacker.name] = train_err
        
        except:
            if self.record is not None:
                self.record.set_state(RecordState.RUNNING)
            raise

        self.record.save_eval(results)


if __name__ == "__main__":
    parser = Attacker.add_arguments()
    parser = LangevinAttack.add_arguments(parser)

    args = parser.parse_args()

    torch.manual_seed(1234)
    attack = LangevinAttack(args)
    attack.run()
