# Should test if the langevin actually produces adversarial against a pretrained classifier
from adv_train.launcher import Launcher
from adv_train.model import (
    DatasetType,
    MnistModel,
    CifarModel,
    load_dataset,
    load_classifier,
)
import torch
from torch.utils.data import DataLoader
from autoattack import AutoAttack


class AutoAttackEval(Launcher):
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

    def launch(self):
        torch.manual_seed(1234)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        dataset = load_dataset(self.dataset, train=False)

        self.model = load_classifier(
            self.dataset,
            self.type,
            model_path=self.model_path,
            name=self.name,
            model_dir=self.model_dir,
            device=self.device,
            eval=True,
        )

        test_loader = DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=0)
        l = [x for (x, y) in test_loader]
        x_test = torch.cat(l, 0)
        l = [y for (x, y) in test_loader]
        y_test = torch.cat(l, 0)

        if self.dataset == DatasetType.CIFAR:
            epsilon = 8/256
        elif self.dataset == DatasetType.MNIST:
            epsilon = 0.3

        adversary = AutoAttack(self.model, norm="Linf", eps=epsilon, device=self.device)

        adversary.run_standard_evaluation(x_test, y_test, bs=self.batch_size)


if __name__ == "__main__":
    parser = AutoAttackEval.add_arguments()

    args = parser.parse_args()

    torch.manual_seed(1234)
    attack = AutoAttackEval(args)
    attack.run()
