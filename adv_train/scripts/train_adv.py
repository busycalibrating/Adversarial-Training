import argparse
import copy
import logging
import os
import pprint
import subprocess
import time
import tqdm
import uuid
import wandb


# Pytorch stuff
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.cuda as cuda

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

from adv_train.utils.logger import Database, RecordState
from adv_train.utils.wandb import WandB

# Script to train a robust classifier.
# Should support different ways of doing adversarial training.

# TODO: Would be nice to change the args parsing,
# so that the argument is automatically added to class !

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdversarialTraining(Launcher):
    @classmethod
    def add_argument(cls, parser=None):
        parser = super().add_arguments(parser)
        parser = WandB.add_arguments(parser)

        parser.add_argument(
            "--dataset",
            default=DatasetType.MNIST,
            type=DatasetType,
            choices=DatasetType,
        )
        parser.add_argument("--seed", default=42, type=int, help="Random seed")
        parser.add_argument("--lr", default=0.1, type=float, help="Learning rate")
        parser.add_argument("--n_epochs", default=10, type=int, help="Number of training epochs")
        parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
        parser.add_argument(
            "--save_model",
            default=None,
            type=str,
            help="If specified, path to where a model is saved every epoch",
        )
        parser.add_argument(
            "--model_dir", default=None, type=str, help="Where to load models from (?)"
        )
        parser.add_argument(
            "--eval_name", default=None, type=str, help="Evaluates against another trained model(?)"
        )
        parser.add_argument("--eval_clean_flag", action="store_true", help="")
        parser.add_argument("--eval_adv", default=None, type=Attacker, choices=Attacker, help="")
        parser.add_argument(
            "--dest",
            default=None,
            type=str,
            help="An additional save destination for the final model file",
        )
        parser.add_argument("--train_on_clean", action="store_true", help="")
        parser.add_argument("--n_adv", default=1, type=int, help="")
        parser.add_argument("--restart", action="store_true", help="")
        parser.add_argument("--log_dir", default="./logs", type=str, help="")
        parser.add_argument(
            "--fancy_db_name",
            action="store_true",
            help="If enabled, instead of creating a db entry in log_dir/<uuid>, creates an entry in logdir/<wandb_group>/<date>_",
        )

        args, _ = parser.parse_known_args()

        if args.dataset == DatasetType.MNIST:
            parser.add_argument(
                "--type",
                default=MnistModel.MODEL_A,
                type=MnistModel,
                choices=MnistModel,
                help="MNIST model types",
            )
        elif args.dataset == DatasetType.CIFAR:
            parser.add_argument(
                "--type",
                default=CifarModel.RESNET_18,
                type=CifarModel,
                choices=CifarModel,
                help="CIFAR model types",
            )
        # TODO: add ImageNet stuff here

        return parser

    def __init__(self, args):
        super().__init__(args)
        self.wandb_run = None
        self.db_record_name = None

    def init_wandb(self):
        if self.wandb_project is not None:
            logger.info(f"Logging to Weights and Biases")
            logger.info(f"Project:\t{self.wandb_project}")
            logger.info(f"Entity:\t{self.wandb_entity}")
            logger.info(f"Name:\t{self.wandb_name}")
            logger.info(f"Group:\t{self.wandb_group}")
            # if kwargs is not None:
            # logger.info(f"Additional kwargs:\t{kwargs}")

            self.wandb_run = wandb.init(
                name=self.wandb_name,
                entity=self.wandb_entity,
                project=self.wandb_project,
                group=self.wandb_group,
            )
            wandb.config.update(self)
        else:
            logger.info(f"Weights and Biases not configured for this run")

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

            self.num_grad += self.attacker.nb_iter * len(x)

            if not self.attacker.projection.is_valid(x, x_adv):
                raise ValueError()

            self.opt.zero_grad()
            loss, pred = self.forward(x_adv, y, return_pred=True)

            # trains on both clean + adversarial examples
            if self.train_on_clean:
                loss += self.forward(x, y, return_pred=False)

            loss.backward()
            self.opt.step()

            self.num_grad += len(x)

            total_err += (pred.max(dim=1)[1] != y).sum().item()
            total_loss += loss.item()

            if self.restart:
                continue

            # TODO: will need to update this for imagenet
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

    def launch(self, seed: int = None):
        """
        Launches a job; seed is an optional argument to support slurm array submissions. If not
        specified, will default to self.seed
        """
        # TODO: fix logging interface for submitit?
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Overwrite some basic configs
        if "SLURMD_NODENAME" in os.environ:
            # get node name for debugging in the future if necessary
            self.slurmd_nodename = os.environ.get("SLURMD_NODENAME")

        if seed is not None:
            self.seed = seed

        if args.fancy_db_name:
            # TODO: hacky, fix this later
            if args.wandb_group is None:
                raise RuntimeError(
                    "Specified --fancy_db_entry without a --wandb_group (required). "
                    "This won't log to WandB if you don't set --wandb_project"
                )
            self.log_dir = os.path.join(self.log_dir, args.wandb_group)
            self.db_record_name = f'{time.strftime("%Y%m%d-%H%M%S")}__{str(uuid.uuid4())[:8]}'

        logger.info(f"Logging to '{os.path.join(self.log_dir, self.db_record_name)}'")
        logger.info(pprint.pformat(vars(self)))

        # init tracking (wandb, local db)
        self.init_wandb()

        db = Database(self.log_dir)
        self.record = db.create_record(self.db_record_name)
        self.record.save_hparams(self.args)

        try:
            self.num_grad = 0
            torch.manual_seed(self.seed)
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

            self.model = load_classifier(self.dataset, self.type, device=self.device, eval=False)

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

                # train epoch; time the training using cuda Events
                start, end = cuda.Event(enable_timing=True), cuda.Event(enable_timing=True)
                start.record()
                train_err, train_loss = self.epoch_adversarial_lan()
                end.record()
                torch.cuda.synchronize()  # waits for everything to finish running
                train_time = (
                    start.elapsed_time(end) / 1000
                )  # Event.elapsed_time(...) returns milliseconds

                attacker_err = -1.0
                if self.model_eval is not None:
                    attacker_err, _ = self.eval_attacker()

                clean_err = -1.0
                if self.eval_clean_flag:
                    clean_err, _ = self.eval()

                adv_err = -1.0
                if self.eval_adv is not None:
                    adv_err, _ = self.eval(attacker=self.eval_adv)

                logger.info(
                    "Iter: %i, Train time: %.2fs, Train error: %.2f%%,  Train Loss: %.4f, Attacker error: %.2f%%, Clean error: %.2f%%, Adversarial error: %.2f%%"
                    % (epoch, train_time, train_err, train_loss, attacker_err, clean_err, adv_err)
                )  # TODO: Replace this with a Logger interface

                log = {
                    "epoch": epoch,
                    "train_time": train_time,
                    "train_err": train_err,
                    "train_loss": train_loss,
                    "attacker_err": attacker_err,
                    "clean_err": clean_err,
                    "adv_err": adv_err,
                    "num_grad": self.num_grad,
                }
                self.record.add(log)
                if self.wandb_run is not None:
                    self.wandb_run.log(log)

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
            if self.wandb_run is not None:
                model_filepath = (
                    self.save_model if self.save_model is not None else self.record.model_filepath
                )
                artifact = wandb.Artifact("model", type="model")
                artifact.add_file(model_filepath)
                self.wandb_run.log_artifact(artifact)
                self.wandb_run.finish()

        except:
            self.record.fail()
            raise


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = Attacker.add_arguments(parser)
    parser = AdversarialTraining.add_argument(parser)
    args = AdversarialTraining.parse_args_with_config(parser)

    logger.info(pprint.pformat(vars(args)))

    array_args = {}
    if isinstance(args.seed, list):
        array_args['seed'] = args.seed

    adv_train = AdversarialTraining(args)
    adv_train.run(array_args)
