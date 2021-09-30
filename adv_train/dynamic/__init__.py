from .langevin import Langevin
from advertorch.attacks import LinfPGDAttack, Attack, GradientSignAttack
from .utils import LinfProjection, NoiseType
from enum import Enum
import argparse
import torch.nn as nn
from adv_train.model import DatasetType


# Defines a dummy attacker
class NoAttacker(Attack):
    def __init__(self):
        pass

    def perturb(self, x, y=None):
        return x


class Attacker(Enum):
    FGSM = "fgsm"
    PGD = "pgd"
    PGD_40 = "pgd-40"
    PGD_100 = "pgd-100"
    LANGEVIN = "langevin"
    NONE = "none"

    @classmethod
    def add_arguments(cls, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()

        parser.add_argument(
            "--attacker_type", default=cls.LANGEVIN, type=cls, choices=cls
        )
        parser.add_argument("--nb_iter", default=10, type=int)
        parser.add_argument("--eps_iter", default=1.0, type=float)
        parser.add_argument("--sign_flag", action="store_true")
        parser.add_argument("--noise_scale", default=1, type=float)
        parser.add_argument(
            "--noise_type", default=NoiseType.NORMAL, type=NoiseType, choices=NoiseType
        )

        return parser

    @classmethod
    def load_attacker(cls, classifier, args, attacker_type=None):
        if args.dataset == DatasetType.CIFAR:
            projection = LinfProjection(epsilon=8 / 256, clip_min=-1)
        elif args.dataset == DatasetType.MNIST:
            projection = LinfProjection()

        loss_fn = nn.CrossEntropyLoss()
        if attacker_type is None:
            attacker_type = args.attacker_type
        
        if attacker_type == cls.NONE:
            attacker = NoAttacker()
            attacker.projection = projection
            attacker.nb_iter = 0
        elif attacker_type == cls.FGSM:
            attacker = GradientSignAttack(
                classifier,
                loss_fn=loss_fn,
                eps=projection.epsilon,
                clip_min=projection.clip_min,
                clip_max=projection.clip_max,
            )
            attacker.projection = projection
            attacker.nb_iter = 1
        elif attacker_type == cls.PGD:
            attacker = LinfPGDAttack(
                classifier,
                loss_fn=loss_fn,
                eps=projection.epsilon,
                nb_iter=args.nb_iter,
                eps_iter=args.eps_iter,
                clip_min=projection.clip_min,
                clip_max=projection.clip_max,
                rand_init=False,
            )
            attacker.projection = projection
        elif attacker_type == cls.PGD_40:
            attacker = LinfPGDAttack(
                classifier,
                loss_fn=loss_fn,
                eps=projection.epsilon,
                nb_iter=40,
                eps_iter=args.eps_iter,
                clip_min=projection.clip_min,
                clip_max=projection.clip_max,
                rand_init=False,
            )
            attacker.projection = projection
        elif attacker_type == cls.PGD_100:
            attacker = LinfPGDAttack(
                classifier,
                loss_fn=loss_fn,
                eps=projection.epsilon,
                nb_iter=100,
                eps_iter=args.eps_iter,
                clip_min=projection.clip_min,
                clip_max=projection.clip_max,
                rand_init=False,
            )
            attacker.projection = projection
        elif attacker_type == cls.LANGEVIN:
            attacker = Langevin(
                classifier,
                loss_fn=loss_fn,
                projection=projection,
                nb_iter=args.nb_iter,
                eps_iter=args.eps_iter,
                sign_flag=args.sign_flag,
                noise_scale=args.noise_scale,
            )

        return attacker
