from .train_adv import AdversarialTraining, Attacker, DatasetType, MnistModel, CifarModel
import random
import numpy as np


if __name__ == "__main__":
    parser = Attacker.add_arguments()
    parser = AdversarialTraining.add_argument(parser)
    parser.add_argument("--n_runs", default=1, type=int)

    args = parser.parse_args()

    for i in range(args.n_runs):
        if args.dataset == DatasetType.MNIST:
            args.type = MnistModel.MODEL_A

        elif args.dataset == DatasetType.CIFAR:
            args.type = random.choice(list(CifarModel))

        args.lr = 10**np.random.uniform(-1, -2.5)
        args.eps_iter = 10**np.random.uniform(-1, -2.3)
        args.attacker_type = Attacker.PGD_40
        args.restart = True 
        args.sign_flag = True

        adv_train = AdversarialTraining(args)
        adv_train.run()

        args.restart = False
        args.attacker_type = Attacker.LANGEVIN
        args.nb_iter = 1
        args.noise_scale = 10**np.random.uniform(-3, -0.2)

        adv_train.update_args(args)
        adv_train.run()