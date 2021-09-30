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
            args.type = random.choice([MnistModel.MODEL_A, MnistModel.MODEL_B, MnistModel.MODEL_C, MnistModel.MODEL_D, MnistModel.MADRY_V2])

        elif args.dataset == DatasetType.CIFAR:
            args.type = random.choice(list(CifarModel))

        args.lr = 10**np.random.uniform(-1, -3)
        args.eps_iter = 10**np.random.uniform(-1, -3)
        args.attacker_type = Attacker.PGD_40
        args.restart = True 
        args.sign_flag = True

        adv_train = AdversarialTraining(args)
        adv_train.run()

        args.restart = False
        args.attacker_type = Attacker.LANGEVIN
        args.nb_iter = 1
        args.noise_scale = 10**np.random.uniform(-2, 0)

        adv_train.update_args(args)
        adv_train.run()