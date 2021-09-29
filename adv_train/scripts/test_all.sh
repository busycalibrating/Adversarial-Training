#!/bin/bash

NUM_EPOCHS="100"

echo "Testing against FGSM:"
python -m adv_train.scripts.test_langevin --n_epochs $NUM_EPOCHS --attacker_type fgsm --model_path $1

echo "Testing against PGD-40:"
python -m adv_train.scripts.test_langevin --n_epochs $NUM_EPOCHS --eps_iter 0.01 --attacker_type pgd-40 --model_path $1

echo "Testing against PGD-100:"
python -m adv_train.scripts.test_langevin --n_epochs $NUM_EPOCHS --eps_iter 0.01 --attacker_type pgd-100 --model_path $1


