# Adversarial Training

## Installation:
To install package just run:
`pip install -e .`

To be able to use Madry models, you need to run:
`./install_madry_challenge.sh -d /home/mila/b/berardhu/share/AdversarialTraining/pretrained_models`
where -d specifies the folder where to save the models.

-------------
## 1. To test the Langevin dynamic on a classifier:
`python -m adv_train.scripts.test_langevin --nb_iter 10 --eps_iter 0.01 --noise_scale 0.4 --sign_flag --model_dir /home/mila/b/berardhu/share/AdversarialTraining/pretrained_models`

To change the model against which you want to evaluate the attacker simply use the flag: `--name train_0`.

-------------------------
## 2. To train a robust classifier using Langevin:
`python -m adv_train.scripts.train_adv --nb_iter 1 --n_epochs 100 --eps_iter 0.01 --noise_scale 0.2 --sign_flag --eval_adv pgd --model_dir /home/mila/b/berardhu/share/AdversarialTraining/pretrained_models`

This will train a classifier using langevin. It will also report the performance of the classifier against a PGD attacker.

To evaluate the attacker against another model simply use the flag: `--eval_name train_0`.

--------------------
## 3. To train a robust classifier using PGD:
You just need to add the  options `--restart` and `--attacker pgd`:

`python -m adv_train.scripts.train_adv --nb_iter 40 --n_epochs 100 --eps_iter 0.01 --eval_adv pgd --model_dir /home/mila/b/berardhu/share/AdversarialTraining/pretrained_models --restart --attacker pgd`

--------------------
## 4. To test the robust classifier against PGD:
`python -m adv_train.scripts.test_langevin --nb_iter 100 --eps_iter 0.01 --n_epochs 1 --attacker_type pgd --model_path /home/mila/b/berardhu/share/AdversarialTraining/saved_model/model_new.pt`

--------------------------
# 5. To test a classifier against FGSM, PGD-40 and PGD-100
`./adv_train/scripts/test_all.sh /home/mila/b/berardhu/share/AdversarialTraining/saved_model/model_new.pt`

---------------
## Baselines

Example of command: `python -m adv_train.scripts.test_langevin --nb_iter 100 --n_epochs 1 --eps_iter 0.01 --name PGD_ATTACK_train_0 --attacker_type pgd`
