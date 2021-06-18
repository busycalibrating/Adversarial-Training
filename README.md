# Adversarial Training

## To test the Langevin dynamic on a classifier:
`python -m adv_train.scripts.test_langevin --n_lan 10 --noise_scale 0.4 --sign_flag --model_dir /home/mila/b/berardhu/AdversarialTraining/pretrained_models`
By default this should achieve around 98% error.
To be able to run this command you need to have access to `/home/mila/b/berardhu/AdversarialTraining/pretrained_models` on the mila cluster. Let me know if you don't have access to it !