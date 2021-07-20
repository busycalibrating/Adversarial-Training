from adv_train.launcher import Launcher
from adv_train.model.mnist import load_mnist_classifier, MnistModel, load_mnist_dataset
from adv_train.dynamic import Attacker
from adv_train.dataset import AdversarialDataset
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = load_mnist_dataset()
dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=4)

def fgsm(model, X, y, epsilon=0.3):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()
  
def pgd_linf(model, X, y, epsilon=0.3, alpha=0.01, num_iter=100, randomize=False):
    """ Construct FGSM adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

def epoch_adversarial(loader, model, attack, opt=None, **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        delta = attack(model, X, y, **kwargs)
        yp = model(X+delta)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

model_cnn_robust = load_mnist_classifier(MnistModel.MODEL_A, device=device, eval=False)

#Model a load

model_cnn_robust.load_state_dict(torch.load("model_A_100_v3.pth"))

print("FGSM: ", epoch_adversarial(dataloader, model_cnn_robust, fgsm)[0])
print("PGD, 40 iter: ", epoch_adversarial(dataloader, model_cnn_robust, pgd_linf, num_iter=40)[0])
print("PGD, 100 iter: ", epoch_adversarial(dataloader, model_cnn_robust, pgd_linf, num_iter=100)[0])

#Use Madry 

torch.manual_seed(0)
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

model_cnn = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(),
                          nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
                          Flatten(),
                          nn.Linear(7*7*64, 100), nn.ReLU(),
                          nn.Linear(100, 10)).to(device)

#opt = optim.SGD(model_cnn.parameters(), lr=1e-1)
#for t in range(10):
    #train_err, train_loss = epoch_adversarial(dataloader, model_cnn, pgd_linf, opt)
    #test_err, test_loss = epoch(test_loader, model_cnn_robust)
    #adv_err, adv_loss = epoch_adversarial(test_loader, model_cnn_robust, pgd_linf)
    #if t == 4:
   #     for param_group in opt.param_groups:
  #          param_group["lr"] = 1e-2
 #   print(train_err)

#torch.save(model_cnn.state_dict(), "model_cnn_madry.pt")

#print("FGSM: ", epoch_adversarial(dataloader, model_cnn, fgsm)[0])
#print("PGD, 40 iter: ", epoch_adversarial(dataloader, model_cnn, pgd_linf, num_iter=40)[0])
