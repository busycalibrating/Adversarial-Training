import torch.nn as nn
import torchvision
import math
import matplotlib.pyplot as plt


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(len(x), -1)


def plot(tensor, n_adv=100, nrow=None, scale_noise=1, save=None, fig=None):
    if nrow is None:
        nrow = int(math.sqrt(n_adv))

    img = torchvision.utils.make_grid(tensor, nrow=nrow)
    if save is not None:
        torchvision.utils.save_image(img, save)

    if fig is None:
        fig = plt.figure()

    img = img.permute(1, 2, 0)
    plt.imshow(img)
    return fig
