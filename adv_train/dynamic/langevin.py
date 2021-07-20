import torch
import torch.autograd as autograd
import torch.nn as nn
import math
from advertorch.attacks import Attack, LabelMixin
from .utils import LinfProjection, NoiseType
# This should abstract the Langevin dynamics, make it easy to change the dynamics or propose new dynamics MCMC type.


class Langevin(Attack, LabelMixin):
    def __init__(self, predict, loss_fn=None, projection=LinfProjection(), nb_iter=1, eps_iter=1, sign_flag=False, noise_scale=1., targeted=False,
                 noise_type=NoiseType.UNFIFORM):
        # `forward` should be a function (or class ?), that outputs a scalar. Not sure what the best way to implement this ?
        # `projection` is a class that project back onto the constraint set.
        super().__init__(predict, loss_fn, projection.clip_min, projection.clip_max)
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")
        
        self.projection = projection
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter

        self.sign_flag = sign_flag
        self.noise_scale = noise_scale
        self.targeted = targeted
        self.noise_type = noise_type

    def forward(self, x, y):
        pred = self.predict(x)
        loss = self.loss_fn(pred, y)
        return loss

    def _perturb(self, x, y, x_ref):
        # Function that computes a single step of Langevin (hidden)
        # x_ref if specified is used to project back on the constraint set.
        loss = self.forward(x, y)
        grad = autograd.grad(loss, x)[0]

        noise = torch.zeros_like(x)
        if self.noise_type == NoiseType.UNFIFORM:
            noise.uniform_(-1, 1)
        if self.noise_type == NoiseType.NORMAL:
            noise.normal_()
        
        if self.sign_flag:
            grad = grad.sign()
        x = x + self.eps_iter*grad + self.noise_scale*math.sqrt(2*self.eps_iter) * noise

        x = self.projection(x, x_ref)
        return x

    def perturb(self, x, y=None):
        # Function computing several step of Langevin (interface)
        x, y = self._verify_and_process_inputs(x, y)

        x_ref = x.clone()
        x.requires_grad_()
        for i in range(self.nb_iter):
            x.data = self._perturb(x, y, x_ref=x_ref)
        return x.data
      