import torch
import torch.autograd as autograd
import math 
# This should abstract the Langevin dynamics, make it easy to change the dynamics or propose new dynamics MCMC type.


class Projection:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self, x, x_ref):
        return x


class LinfProjection(Projection):
    def __init__(self, epsilon=0.3):
        super().__init__(epsilon)

    def __call__(self, x, x_ref):
        x = torch.max(torch.min(x, x_ref + self.epsilon), x_ref - self.epsilon)
        return x


class Langevin:
    def __init__(self, forward, n_lan, projection=LinfProjection(), lr=1., noise_scale=1., sign_flag=False):
        # `forward` should be a function (or class ?), that outputs a scalar. Not sure what the best way to implement this ?
        # `projection` is a class that project back onto the constraint set.
        self.forward = forward
        self.n_lan = n_lan
        self.projection = projection
        self.lr = lr
        self.sign_flag = sign_flag
        self.noise_scale = noise_scale

    def _step(self, x, y, x_ref=None):
        # Function that computes a single step of Langevin (hidden)
        # x_ref if specified is used to project back on the constraint set.
        loss = self.forward(x, y)
        grad = autograd.grad(loss, x)[0]
        noise = torch.randn_like(x)
        
        if self.sign_flag:
            grad = grad.sign()
        x = x + self.lr*grad + self.noise_scale*math.sqrt(2*self.lr) * noise

        if x_ref is not None:
            x = self.projection(x, x_ref)
        x = x.clamp(0, 1)
        return x

    def step(self, x, y, x_ref=None):
        # Function computing several step of Langevin (interface)
        for i in range(self.n_lan):
            x.data = self._step(x, y, x_ref=x_ref)
        return x
      