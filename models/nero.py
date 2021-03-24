import torch
from torch.optim.optimizer import Optimizer

def neuron_norm(x):
    if x.dim() > 1:
        view_shape = [x.shape[0]] + [1]*(x.dim()-1)
        x = x.view(x.shape[0],-1)
        return x.norm(dim=1).view(*view_shape)
    else:
        return x.abs()

def neuron_mean(x):
    if x.dim() > 1:
        view_shape = [x.shape[0]] + [1]*(x.dim()-1)
        x = x.view(x.shape[0],-1)
        return x.mean(dim=1).view(*view_shape)
    else:
        raise Exception("neuron_mean not defined on 1D tensors.")

class Nero(Optimizer):

    def __init__(self, params, lr=0.01, beta=0.999, constraints=True):
        self.beta = beta
        self.constraints = constraints
        defaults = dict(lr=lr)
        super(Nero, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                if self.constraints and p.dim() > 1:
                    p.data -= neuron_mean(p)
                    p.data /= neuron_norm(p)
                state = self.state[p]
                state['step'] = 0
                state['exp_avg_sq'] = torch.zeros_like(neuron_norm(p))
                state['scale'] = neuron_norm(p).mean()
                if state['scale'] == 0.0:
                    state['scale'] = 0.01

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                state['step'] += 1
                bias_correction = 1 - self.beta ** state['step']
                state['exp_avg_sq'] = self.beta * state['exp_avg_sq'] + (1-self.beta) * neuron_norm(p.grad)**2

                grad_normed = p.grad / (state['exp_avg_sq']/bias_correction).sqrt()
                grad_normed[torch.isnan(grad_normed)] = 0

                p.data -= group['lr'] * state['scale'] * grad_normed

                if self.constraints and p.dim() > 1:
                    p.data -= neuron_mean(p)
                    p.data /= neuron_norm(p)

        return loss
