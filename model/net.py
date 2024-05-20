import torch
import torch.nn as nn
import numpy as np
import utils


def net_select(cfg):
    params_net = {"Tanh": nn.Tanh(), "Tanhshrink": nn.Tanhshrink(), "ReLU": nn.ReLU(), "ReLU6": nn.ReLU6(),
                  "LeakyReLU": nn.LeakyReLU(negative_slope=0.2),"GeLU":nn.GELU()}
    dim = cfg['dim']
    width = cfg['width']
    depth = cfg['depth']
    activation = cfg['activation']
    net_drift = DNN(inputsize=dim, width=width, depth=depth, outputsize=dim, params=params_net, activation=activation)
    net_diffusion = DNN_diffusion(inputsize=dim, width=width, depth=depth, outputsize=dim, params=params_net, activation=activation)
    return net_drift, net_diffusion

class DNN(nn.Module):
    def __init__(self, inputsize: int, width: int, depth: int, outputsize: int, params: dict, activation="Tanh"):
        super().__init__()
        self.input = nn.Linear(inputsize, width)
        self.hidden = nn.Linear(width, width)
        self.output = nn.Linear(width, outputsize)
        self.net = nn.Sequential(*[
            self.input,
            *[params[activation], self.hidden] * depth,
            params[activation],
            self.output,
        ])
        self.net.double()

    def forward(self, x: torch.Tensor):
        return self.net(x.double())


class softplus(nn.Module):
    """Fully connected layer """
    def __init__(self, std_min = 1e-13):
        super().__init__()
        self.act = nn.Softplus()
        self.std_min  = std_min
    def forward(self, x):
        return self.act(x)+self.std_min

    
class DNN_diffusion(nn.Module):
    def __init__(self, inputsize: int, width: int, depth: int, outputsize: int, params: dict, activation="Tanh"):
        super().__init__()
        self.input = nn.Linear(inputsize, width)
        self.hidden = nn.Linear(width, width)
        self.output = nn.Linear(width, outputsize)
        self.act = softplus()
        self.net = nn.Sequential(*[
            self.input,
            *[params[activation], self.hidden] * depth,
            params[activation],
            self.output,self.act
        ])
        self.net.double()

    def forward(self, x: torch.Tensor):
        return self.net(x.double())
    
    
class DNN_sep(nn.Module):
    def __init__(self, inputsize: int, width: int, depth: int, outputsize: int, params: dict, activation="Tanh"):
        super().__init__()
        self.input = nn.Linear(inputsize//2, width)
        self.hidden = nn.Linear(width, width)
        self.output = nn.Linear(width, outputsize//2)
        self.net = nn.Sequential(*[
            self.input,
            *[params[activation], self.hidden] * depth,
            params[activation],
            self.output,
        ])


        self.input1 = nn.Linear(inputsize//2, width)
        self.hidden1 = nn.Linear(width, width)
        self.output1 = nn.Linear(width, outputsize//2)
        self.net1 = nn.Sequential(*[
            self.input1,
            *[params[activation], self.hidden] * depth,
            params[activation],
            self.output1,
        ])
    def forward(self, x: torch.Tensor):
        out = torch.cat((self.net(x[...,0:1]),self.net1(x[...,1:2])),dim = -1)
        return out

