import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def model_select(model):
    if model == 'pdcd':
        drift = drift_poly3
        diffusion = diffusion_constant
    return drift, diffusion

def drift_poly3(x):
    '''
    x:sample,dim
    out: sample,dim
    '''
    return x - x ** 3

def diffusion_constant(x):
    '''
    n,1
    '''
    diag = torch.eye(x.shape[1]).cuda()
    return diag.unsqueeze(0).repeat(x.shape[0],1,1)

