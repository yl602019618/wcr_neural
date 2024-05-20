import torch
import torch.nn as nn
import numpy as np
from torch import optim
from collections import OrderedDict
import time
import psutil
import os
import scipy.io
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import utils
import itertools
import logging
logger = logging.getLogger(__name__)
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

def plot_error(drift_fun,drift_net,diffusion_fun,diffusion_net,device,plot_type):
    if plot_type == '1d':
        plot_1d(drift_fun,drift_net,diffusion_fun,diffusion_net,device)
    elif plot_type == '2d':
        plot_2d(drift_fun,drift_net,diffusion_fun,diffusion_net,device)

def plot_1d(drift_fun,drift_net,diffusion_fun,diffusion_net,device):
    interval=[-2,2]
    n=100
    x = torch.linspace(interval[0], interval[1], n, device=device).unsqueeze(1)
    exact_drift = drift_fun(x) # n,1
    exact_diffusion = diffusion_fun(x)[...,0] # n,1
    with torch.no_grad():
        pred_drift = drift_net(x)# n,1
        pred_diffusion = torch.sqrt(diffusion_net(x)) # n,1
    plt.figure()
    plt.plot(x.cpu().detach().numpy(), exact_drift.cpu().detach().numpy(), label="exact")
    plt.plot(x.cpu().detach().numpy(), pred_drift.cpu().detach().numpy(), label="approximate")
    plt.xlabel("x")
    plt.ylabel("drift")
    plt.legend()
    img = wandb.Image(plt)
    wandb.log({'Image_drift': img})
    plt.close()
    
    plt.figure()
    plt.plot(x.cpu().detach().numpy(), exact_diffusion.cpu().detach().numpy(), label="exact")
    plt.plot(x.cpu().detach().numpy(), pred_diffusion.cpu().detach().numpy(), label="approximate")
    plt.xlabel("x")
    plt.ylabel("diffusion")
    plt.legend()
    img = wandb.Image(plt)
    wandb.log({'Image_diffusion': img})
    plt.close()


def plot_2d(drift_fun,drift_net,diffusion_fun,diffusion_net,device):
    interval=[-2,2]
    n=50
    x = torch.linspace(interval[0], interval[1], n, device=device)
    y = x
    X,Y = torch.meshgrid(x,y)
    data = torch.cat((X.unsqueeze(-1),Y.unsqueeze(-1)),dim  = -1)
    
    
    exact_drift = drift_fun(data)# n,2
    #self.net.load_state_dict(torch.load('1D_drift.pth'))
    exact_diffusion = diffusion_fun(x) # n,2,2
   
    pred_drift = drift_net(data)# n,2
    pred_diffusion = torch.sqrt(diffusion_net(x))# n,2

    plt.figure(figsize = (16,64))
    plt.subplot(4,2,1)
    plt.contourf(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), exact_drift.detach().cpu().numpy()[...,0],80, cmap='rainbow')
    plt.colorbar()
    #plt.savefig("results/exact1.png")
    #plt.show()
    #plt.close()
    plt.subplot(4,2,2)
    plt.contourf(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), exact_drift.detach().cpu().numpy()[...,1],80, cmap='rainbow')
    plt.colorbar()
    plt.subplot(4,2,3)
    plt.contourf(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), pred_drift.detach().cpu().numpy()[...,0],80, cmap='rainbow')
    plt.colorbar()
    plt.subplot(4,2,4)
    plt.contourf(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), pred_drift.detach().cpu().numpy()[...,1],80, cmap='rainbow')
    plt.colorbar()
    

    plt.subplot(4,2,5)
    plt.contourf(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), exact_diffusion.detach().cpu().numpy()[...,0,0],80, cmap='rainbow')
    plt.colorbar()
    #plt.savefig("results/exact1.png")
    #plt.show()
    #plt.close()
    plt.subplot(4,2,6)
    plt.contourf(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), exact_diffusion.detach().cpu().numpy()[...,1,1],80, cmap='rainbow')
    plt.colorbar()
    plt.subplot(4,2,7)
    plt.contourf(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), pred_diffusion.detach().cpu().numpy()[...,0],80, cmap='rainbow')
    plt.colorbar()
    plt.subplot(4,2,8)
    plt.contourf(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), pred_diffusion.detach().cpu().numpy()[...,1],80, cmap='rainbow')
    plt.colorbar()
    img = wandb.Image(plt)
    wandb.log({'Image': img})
    plt.close()