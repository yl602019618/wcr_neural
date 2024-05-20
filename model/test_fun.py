import torch
import torch.nn as nn
import numpy as np
import utils



class Gaussian(torch.nn.Module): 
    def __init__(self, mu, sigma, device):
        """mu: [gauss_num, dim]; sigma: 1"""
        super(Gaussian, self).__init__()
        '''
        mu: gauss, time, dim
        sigma: gauss, dim
        '''
        self.mu = mu.to(device).unsqueeze(2)  # [gauss_num, time, 1, dim]
        self.sigma = sigma.unsqueeze(1).unsqueeze(1) # [gauss_num,1,1, dim]
        self.gauss_num, self.dim = mu.shape[0],mu.shape[2]
        self.device = device
        self.g0 = None

    def gaussZero(self, x):
        #print(x.shape,self.mu.shape,self.sigma.shape)
        """x: [t, sample, dim], return [gauss , t, sample]"""
        func = 1
        for d in range(self.dim):
            func = func * 1 / (self.sigma * torch.sqrt(2 * torch.tensor(torch.pi))) * torch.exp(
                -0.5 * (x[:, :, d].unsqueeze(0) - self.mu[:, :, :, d]) ** 2 / self.sigma ** 2)
        return func

    def gaussFirst(self, x, g0):
        """x: [t, sample, dim], g0: [gauss, t, sample], return[gauss, t, sample, dim]"""
        func = torch.zeros(self.gauss_num, x.shape[0], x.shape[1], x.shape[2])
        for d in range(self.dim):
            func[:, :, :, d] = -(x[:, :, d].unsqueeze(0) - self.mu[:, :, :, d])/self.sigma**2 * g0
        return func

    def gaussSecond(self, x, g0):
        """x: [t, sample, dim], g0: [gauss, t, sample, 1], return[gauss, t, sample, dim, dim]"""
        func = torch.zeros(self.gauss_num, x.shape[0], x.shape[1], x.shape[2])
        for k in range(x.shape[2]):
            func[:, :, :, k] = (
                            -1/self.sigma**2 + (-(x[:, :, k].unsqueeze(0)-self.mu[:, :, :, k])/self.sigma**2)
                            * (-(x[:, :, k].unsqueeze(0)-self.mu[:, :, :, k])/self.sigma**2)
                            ) * g0
        return func
    
    def forward(self, x, diff_order=0):
        if self.g0 is None:
            self.g0 = self.gaussZero(x).to(self.device)
        if diff_order == 0:
            return self.g0
        elif diff_order == 1:
            return self.gaussFirst(x, self.g0).to(self.device)
        elif diff_order == 2:
            return self.gaussSecond(x, self.g0).to(self.device)
        else:
            raise RuntimeError("higher order derivatives of the gaussian has not bee implemented!")


class Gaussian_TD(torch.nn.Module): 
    def __init__(self, mu, sigma, device):
        """mu: [gauss_num,t, dim]; sigma: [gauss_num]"""
        super(Gaussian_TD, self).__init__()
        self.mu = mu.to(device).unsqueeze(2)   # [gauss_num, t, 1, dim]
        self.sigma = sigma.unsqueeze(1).unsqueeze(1) #gauss_num,1,1
        self.gauss_num,self.t_num, self.dim = mu.shape
        self.device = device
        self.g0 = None

    def gaussZero(self, x):
        """x: [t, sample, dim],
         self.mu: [gauss,t,1,dim] 
         return [gauss, t, sample]"""
        func = 1
        for d in range(self.dim):
            func = func * 1 / (self.sigma * torch.sqrt(2 * torch.tensor(torch.pi))) * torch.exp(
                -0.5 * (x[:, :, d].unsqueeze(0) - self.mu[:, :, :, d]) ** 2 / self.sigma ** 2)
        return func

    def gaussFirst(self, x, g0):
        """x: [t, sample, dim], 
        g0: [gauss, t, sample], 
        return[gauss, t, sample, dim]"""

        func = torch.zeros(self.gauss_num, x.shape[0], x.shape[1], x.shape[2])
        for d in range(self.dim):
            func[:, :, :, d] = -(x[:, :, d].unsqueeze(0) - self.mu[:, :, :, d])/self.sigma**2 * g0
        return func

    def gaussSecond(self, x, g0):
        """x: [t, sample, dim], 
        g0: [gauss, t, sample], 
        
        return[gauss, t, sample, dim, dim]"""
        func = torch.zeros(self.gauss_num, x.shape[0], x.shape[1], x.shape[2], x.shape[2])
        for k in range(x.shape[2]):
            for j in range(x.shape[2]):
                if k == j:
                    func[:, :, :, k, j] = (
                                    -1/self.sigma**2 + (-(x[:, :, k].unsqueeze(0)-self.mu[:, :, :, k])/self.sigma**2)
                                    * (-(x[:, :, j].unsqueeze(0)-self.mu[:, :, :, j])/self.sigma**2)
                                    ) * g0
                else:
                    pass
        return func
    
    def forward(self, x, diff_order=0):
        if self.g0 is None:
            self.g0 = self.gaussZero(x).to(self.device)
        if diff_order == 0:
            return self.g0
        elif diff_order == 1:
            return self.gaussFirst(x, self.g0).to(self.device)
        elif diff_order == 2:
            return self.gaussSecond(x, self.g0).to(self.device)
        else:
            raise RuntimeError("higher order derivatives of the gaussian has not bee implemented!")




    
    