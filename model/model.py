import torch
import torch.nn as nn
import numpy as np
from torch import optim
import time

import utils
import itertools
import logging
logger = logging.getLogger(__name__)
# from postprocess import compute_error1D,plot_drift1D
import matplotlib.pyplot as plt
import seaborn as sns
import pytorch_lightning as pl
import wandb
import matplotlib.pyplot as plt
from torch import optim
from utils.sampling import sampleTestFunc_all
from utils.loss import LpLoss, RRMSE, compute_error
from utils.plot import plot_error
class Model(nn.Module):
    def __init__(self, t, data, testFunc, drift, diffusion, net_drift,net_diffusion, cfg_train, cfg_nn, device):
        super(Model, self).__init__()
        '''
        t: observation time 
        t_number: obseravation snapshot number
        data: observation point cloud data
        device: cpu or cuda
        testFunc: test function
        net_drift: NN parameterization of drift function
        net_diffusion: NN parametrization of diffusion function
        ob_size: observation trajectory num
        sgd_ration: 
        plot_path: path to plot the visualization result
        dimension: dimension 
        variance_min/max: the variance interval of sampling test function
        gauss_num: total gauss sampling num
        td_type: temporal difference scheme
        gauss_samp_way: sampling method
        sgd_ratio: If one use sgd training, the ratio of one sgd batch of total train sample
        train_ratio:If one use early stopping, the train ratio of total sample
        error_type: how to calculate error
        plot_type: how to log the result 
        '''
        self.t = t #观测数据的时间轴
        self.t_number = len(t)
        self.data = data.to(device) #观测数据 (nt,n_sample,dim)
        self.device = device
        self.testFunc = testFunc #测试函数，Gaussian
        self.cfg_train = cfg_train
        self.cfg_nn = cfg_nn
        self.net_drift = net_drift.to(device) #表示漂移项等神经网络
        self.net_diffusion = net_diffusion.to(device)
        self.ob_size = data.shape[1] # 轨道数量
        self.sgd_ration = 0.8 #
        self.dimension = self.data.shape[-1]
        self.gauss_num = cfg_train["gauss_number"]
        self.td_type =cfg_train["LMM"]
        self.drift = drift
        self.diffusion = diffusion
        self.mean_samp_way = cfg_train["mean_samp_way"] 
        self.var_samp_way = cfg_train["var_samp_way"] 
        self.sgd_ratio = cfg_train["sgd_ratio"]
        self.train_ratio = 0.8
        self.samp_coef = cfg_train['samp_coef']
        self.error_type = cfg_train['error_type']
        self.plot_type = cfg_train['plot_type']
        self.sample_gaussian()

    def sample_gaussian(self):
        self.mu_list_all, self.variance = sampleTestFunc_all(data = self.data, 
                           samp_number = self.gauss_num, 
                           mean_samp_way = self.mean_samp_way, 
                           var_samp_way = self.var_samp_way, 
                           samp_coef = self.samp_coef,
                           device = self.device)
        self.sampleTestFunc()
        self.data_test = self.data.reshape(-1,self.dimension)
        index_test = np.arange(self.data_test.shape[0])
        np.random.shuffle(index_test)
        self.data_test = self.data_test[index_test[:10000],:]

    def sampleTestFunc(self):
        '''
        Compute the Gaussian and its derivative for the data point
        '''
        # for i in range(self.sampling_number):
        
            
            
        gaussn = self.testFunc(self.mu_list_all, self.variance, self.device)
        gaussnp1 = self.testFunc(self.mu_list_all, self.variance, self.device)
        gaussnp2 = self.testFunc(self.mu_list_all, self.variance, self.device)
        TX = self.data  # [t, sample, dim]

        self.gaussn_0 = gaussn(TX[:-2], diff_order=0) #gauss,t-2,sample
        self.gaussn_1 = gaussn(TX[:-2], diff_order=1) #gauss,t-2,sample,dim
        self.gaussn_2 = gaussn(TX[:-2], diff_order=2) #gauss,t-2,sample,dim,dim  

        self.gaussnp1_0 = gaussnp1(TX[1:-1], diff_order=0) #gauss,t-2,sample
        self.gaussnp1_1 = gaussnp1(TX[1:-1], diff_order=1) #gauss,t-2,sample,dim
        self.gaussnp1_2 = gaussnp1(TX[1:-1], diff_order=2) #gauss,t-2,sample,dim,dim

        self.gaussnp2_0 = gaussnp2(TX[2:], diff_order=0) #gauss,t-2,sample
        self.gaussnp2_1 = gaussnp2(TX[2:], diff_order=1) #gauss,t-2,sample,dim
        self.gaussnp2_2 = gaussnp2(TX[2:], diff_order=2) #gauss,t-2,sample,dim,dim

    def compute_loss(self,TX, mode = 'sgd'):
        '''
        Compute the residual loss for each frame
        '''

        drift_pred = self.net_drift(TX)   # [t, sample, dim]
        diffusion_pred = self.net_diffusion(TX) # [t,sample, dim]
        # gauss: gauss,t,sample
        # gauss1: gauss,t,sample,dim
        # gauss2: gauss,t,sample,dim,dim
        # drift_pred: t,sample.dim
        
        index_all = np.arange(self.gauss_num)
        if mode == 'train':
            index = index_all[:int(self.gauss_num)]
        elif mode == 'sgd':
            n = int(self.train_ratio*self.gauss_num)
            index = np.arange(n)
            np.random.shuffle(index)
            index = index[:int(self.sgd_ratio*n)]
        elif mode == 'val':
            n = int(self.train_ratio*self.gauss_num)
            index = np.arange(n)
            index = index[int(self.sgd_ratio*n):]
        


        # Compute the (mu,\grad Gauss)
        # drift:t-2,sample,dim -> 1,t-2,sample,dim
        # gaussn_1: gauss,t-2,sample,dim
        # mean: gauss*(t-2)
        An_1 = torch.mean(torch.sum(drift_pred[:-2,:,:].unsqueeze(0) * self.gaussn_1[index], dim=3), dim=2).view(-1) #gauss*t-2
        # Compute the (sigma,Dij Gauss)
        # self.D: sample,dim
        # gaussn_2: gauss,t-2,sample,dim
        # mean: gauss*(t-2)
        self.D = 1/2*(diffusion_pred)#  [t,sample, dim]
        
        An_2 = torch.mean(torch.einsum("ijkl,jkl->ijk", [self.gaussn_2[index], self.D[:-2,:,:]]), dim=2).view(-1) ##gauss*t

        An = An_1 + An_2

        Anp1_1 = torch.mean(torch.sum(drift_pred[1:-1,:,:].unsqueeze(0) * self.gaussnp1_1[index], dim=3), dim=2).view(-1)
        Anp1_2 = torch.mean(torch.einsum("ijkl,jkl->ijk", [self.gaussnp1_2[index], self.D[1:-1,:,:]]), dim=2).view(-1)
        Anp1 = Anp1_1 + Anp1_2

        Anp2_1 = torch.mean(torch.sum(drift_pred[2:,:,:].unsqueeze(0) * self.gaussnp2_1[index], dim=3), dim=2).view(-1)
        Anp2_2 = torch.mean(torch.einsum("ijkl,jkl->ijk", [self.gaussnp2_2[index], self.D[2:,:,:]]), dim=2).view(-1)
        Anp2 = Anp2_1 + Anp2_2
        

        # taking mean in the dimension of sample 
        rbn = torch.mean(self.gaussn_0[index], dim=2).view(-1) #gauss*t
        #rbnp1 = torch.mean(self.gaussnp1_0[index], dim=2).view(-1) #gauss*t
        rbnp2 = torch.mean(self.gaussnp2_0[index], dim=2).view(-1) #gauss*t
        
        
        dt = (torch.max(self.t)-torch.min(self.t)) / (self.t_number - 1)
        
        if self.td_type == 'LMM_3':
            Aq = (An + 4*Anp1 + Anp2) / 3 * dt
            bq = rbnp2 - rbn
        residual = Aq - bq
        #self.loss_buffer.append(self.loss)
        return residual

    def compute_error(self):
        with torch.no_grad():
            drift_error, diffusion_error = compute_error(drift = self.drift,
                                                         net_drift=self.net_drift,
                                                         diffusion=self.diffusion,
                                                         net_diffusion=self.net_diffusion,
                                                         device=self.device,
                                                         data=self.data,
                                                         error_type=self.error_type)
        return drift_error, diffusion_error
    def plot_error(self):
        with torch.no_grad():
            plot_error(drift_fun = self.drift,
                       drift_net = self.net_drift,
                       diffusion_fun = self.diffusion,
                       diffusion_net = self.net_diffusion,
                       device = self.device,
                       plot_type = self.plot_type)
    

class Model_pl(pl.LightningModule):
    def __init__(self, t, data, testFunc, drift, diffusion, net_drift,net_diffusion, cfg_train, cfg_nn, device):
        super(Model_pl, self).__init__()
        self.model = Model( t = t, 
                           data = data, 
                           testFunc = testFunc, 
                           drift = drift, 
                           diffusion = diffusion, 
                           net_drift = net_drift,
                           net_diffusion = net_diffusion, 
                           cfg_train = cfg_train, 
                           cfg_nn = cfg_nn, 
                           device = device)
        self.lr = cfg_nn['lr'] 
        self.step_size = cfg_nn['step_size'] 
        self.weight_decay = cfg_nn['weight_decay'] 
        self.gamma = cfg_nn['gamma'] 
        self.criterion = nn.MSELoss()#H1_loss(alpha = 0.1)
        self.criterion_val = nn.MSELoss()
        
        
        
    def forward(self, x,mode):
        loss = self.model.compute_loss(x,mode)
        return loss

    def training_step(self, batch: torch.Tensor, batch_idx):  
         
        residual = self(batch[0][0],'sgd')
        #print(out.shape,y.shape)
        #loss,l2, l_phy = self.criterion(out,y)#torch.mean(torch.abs(out.view(batch_size,-1)-10*y.view(batch_size,-1)) ** 2)
        loss = torch.sum(residual**2)
        self.log("loss", loss, on_epoch=True, prog_bar=True, logger=True)
        # self.log("loss_l2", l2, on_epoch=True, prog_bar=True, logger=True)
        # self.log("l_phy", l_phy, on_epoch=True, prog_bar=True, logger=True)
        # wandb.log({"loss": loss.item(),'loss_data':l2.item(),'loss_phy':l_phy.item()})
        wandb.log({"loss": loss.item()})
        return loss

    def validation_step(self, val_batch: torch.Tensor, batch_idx):
         
        residual_val = self(val_batch[0][0],'val')
        drift_error, diffusion_error = self.model.compute_error()
       
        val_residual = torch.sum(residual_val**2)

        val_loss = drift_error  + diffusion_error

        self.log('val_residual', val_residual, on_epoch=True, prog_bar=True, logger=True)
        wandb.log({'val_residual': val_residual.item()})
        self.log('drift_error', drift_error, on_epoch=True, prog_bar=True, logger=True)
        wandb.log({'drift_error': drift_error.item()})
        self.log('diffusion_error', diffusion_error, on_epoch=True, prog_bar=True, logger=True)
        wandb.log({'diffusion_error': diffusion_error.item()})
        
        self.log('val_loss',val_loss, on_epoch=True, prog_bar=True, logger=True)

        self.model.plot_error()

        return val_loss
    

    def configure_optimizers(self, optimizer=None, scheduler=None):
        if optimizer is None:
            optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if  scheduler is None:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
            #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = self.step_size, eta_min= self.eta_min)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler
        },
    }