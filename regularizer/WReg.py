# This code belongs to the paper
#
# M. Piening, F. Altekrüger, J. Hertrich, P. Hagemann, A. Walther and G. Steidl (2023).
# Learning from small data sets: Patch-based regularizers in inverse problems for image reconstruction
# ArXiv preprint:2312.16611
#
# Please cite the paper, if you use the code.
#
# The script implements the WPP used in the paper.

import torch
from torch import nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class semidual(nn.Module):
    '''
    Computes the semi-dual loss between inputy and inputx for the dual 
    variable psi
    '''    
    def __init__(self, inputy, device=DEVICE, usekeops=False):
        super(semidual, self).__init__()
        self.psi = nn.Parameter(torch.zeros(inputy.shape[0], device=device))
        self.yt = inputy.transpose(1,0)	
        self.usekeops = usekeops
        self.y2 = torch.sum(self.yt **2,0,keepdim=True)
        self.patch_weights = torch.ones(inputy.shape[0],device=device,
                                        dtype=torch.float)
        
    def forward(self, inputx):
        if self.usekeops:
            from pykeops.torch import LazyTensor
            y = self.yt.transpose(1,0)
            x_i = LazyTensor(inputx.unsqueeze(1).contiguous())
            y_j = LazyTensor(y.unsqueeze(0).contiguous())
            v_j = LazyTensor(self.psi.unsqueeze(0).unsqueeze(2).contiguous())
            sx2_i = LazyTensor(torch.sum(inputx**2,1,keepdim=True).unsqueeze(2).contiguous())
            sy2_j = LazyTensor(self.y2.unsqueeze(2).contiguous())
            rmv = sx2_i + sy2_j - 2*(x_i*y_j).sum(-1) - v_j
            amin = rmv.argmin(dim=1).view(-1)
            loss = (torch.mean(torch.sum((inputx-y[amin,:])**2,1)-self.psi[amin])+
                torch.sum(self.patch_weights*self.psi)/torch.sum(self.patch_weights)
                    )
        else:
            cxy = (torch.sum(inputx**2,1,keepdim=True) + self.y2 -
                    2*torch.matmul(inputx,self.yt))
            loss = (torch.mean(torch.min(cxy - self.psi.unsqueeze(0),1)[0]) + 
                torch.sum(self.patch_weights*self.psi)/torch.sum(self.patch_weights)
                )
        return loss

class WassersteinPrior(nn.Module):
    def __init__(self, reference_img=None,iter_psi=10,iter_psi_start=50,
                    usekeops=True,restart_psi=False,**kwargs):
        super(WassersteinPrior, self).__init__()
        self.reference_img = reference_img
        self.reach = None
        self.regularizer_function = lambda x: self.wloss(x,reference_img)
        self.keops = usekeops
        self.iter_psi = iter_psi
        self.iter_psi_start = iter_psi_start
        self.restart_psi = restart_psi
        if self.reference_img is not None:
            self.semidual_loss = semidual(self.reference_img,usekeops=self.keops)
        
    def forward(self, fake_data):
        '''
        computes the WPP regularizer 
        '''
        return self.regularizer_function(fake_data)

    def wloss(self,fake_data,reference_img):
        '''
        computes the W_2^2 distance via the dual formulation
        '''
        # warm start of maximizer psi
        if torch.max(torch.abs(self.semidual_loss.psi)) == 0:
            self.iter_psi_tmp = self.iter_psi_start
        
        if self.restart_psi:
            self.semidual_loss = semidual(self.reference_img,usekeops=self.keops)
            
        #gradient ascent to find maximizer psi for dual formulation of W2^2
        fake_tmp = fake_data.detach()
        optim_psi = torch.optim.Adam([self.semidual_loss.psi], lr=1e-3)
        for i in range(self.iter_psi_tmp):
            sem = -self.semidual_loss(fake_tmp)
            optim_psi.zero_grad()
            sem.backward()
            optim_psi.step()
        self.iter_psi_tmp = max(self.iter_psi_tmp//2,self.iter_psi)
        return self.semidual_loss(fake_data)

    def train(self,data,**kwargs):
        self.reference_img = data
        self.regularizer_function = lambda x: self.wloss(x,self.reference_img)
        self.semidual_loss = semidual(self.reference_img,usekeops=self.keops)
        return
        
    def load_weights(self,file_name):
        return


