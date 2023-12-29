# This code belongs to the paper
#
# M. Piening, F. Altekrüger, J. Hertrich, P. Hagemann, A. Walther and G. Steidl (2023).
# Learning from small data sets: Patch-based regularizers in inverse problems for image reconstruction
# ArXiv preprint:2312.16611
#
# Please cite the paper, if you use the code.
#
# The script reproduces the posterior sampling for CT in the paper.

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import argparse

import utils
import regularizer.NFReg as NFmodel
import regularizer.EPLLReg as EPLLmodel
import regularizer.WReg as Wmodel
import regularizer.ALR as ALRmodel
#import regularizer.SuSReg as SuSmodel
import regularizer.SReg as Smodel

import dival
from dival import get_standard_dataset
import odl
from odl.contrib.torch import OperatorModule
from dival.util.torch_losses import poisson_loss
from functools import partial

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def langevin(img, patch_size, n_patches_out, regularizer, n_iter_max,
                learning_rate=0.03, lam=1, batch_size = 10):
    '''
    Reconstruction for given observation and given regularizer
    Implementation of unadjusted Langevin algorithm
    '''
    # fixed parameters
    img = img.to(DEVICE)
    center = False
    init = fbp(obs)
    pad_size = 4 # pad the image before extracting patches to avoid boundary effects
    pad = [pad_size]*4  
    # create patch extractors
    input_im2pat = utils.patch_extractor(patch_size, pad=False, center=center)
        
    # define the poisson loss
    photons_per_pixel = 4096
    mu_max = 81.35858
    criterion = partial(poisson_loss,photons_per_pixel=photons_per_pixel,
                        mu_max=mu_max)
        
    # intialize optimizer for image
    fake_img = init.detach().clone()
    fake_img = fake_img.tile(batch_size,1,1,1)
    # add random noise for more variety
    fake_img += 0.01*torch.randn_like(fake_img)
    
    beta_sqrt = torch.sqrt(2 * torch.tensor(learning_rate))

    # Main loop
    init = init.tile(batch_size,1,1,1)
    for it in tqdm(range(n_iter_max)):
        #compute regularizer for each image separately
        grads = torch.tensor([],device=DEVICE)
        for i in range(batch_size):
            fake = fake_img[i].clone().unsqueeze(0).requires_grad_()
            tmp_img = F.pad(fake, pad = pad, mode= 'reflect')
            tmp = input_im2pat(tmp_img, n_patches_out)
            loss = criterion(operator(fake),obs) + lam  * regularizer(tmp)
            grad = torch.autograd.grad(loss, fake)[0]
            grads = torch.cat([grads,grad],dim=0)
        # ULA
        fake_img.data = fake_img.data - grads*learning_rate + (beta_sqrt * 
                                            torch.randn_like(fake_img))
    return fake_img
    
if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch', type=int, default=6,
                        help='Patch size for training')
    parser.add_argument('--reg', type=str, default='nf', 
                        choices=['nf', 'epll', 'w', 'alr', 'sus', 's'],
                        help='Choose between different regularizers. \
                              You have the choice between patchNR (nf), \
                              EPLL (epll), WPP (w), ALR (alr), \
                              Sinkhorn (s) and Semi-unbalanced Sinkhorn (sus).')
    parser.add_argument('--angle', type = str, default = 'full', 
                        choices=['lim','full'],
                        help='Choose between limited angle (lim) and \
                              full angle (full) CT.')
    args = parser.parse_args()
    #load images
    dataset = get_standard_dataset('lodopab', impl='astra_cuda')
    test = dataset.create_torch_dataset(part='test',
                        reshape=((1,1,) + dataset.space[0].shape,
                        (1,1,) + dataset.space[1].shape))
    ray_trafo = dataset.ray_trafo                    
    if args.angle == 'lim':
        lim_dataset = dival.datasets.angle_subset_dataset.AngleSubsetDataset(dataset,
	                   slice(100,900),impl='astra_cuda')                   
        test = lim_dataset.create_torch_dataset(part='test',
                        reshape=((1,1,) + lim_dataset.space[0].shape,
                        (1,1,) + lim_dataset.space[1].shape))  
        ray_trafo = lim_dataset.ray_trafo                            
    
    gt = test[64][1].to(DEVICE)
    obs = test[64][0].to(DEVICE)

    #load operator and FBP
    operator = OperatorModule(ray_trafo).to(DEVICE)
    fbp = odl.tomo.analytic.filtered_back_projection.fbp_op(ray_trafo,
	                    filter_type = 'Hann', frequency_scaling = 0.641025641025641)
    fbp = OperatorModule(fbp)

    patch_size = args.patch
    lrate = 5e-7    
    
    # patchNR
    if args.reg == 'nf':
        try:
            weights = torch.load(f'model_weights/patchNR_ct_p{args.patch}.pth')
        except:
            raise Exception('Model weights cannot be found. Please ensure \
                    that weights are named correctly or train the model \
                    by calling the skript of the regularizer and set the flag \
                    "train" to True.')
        reg_model = NFmodel.NF_regularizer(patch_size = weights['patch'], 
                            num_layers = weights['layer'], 
                            subnet_nodes = weights['nodes'])
        reg_model.NF.load_state_dict(weights['state_dict'])
        n_pat = 40000
        lam = 6000
        iteration = 50000
        lrate = 2e-7
        
    # EPLL
    elif args.reg == 'epll':
        try:
            path = f'model_weights/gmm_ct_p{args.patch}.joblib'
        except:
            raise Exception('Model weights cannot be found. Please ensure \
                    that weights are named correctly or train the model \
                    by calling the skript of the regularizer and set the flag \
                    "train" to True.')
        reg_model = EPLLmodel.EPLL_regularizer(path)
        lam = 7000
        n_pat = 30000
        iteration = 20000

    # ALR
    elif args.reg == 'alr':
        try:
            weights = torch.load(f'model_weights/ALR_ct_p{args.patch}.pth')
        except:
            raise Exception('Model weights cannot be found. Please ensure \
                    that weights are named correctly or train the model \
                    by calling the skript of the regularizer and set the flag \
                    "train" to True.')
        reg_model = ALRmodel.ALR(patch_size=patch_size)
        reg_model.localAReg_CNN.load_state_dict(weights['state_dict'])
        lam = 4*1e+6
        n_pat = 40000
        iteration = 30000

    elif args.reg in ['w','s','sus']:
        # load reference data
        from dival import get_standard_dataset
        dataset = get_standard_dataset('lodopab', impl='astra_cuda')
        train = dataset.create_torch_dataset(part='train',
                            reshape=((1,1,) + dataset.space[0].shape,
                            (1,1,) + dataset.space[1].shape))
        example_img = torch.cat([train[3][1],train[5][1],train[8][1],
                        train[11][1],train[37][1],train[75][1]]).to(DEVICE)
        extractor = utils.patch_extractor(patch_size)
        learn = torch.tensor([],device=DEVICE)
        for i in range(len(example_img)):
            learn = torch.cat([learn,extractor(example_img[i].unsqueeze(0))],dim=0)
        idx = torch.tensor(np.random.choice(learn.shape[0],10000))
        data = learn[idx]
        
        if args.reg == 'w':
            reg_model = Wmodel.WassersteinPrior(reference_img=data,
                                        iter_psi=10,restart_psi=True)
            lam = 5*1e+7#4*1e+6
            n_pat = 40000
            iteration = 20000
            
        # Sinkhorn
        elif args.reg == 's':
            reg_model = Smodel.SinkhornPrior(data,scaling=0.5,blur=.01)
            lam = 7*1e+7
            n_pat = 40000
            iteration = 10000
            
        # Semi-unbalanced Sinkhorn
        elif args.reg == 'sus':
            reg_model = SuSmodel.SuSPP(data,reach=.005,scaling=.5,
                                            blur=0.01)
            lam = 1*1e+8
            n_pat = 40000
            iteration = 10000

    fake_img = langevin(obs,patch_size = patch_size, 
                        n_patches_out = n_pat, regularizer = reg_model, 
                        n_iter_max = iteration, learning_rate=lrate, 
                        lam=lam)

    # save images
    utils.save_img(torch.mean(fake_img.detach(),dim=0,keepdim=True),
                    f'results/langevin_ct/mean_{args.reg}')
    std = torch.std(fake_img.detach(),dim=0,keepdim=True)
    std -= std.min()
    utils.save_img(std/std.max(), f'results/langevin_ct/std_{args.reg}')
