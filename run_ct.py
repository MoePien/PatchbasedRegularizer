# This code belongs to the paper
#
# M. Piening, F. Altekrüger, J. Hertrich, P. Hagemann, A. Walther and G. Steidl (2023).
# Learning from small data sets: Patch-based regularizers in inverse problems for image reconstruction
# ArXiv preprint:2312.16611
#
# Please cite the paper, if you use the code.
#
# The script reproduces the CT example in the paper.

import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import argparse

import utils
import regularizer.NFReg as NFmodel
import regularizer.EPLLReg as EPLLmodel
import regularizer.WReg as Wmodel
import regularizer.ALR as ALRmodel
import regularizer.SuSReg as SuSmodel
import regularizer.SReg as Smodel

import dival
from dival import get_standard_dataset
import odl
from odl.contrib.torch import OperatorModule
from dival.util.torch_losses import poisson_loss
from functools import partial

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def reconstruct_MAP(img, operator, lam, patch_size, n_patches_out, regularizer,
                    n_iter_max,learning_rate=0.03):
    '''
    Reconstruction for given observation and given regularizer
    '''
    # fixed parameters
    obs = img.to(DEVICE)
    center = False 
    init = fbp(obs)
    pad_size = 4 #pad the image before extracting patches to avoid boundary effects
    pad = [pad_size]*4  

    # create patch extractors
    input_im2pat = utils.patch_extractor(patch_size, pad=False, center=center)

    #define the poisson loss
    photons_per_pixel = 4096
    mu_max = 81.35858
    criterion = partial(poisson_loss,photons_per_pixel=photons_per_pixel,
                        mu_max=mu_max)
    
    # intialize optimizer for image
    fake_img = init.clone().detach().requires_grad_(True)
    optim_img = torch.optim.Adam([fake_img], lr=learning_rate)

    # Main loop
    for it in tqdm(range(n_iter_max)):
        optim_img.zero_grad()
        #extract patches
        tmp = F.pad(fake_img, pad = pad, mode= 'reflect')
        fake_data = input_im2pat(tmp,n_patches_out)
        #reg
        reg = regularizer(fake_data)
        #data fidelity
        data_fid = criterion(operator(fake_img),obs)
        #loss
        loss =  data_fid + lam*reg
        loss.backward()
        optim_img.step()    
    return fake_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reg', type=str, default='nf', 
                        choices=['nf', 'epll', 'w', 'alr', 'sus', 's'],
                        help='Choose between different regularizers. \
                              You have the choice between patchNR (nf), \
                              EPLL (epll), WPP (w), ALR (alr), \
                              Sinkhorn (s) and Semi-unbalanced Sinkhorn (sus).')
    parser.add_argument('--patch', type=int, default=6,
                        help='Patch size for training')
    parser.add_argument('--angle', type = str, default = 'full', 
                        choices=['lim','full'],
                        help='Choose between limited angle (lim) and \
                              full angle (full) CT.')
    args = parser.parse_args()
    #input parameters
    patch_size = args.patch
    
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
        lam = 700
        n_pat = 40000
        iteration = 300
        if args.angle == 'lim':
            iteration = 3000
        lrate = 0.005
        
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
        lam = 1000
        n_pat = 30000
        iteration = 300
        if args.angle == 'lim':
            iteration = 3000
        lrate = 0.01

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
        lam = 5.5*1e+5
        n_pat = 40000
        iteration = 300
        if args.angle == 'lim':
            iteration = 3000
        lrate = 0.005
    
    # WPP
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
            lam = 4*1e+6
            n_pat = 1000000
            iteration = 300
            if args.angle == 'lim':
                iteration = 3000
            lrate = 0.001
        
        # Sinkhorn
        elif args.reg == 's':
            reg_model = Smodel.SinkhornPrior(data,scaling=0.5,blur=.01)
            lam = 1*1e+7
            n_pat = 40000
            lrate = 0.001
            iteration = 100
            if args.angle == 'lim':
                iteration = 1000
    
        # Semi-unbalanced Sinkhorn
        elif args.reg == 'sus':
            reg_model = SuSmodel.SuSPP(data,reach=.005,scaling=.5,
                                            blur=0.01)
            lam = 1*1e+7
            n_pat = 40000
            lrate = 0.002
            iteration = 200
            if args.angle == 'lim':
                iteration = 500

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
    
    gt = test[0][1].to(DEVICE)
    obs = test[0][0].to(DEVICE)
    
    #load operator and FBP
    operator = OperatorModule(ray_trafo).to(DEVICE)
    fbp = odl.tomo.analytic.filtered_back_projection.fbp_op(ray_trafo,
	                    filter_type = 'Hann', frequency_scaling = 0.641025641025641)
    fbp = OperatorModule(fbp)
    
    rec = reconstruct_MAP(obs, operator = operator,lam = lam, 
                patch_size = patch_size, n_patches_out = n_pat, 
                regularizer = reg_model, n_iter_max = iteration, 
                learning_rate = lrate)
    utils.save_img(rec,f'results/ct/result_{args.reg}_{args.angle}')
    torch.save(rec,f'results/ct/tens_{args.reg}_{args.angle}.pt')



