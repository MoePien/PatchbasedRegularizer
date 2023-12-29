# This code belongs to the paper
#
# M. Piening, F. Altekrüger, J. Hertrich, P. Hagemann, A. Walther and G. Steidl (2023).
# Learning from small data sets: Patch-based regularizers in inverse problems for image reconstruction
# ArXiv preprint:2312.16611
#
# Please cite the paper, if you use the code.
#
# The script reproduces the zero-shot superresolution example in the paper.

import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import os

import utils
import regularizer.NFReg as NFmodel
import regularizer.EPLLReg as EPLLmodel
import regularizer.WReg as Wmodel
import regularizer.ALR as ALRmodel
#import regularizer.SuSReg as SuSmodel
import regularizer.SReg as Smodel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Downsample(input_img, scale = 0.5):
    ''' 
    downsamples an img by factor 2 using gaussian downsample
    '''
    if scale > 1:
        print('Error. Scale factor is larger than 1.')
        return
    gaussian_std = 1
    kernel_size = 16
    gaussian_down = utils.gaussian_downsample(kernel_size,gaussian_std,
                                        int(1/scale),pad=False)

    #gaussian downsample
    out = gaussian_down(input_img).to(DEVICE)
    return out

def reconstruct_MAP(img, lam, patch_size, n_patches_out, regularizer, 
                    n_iter_max, n_reference, operator, learning_rate=0.03):
    '''
    Reconstruction for given observation and given regularizer
    '''
    # fixed parameters
    lr_img = img.to(DEVICE)
    center = False 
    init = F.interpolate(img, scale_factor=1/scale_factor, mode='bicubic')

    # create patch extractors
    input_im2pat = utils.patch_extractor(patch_size, pad=False, center=center)

    #train regularizer
    regularizer.train(data=input_im2pat(lr_img,n_reference),
                save_name=f'model_weights/{args.reg}_zeroshot.pth',
                optimizer_steps=train_steps,batch_size=batch_size, #nf
                max_iter=max_iter,tol=tol,n_components=n_components, #epll
                epochs = epochs) #alr
    regularizer.load_weights(f'model_weights/{args.reg}_zeroshot.pth')
    if os.path.exists(f'model_weights/{args.reg}_zeroshot.pth'):
        os.remove(f'model_weights/{args.reg}_zeroshot.pth')
        
    # intialize optimizer for image
    fake_img = init.clone().detach().requires_grad_(True)
    optim_img = torch.optim.Adam([fake_img], lr=learning_rate)

    # Main loop
    for it in tqdm(range(n_iter_max)):
        optim_img.zero_grad()
        #extract patches
        tmp = F.pad(fake_img, pad = (7,7,7,7), mode= 'reflect')
        fake_data = input_im2pat(tmp,n_patches_out)
        #reg
        reg = regularizer(fake_data)
        #data fidelity
        data_fid = torch.sum((operator(tmp) - lr_img)**2)
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
    args = parser.parse_args()
    
    #input parameters
    patch_size = args.patch
    train_steps = 0
    batch_size = 0
    max_iter = 0
    n_components = 0
    tol = 1
    epochs = 0
    n_reference = int(1e+10)
    
    # patchNR
    if args.reg == 'nf':
        reg_model = NFmodel.NF_regularizer(patch_size = patch_size, 
                            num_layers = 4, subnet_nodes = 512)
        lam = 0.25
        n_pat = 70000
        iteration = 200
        lrate = 0.03
        #params for training
        train_steps = 5000
        batch_size = 512
        
    # EPLL
    elif args.reg == 'epll':
        reg_model = EPLLmodel.EPLL_regularizer()
        lam = 0.35
        n_pat = 30000
        iteration = 200
        lrate = 0.03
        tol = 1e-4
        max_iter = 500
        n_components = 50
    
    # WPP
    elif args.reg == 'w':
        reg_model = Wmodel.WassersteinPrior(iter_psi=10,restart_psi=True)
        lam = 106
        n_reference = 10000
        n_pat = 1000000
        iteration = 300
        lrate = 0.03
        
    # ALR
    elif args.reg == 'alr':
        reg_model = ALRmodel.ALR(patch_size=patch_size)
        lam = 70
        n_pat = 100000
        iteration = 100
        lrate = 0.03
        epochs = 10
        batch_size = 64
    
    # Sinkhorn
    elif args.reg == 's':
        reg_model = Smodel.SinkhornPrior(scaling=0.5,blur=0.01)
        lam = 106
        n_reference = 10000
        n_pat = 50000
        lrate = 0.01
        iteration = 100
    
    # Semi-unbalanced Sinkhorn
    elif args.reg == 'sus':
        reg_model = SuSmodel.SuSPP(reach=0.005,scaling=.5,blur=.05)
        lam = 306
        n_pat = 30000
        n_reference = 10000
        lrate = 0.003
        iteration = 100
    
    scale_factor = 0.5
    hr = utils.imread('data/bsd_img.jpg',as_gray=True)[...,1:,1:]
    operator = lambda x: Downsample(x,scale=scale_factor) 
    lr = operator(hr)
    lr = lr + 0.01*torch.randn_like(lr)

    rec = reconstruct_MAP(lr,lam = lam, patch_size = patch_size, n_patches_out = n_pat,
                  regularizer = reg_model, n_iter_max = iteration, learning_rate=lrate,
                  n_reference = n_reference,operator=operator)
    utils.save_img(rec,f'results/zeroshot_sr/result_{args.reg}')

