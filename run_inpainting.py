# This code belongs to the paper
#
# M. Piening, F. Altekrüger, J. Hertrich, P. Hagemann, A. Walther and G. Steidl (2023).
# Learning from small data sets: Patch-based regularizers in inverse problems for image reconstruction
# ArXiv preprint:2312.16611
#
# Please cite the paper, if you use the code.
#
# The script reproduces the inpainting example in the paper.

import torch
import torchvision
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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_mask(img, mcenter, mboundary):
    """
    creates inpainting mask
    """
    mask = torch.ones_like(img)
    mask[:,:,mcenter[0]-mboundary:mcenter[0]+mboundary,mcenter[1]-mboundary:mcenter[1]+mboundary] = 0
    return mask

def mask_image(img, img_mask, invert=True, val_to_fill=np.nan):
    """
    applies inpainting mask to image
    """
    if invert:
        img[~img_mask.to(bool)] = val_to_fill
    else:
        img[img_mask.to(bool)] = val_to_fill
    return img

def mask_learn_image(img, img_mask, img_mask_learn, val_to_fill=np.nan):
    """
    applies mask for the reference patches
    """
    img[~img_mask.to(bool) + img_mask_learn.to(bool)] = val_to_fill
    return img

def reconstruct_MAP(img, patch_size, n_patches_out, regularizer, n_iter_max, lam,
                    img_mask, img_mask_learn, n_reference, learning_rate=0.03):

    '''
    Reconstruction for given observation and given regularizer
    '''
    # fixed parameters
    corr_img = img.to(DEVICE)
    center = False
    init = corr_img.detach().clone()
    init[~img_mask.to(bool)] = mask_learn_image(init, img_mask, img_mask_learn, 
                                    val_to_fill=np.nan).nanmean()
    # create patch extractors
    input_im2pat = utils.patch_extractor(patch_size, pad=False, center=center)

    #train regularizer
    learn_img = mask_learn_image(hr.detach().clone(), img_mask, img_mask_learn, np.nan)
    learn = input_im2pat(learn_img, n_reference)
    regularizer.train(data=learn,save_name=f'model_weights/{args.reg}_zeroshot_inp.pth',
                optimizer_steps=train_steps,batch_size=batch_size, #nf
                max_iter=max_iter,tol=tol,n_components=n_components,  #epll
                epochs = epochs) #alr
    regularizer.load_weights(f'model_weights/{args.reg}_zeroshot_inp.pth')
    if os.path.exists(f'model_weights/{args.reg}_zeroshot_inp.pth'):
        os.remove(f'model_weights/{args.reg}_zeroshot_inp.pth')

    # intialize optimizer for image
    fake_img = init.detach().clone()

    fill_area = fake_img[~img_mask.to(bool)].detach().clone()
    #add random noise
    fill_area += 0.5*torch.randn_like(fill_area)
    fill_area = fill_area.requires_grad_(True)
    
    optim_img = torch.optim.Adam([fill_area], lr=learning_rate)
    fake_img[~img_mask.to(bool)] = fill_area

    # Main loop
    progress_bar = tqdm(range(n_iter_max), total=n_iter_max, position=0, leave=True)
    for it in progress_bar:
        optim_img.zero_grad()
        #extract patches
        fake_img = init.detach().clone()
        fake_img[~img_mask.to(bool)] = fill_area
        fake_data = input_im2pat(mask_image(fake_img, img_mask_learn, 
                    invert=False, val_to_fill=np.nan), n_patches_out)
        #reg
        reg = regularizer(fake_data)
        #loss
        loss = lam*reg
        loss.backward(retain_graph=True)
        optim_img.step()
        descr = f"Regularization: {(reg).detach().cpu().item():.4f}."
        progress_bar.set_description(descr)
    fake_img = corr_img.detach().clone()
    fake_img[~img_mask.to(bool)] = fill_area
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
    args = parser.parse_args()
    hr = utils.imread(f'data/butterfly.png', as_gray=True)
    mboundary = 20
    mcenter = (180, 180)
    learn_boundary = 40
    img_mask = create_mask(hr, mcenter, mboundary)
    img_mask_learn = create_mask(hr, mcenter, mboundary + learn_boundary)
    
    obs = mask_image(hr, img_mask, invert=True, val_to_fill=0)

    patch_size = args.patch
    train_steps = 0
    batch_size = 0
    max_iter = 0
    n_components = 0
    tol = 1
    epochs = 0
    n_reference = 0
    lam = 1

    # patchNR
    if args.reg == 'nf':
        reg_model = NFmodel.NF_regularizer(patch_size=patch_size)
        n_pat_out = 50000
        #params for training
        train_steps = 10000
        batch_size = 512
        iteration = 15000
        lrate = 0.03
        lam=100
        
    # EPLL
    elif args.reg == 'epll':
        reg_model = EPLLmodel.EPLL_regularizer()
        n_pat_out = 50000
        tol = 5e-5
        max_iter = 500
        n_components = 50
        iteration = 10000
        lrate = 0.01

    # ALR
    elif args.reg == 'alr':
        reg_model = ALRmodel.ALR(patch_size=patch_size)
        n_pat_out = 50000
        epochs = 50
        batch_size = 128
        iteration = 400
        lrate = 0.03

    # WPP
    elif args.reg == 'w':
        reg_model = Wmodel.WassersteinPrior(iter_psi = 30, iter_psi_start = 1000,
                                restart_psi=True)
        n_reference = 10000
        n_pat_out = 300000
        lam = 100
        iteration = 800
        lrate = 0.03

    # Sinkhorn
    elif args.reg == 's':
        reg_model = Smodel.SinkhornPrior()
        n_reference = 10000
        n_pat_out = 5000
        iteration = 1000
        lrate=0.03
        
    # Semi-unbalanced Sinkhorn
    elif args.reg == 'sus':
        n_reference = 10000
        n_pat_out = 5000
        reg_model = SuSmodel.SuSPP(reach=10)
        iteration = 400
        lam = 1
        lrate=0.03

    rec = reconstruct_MAP(obs,patch_size = patch_size, n_patches_out = n_pat_out,
                    regularizer = reg_model, n_iter_max = iteration, lam=lam,
                    img_mask=img_mask, img_mask_learn=img_mask_learn,
                    n_reference = n_reference,learning_rate=lrate)
    utils.save_img(rec, f'results/zeroshot_in/result_{args.reg}')


