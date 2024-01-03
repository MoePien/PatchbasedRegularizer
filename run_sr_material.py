# This code belongs to the paper
#
# M. Piening, F. Altekrüger, J. Hertrich, P. Hagemann, A. Walther and G. Steidl (2023).
# Learning from small data sets: Patch-based regularizers in inverse problems for image reconstruction
# ArXiv preprint:2312.16611
#
# Please cite the paper, if you use the code.
#
# The script reproduces the superresolution example in the paper.

import torch
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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Downsample(input_img, scale = 0.25):
    ''' 
    downsamples an img by factor 4 using gaussian downsample
    '''
    if scale > 1:
        print('Error. Scale factor is larger than 1.')
        return
    gaussian_std = 2
    kernel_size = 16
    gaussian_down = utils.gaussian_downsample(kernel_size,gaussian_std,
                                            int(1/scale),pad=True)

    #gaussian downsample with zero padding
    out = gaussian_down(input_img).to(DEVICE)
    return out


def reconstruct_MAP(img, lam, patch_size, n_patches_out, regularizer, 
                    n_iter_max, operator, learning_rate=0.03):
    '''
    Reconstruction for given observation and given regularizer
    '''
    # fixed parameters
    lr_img = img.to(DEVICE)
    center = False 
    init = F.interpolate(img, scale_factor=4, mode='bicubic')

    # create patch extractors
    input_im2pat = utils.patch_extractor(patch_size, pad=False, center=center)

    # intialize optimizer for image
    fake_img = init.clone().detach().requires_grad_(True)
    optim_img = torch.optim.Adam([fake_img], lr=learning_rate)

    # Main loop
    for it in tqdm(range(n_iter_max)):
        optim_img.zero_grad()
        #extract patches
        fake_data = input_im2pat(fake_img,n_patches_out)
        #reg
        reg = regularizer(fake_data)
        #data fidelity
        data_fid = torch.sum((operator(fake_img) - lr_img)**2)
        #loss
        loss =  data_fid + lam*reg
        loss.backward()
        optim_img.step()
    return fake_img


if __name__ == '__main__':
    torch.manual_seed(42)
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
    patch_size = args.patch
    
    # patchNR
    if args.reg == 'nf':
        try:
            weights = torch.load(f'model_weights/patchNR_material_p{args.patch}.pth')
        except:
            raise Exception('Model weights cannot be found. Please ensure \
                    that weights are named correctly or train the model \
                    by calling the skript of the regularizer and set the flag \
                    "train" to True.')
        reg_model = NFmodel.NF_regularizer(patch_size = weights['patch'], 
                            num_layers = weights['layer'], 
                            subnet_nodes = weights['nodes'])
        reg_model.NF.load_state_dict(weights['state_dict'])
        lam = 0.15
        n_pat = 130000
        iteration = 300
        lrate = 0.03
        
    # EPLL
    elif args.reg == 'epll':
        try:
            path = f'model_weights/gmm_material_p{args.patch}.joblib'
        except:
            raise Exception('Model weights cannot be found. Please ensure \
                    that weights are named correctly or train the model \
                    by calling the skript of the regularizer and set the flag \
                    "train" to True.')
        reg_model = EPLLmodel.EPLL_regularizer(path)
        lam = 0.35
        n_pat = 30000
        iteration = 500
        lrate = 0.03

    # ALR
    elif args.reg == 'alr':
        try:
            weights = torch.load(f'model_weights/ALR_material_p{args.patch}.pth')
        except:
            raise Exception('Model weights cannot be found. Please ensure \
                    that weights are named correctly or train the model \
                    by calling the skript of the regularizer and set the flag \
                    "train" to True.')
        reg_model = ALRmodel.ALR(patch_size=weights['patch'])
        reg_model.localAReg_CNN.load_state_dict(weights['state_dict'])
        lam = 70
        n_pat = 100000
        iteration = 100
        lrate = 0.03

    # WPP
    elif args.reg == 'w':
        reference_img = utils.imread('data/img_learn_material.png')
        extractor = utils.patch_extractor(patch_size)
        data = extractor(reference_img,10000)
        reg_model = Wmodel.WassersteinPrior(data,iter_psi_start=10,
                            restart_psi=True)
        lam = 166
        n_pat = 1000000
        iteration = 200
        lrate = 0.03

    # Sinkhorn
    elif args.reg == 's':
        reference_img = utils.imread('data/img_learn_material.png')
        extractor = utils.patch_extractor(patch_size)
        data = extractor(reference_img,10000)
        reg_model = Smodel.SinkhornPrior(data,blur=0.01,scaling=0.5)
        lam = 206
        n_pat = 60000
        lrate = 0.03
        iteration = 100
    
    # Semi-unbalanced Sinkhorn
    elif args.reg == 'sus':
        reference_img = utils.imread('data/img_learn_material.png')
        extractor = utils.patch_extractor(patch_size)
        data = extractor(reference_img,10000)
        reg_model = SuSmodel.SuSPP(data,blur=.01,reach=.005,
                                    scaling=.5)
        lam = 236
        n_pat = 50000
        lrate = 0.03
        iteration = 200

    # load test image and create lr observation
    hr = utils.imread('data/img_test_material.png')
    operator = Downsample
    lr = operator(hr)
    lr = lr + 0.01*torch.randn_like(lr)
    
    # MAP reconstruction
    rec = reconstruct_MAP(lr,lam = lam, patch_size = patch_size, 
                        n_patches_out = n_pat, regularizer = reg_model, 
                        n_iter_max = iteration, learning_rate=lrate,
                        operator = operator)
    utils.save_img(rec,f'results/material_sr/result_{args.reg}')

