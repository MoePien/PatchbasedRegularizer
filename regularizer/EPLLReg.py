# This code belongs to the paper
#
# M. Piening, F. Altekrüger, J. Hertrich, P. Hagemann, A. Walther and G. Steidl (2023).
# Learning from small data sets: Patch-based regularizers in inverse problems for image reconstruction
# ArXiv preprint:2312.16611
#
# Please cite the paper, if you use the code.
#
# The script implements the EPLL used in the paper.

import torch
from torch import nn
import torch.distributions as D
import numpy as np
from sklearn import mixture
import joblib
import argparse
import utils

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EPLL_regularizer(nn.Module):
    def __init__(self, weight_file=None):
        super(EPLL_regularizer, self).__init__()
        if weight_file is not None:
            self.load_weights(weight_file)

    def forward(self, fake_data):
        '''
        computes the EPLL regularizer 
        '''
        ll = self.gmm.log_prob(fake_data)
        reg = -torch.mean(ll)
        return reg
        
    def train(self,data,save_name='gmm_material',n_components=200,
                max_it=1000,tol=1e-4, **kwargs):
        '''
        starts the training of the GMM
        input: data in form of training patches
        saves the corresponding weights
        '''
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type='full',
                                      max_iter=max_it, tol=tol,
                                      verbose=2, verbose_interval=10)
        gmm.fit(data.detach().cpu())
        joblib.dump(gmm, f'{save_name}')

    def load_weights(self, weight_file):
        model = joblib.load(weight_file)
        mix = D.Categorical(torch.from_numpy(np.array(model.weights_)).float().to(DEVICE))
        comp = D.MultivariateNormal(torch.from_numpy(np.array(model.means_)).float().to(DEVICE),
                                    torch.from_numpy(np.array(model.covariances_)).float().to(DEVICE))
        self.gmm = D.MixtureSameFamily(mix, comp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type = bool, default = False,
                        help='Set flag to True if you want to retrain the model')
    parser.add_argument('--patch', type = int, default = 6,
                        help='Patch size for training')
    parser.add_argument('--dataset', type = str, default = 'material', 
                        choices=['material','ct'],
                        help='Choose dataset for training. You have the \
                              choice between material data (material) \
                              and CT data (ct).')
    args = parser.parse_args()
    
    if args.train:
        im2pat = utils.patch_extractor(patch_size=args.patch)
        epll = EPLL_regularizer()
        if args.dataset == 'material':
            example_img = utils.imread('data/img_learn_material.png')
            learn = im2pat(example_img)
            epll.train(data=learn,save_name=f'model_weights/gmm_material_p{args.patch}.joblib')
        elif args.dataset == 'ct':
            from dival import get_standard_dataset
            dataset = get_standard_dataset('lodopab', impl='astra_cuda')
            train = dataset.create_torch_dataset(part='train',
                                reshape=((1,1,) + dataset.space[0].shape,
                                (1,1,) + dataset.space[1].shape))
            example_img = torch.cat([train[3][1],train[5][1],train[8][1],
                            train[11][1],train[37][1],train[75][1]]).to(DEVICE)
            learn = torch.tensor([],device=DEVICE)
            for i in range(len(example_img)):
                learn = torch.cat([learn,im2pat(example_img[i].unsqueeze(0))],dim=0)
            epll.train(data=learn,save_name=f'model_weights/gmm_ct_p{args.patch}.joblib')
