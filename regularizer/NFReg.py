# This code belongs to the paper
#
# M. Piening, F. Altekrüger, J. Hertrich, P. Hagemann, A. Walther and G. Steidl (2023).
# Learning from small data sets: Patch-based regularizers in inverse problems for image reconstruction
# ArXiv preprint:2312.16611
#
# Please cite the paper, if you use the code.
#
# The script implements the patchNR used in the paper.

import torch
from torch import nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import argparse
from tqdm import tqdm
import numpy as np
import utils

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_NF(num_layers, sub_net_size, dimension):
    '''
    Creates the patchNR network
    '''
    def subnet_fc(c_in, c_out):
        return nn.Sequential(nn.Linear(c_in, sub_net_size), nn.ReLU(),
                        nn.Linear(sub_net_size, sub_net_size), nn.ReLU(),
                        nn.Linear(sub_net_size,  c_out))
    nodes = [Ff.InputNode(dimension, name='input')]
    for k in range(num_layers):
        nodes.append(Ff.Node(nodes[-1],
                          Fm.GLOWCouplingBlock,
                          {'subnet_constructor':subnet_fc, 'clamp':1.6},
                          name=F'coupling_{k}'))
        nodes.append(Ff.Node(nodes[-1],
                          Fm.PermuteRandom,
                          {'seed':(k+1)},
                          name=F'permute_flow_{k}'))
    nodes.append(Ff.OutputNode(nodes[-1], name='output'))
    model = Ff.ReversibleGraphNet(nodes, verbose=False).to(DEVICE)
    return model

class NF_regularizer(nn.Module):
    def __init__(self, patch_size = 6, num_layers = 5, 
                    subnet_nodes = 512, weight_file = None):
        super(NF_regularizer, self).__init__()
        self.num_layers = num_layers
        self.subnet_nodes = subnet_nodes
        self.patch_size = patch_size
        self.NF = create_NF(self.num_layers, self.subnet_nodes, 
                            dimension=self.patch_size**2)
        #load network weights if given
        if weight_file is not None:
            self.load_weights(weight_file)

    def forward(self, fake_data):
        '''
        computes the patchNR regularizer 
        '''
        pred_inv, log_det_inv = self.NF(fake_data,rev=True)
        reg = torch.mean(torch.sum(pred_inv**2,dim=1)/2) - torch.mean(log_det_inv)
        return reg

    def train(self,data,save_name='patchNR_material',optimizer_steps=750000,
                batch_size=32,**kwargs):
        '''
        starts the training of the patchNR
        input: data in form of training patches
        saves the corresponding weights
        '''
        self.NF = create_NF(self.num_layers, self.subnet_nodes, 
                            dimension=self.patch_size**2)
        optimizer = torch.optim.Adam(self.NF.parameters(), lr = 1e-4)
        for k in tqdm(range(optimizer_steps)):
            idx = torch.tensor(np.random.choice(data.shape[0],batch_size))
            data_tmp = data[idx]
            
            #compute loss
            invs, jac_inv = self.NF(data_tmp, rev = True)
            loss = torch.mean(0.5 * torch.sum(invs**2, dim=1) - jac_inv)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save({'state_dict': self.NF.state_dict(), 'layer': self.num_layers, 
                    'nodes': self.subnet_nodes, 'patch': self.patch_size},
                    f'{save_name}')

    def load_weights(self,file_name):
        weights = torch.load(file_name)['state_dict']
        self.NF.load_state_dict(weights)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type = bool, default = False,
                        help='Set flag to True if you want to retrain the patchNR')
    parser.add_argument('--steps', type = int, default = 750000,
                        help='Number of optimizer steps for training')
    parser.add_argument('--batch', type = int, default = 32,
                        help='Batch size for training')
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
        patchNR = NF_regularizer(patch_size = args.patch)
        if args.dataset == 'material':
            example_img = utils.imread('data/img_learn_material.png')
            learn = im2pat(example_img)
            patchNR.train(data=learn,save_name=f'model_weights/patchNR_material_p{args.patch}.pth',
                            optimizer_steps=args.steps, batch_size=args.batch)
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
            patchNR.train(data=learn,save_name=f'../model_weights/patchNR_ct_p{args.patch}.pth',
                            optimizer_steps=args.steps, batch_size=args.batch)

