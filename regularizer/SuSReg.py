# This code belongs to the paper
#
# M. Piening, F. Altekrüger, J. Hertrich, P. Hagemann, A. Walther and G. Steidl (2023).
# Learning from small data sets: Patch-based regularizers in inverse problems for image reconstruction
# ArXiv preprint:2312.16611
#
# Please cite the paper, if you use the code.
#
# The script implements the Semi-Unbalanced Sinkhorn Patch Prior used in the paper.

import torch.nn as nn
from regularizer.samples_loss import SamplesLoss

class SuSLoss(nn.Module):
    def __init__(self, input_feature_sample, blur=0.001, reach=1, 
                scaling =.9, **kwargs):
        super(SuSLoss, self).__init__()
        self.blur = blur
        self.p = 2
        self.input_feature_sample = input_feature_sample
        #semi-unbalanced
        self.loss_object = SamplesLoss(blur=blur, reach=[None, reach], 
                                            scaling=scaling,debias=False)

    def forward(self, synth_feature_sample):
        return self.loss_object(synth_feature_sample, self.input_feature_sample)

class SuSPP(nn.Module):
    def __init__(self, reference_img=None, reach=1, blur=0.001, 
                scaling=.9):
        super(SuSPP, self).__init__()
        self.reference_img = reference_img
        self.reach = reach
        self.blur = blur
        self.scaling = scaling
        self.regularizer_function = SuSLoss(self.reference_img, 
                                reach=reach, blur=self.blur, 
                                scaling=self.scaling)

    def forward(self, fake_data):
        '''
        computes the SuSPP regularizer 
        '''
        return self.regularizer_function(fake_data)

    def train(self,data,**kwargs):
        self.reference_img = data
        self.regularizer_function = SuSLoss(self.reference_img,
                                    reach=self.reach, blur=self.blur,
                                    scaling=self.scaling)
        return

    def load_weights(self,file_name):
        return
