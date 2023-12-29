# This code belongs to the paper
#
# M. Piening, F. Altekrüger, J. Hertrich, P. Hagemann, A. Walther and G. Steidl (2023).
# Learning from small data sets: Patch-based regularizers in inverse problems for image reconstruction
# ArXiv preprint:2312.16611
#
# Please cite the paper, if you use the code.
#
# The script implements the Sinkhorn Prior used in the paper.

from torch import nn
import geomloss

class SinkhornLoss(nn.Module):
    '''
    calc either Sinkhorn or MMD with gaussian, lapcian or energy kernel
    '''
    def __init__(self, input_feature_sample, blur=0.01, loss="sinkhorn",
                p=2, reach=None, scaling=.5):
        super(SinkhornLoss, self).__init__()
        self.blur = blur
        self.p = p
        self.input_feature_sample = input_feature_sample
        self.loss_object = geomloss.SamplesLoss(loss=loss, p=p, blur=self.blur, 
                            scaling=scaling, reach=reach, debias=False)

    def forward(self, synth_feature_sample):
        return self.loss_object(synth_feature_sample, self.input_feature_sample)

class SinkhornPrior(nn.Module):
    def __init__(self, reference_img=None,reach=None,blur=0.01,scaling=.5):
        super(SinkhornPrior, self).__init__()
        self.reference_img = reference_img
        self.reach = reach
        self.blur = blur
        self.scaling = scaling
        self.regularizer_function = SinkhornLoss(self.reference_img, 
                                reach=self.reach, blur=self.blur,
                                scaling=self.scaling)

    def forward(self, fake_data):
        '''
        computes the Sinkhorn regularizer 
        '''
        return self.regularizer_function(fake_data)
    
    def train(self,data,**kwargs):
        self.reference_img = data
        self.regularizer_function = SinkhornLoss(self.reference_img,
                                        reach=self.reach,blur=self.blur,
                                        scaling=self.scaling)
        return
        
    def load_weights(self,file_name):
        return
