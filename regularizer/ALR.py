# This code belongs to the paper
#
# M. Piening, F. Altekrüger, J. Hertrich, P. Hagemann, A. Walther and G. Steidl (2023).
# Learning from small data sets: Patch-based regularizers in inverse problems for image reconstruction
# ArXiv preprint:2312.16611
#
# Please cite the paper, if you use the code.
#
# The script implements the ALR used in the paper.

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import torchvision
import utils

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Discriminator(nn.Module):
    '''
    A Discriminator model. Taken from
    'Improved Training of Wasserstein GANs'
    '''
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 1, 3, 2, 0, bias=False)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''
        Defines the computation performed at every call.
        input (tensor): input tensor into the calculation.
        returns a four-dimensional vector (NCHW).
        '''
        out = self.main(input)
        out = torch.flatten(out)
        return out

def calculate_gradient_penalty(model, real_images, fake_images, device):
    '''
    Calculates the gradient penalty loss for WGAN GP
    '''
    # Random weight term for interpolation between real and fake data
    alpha = torch.rand((real_images.size(0), 1, 1, 1), device=device)
    # Get random interpolation between real and fake data
    interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)

    model_interpolates = model(interpolates)
    grad_outputs = torch.ones(model_interpolates.size(), device=device, requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=model_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

class ALR(nn.Module):
    def __init__(self, patch_size = 6, weight_file=None):
        super(ALR, self).__init__()
        self.localAReg_CNN = Discriminator().to(DEVICE)
        self.patch_size = patch_size
        if weight_file is not None:
            self.load_weights(weight_file)

    def forward(self, fake_data):
        '''
        computes the ALR regularizer 
        '''
        fake_data = fake_data.reshape(-1, 1, self.patch_size, self.patch_size)
        reg = -self.localAReg_CNN(fake_data).mean()
        return reg

    def train(self, data, fake_data=None, batch_size=64, save_name="D_lar",
                epochs=10, lr=1e-4, mu=10, patch_size=6, **kwargs):
        data = data.reshape(-1, 1, patch_size, patch_size)
        patch_dataset = torch.utils.data.TensorDataset(data)
        dataloader = torch.utils.data.DataLoader(patch_dataset, 
                                        batch_size=batch_size)
        if fake_data is not None:
            fake_data = fake_data.reshape(-1, 1, patch_size, patch_size)
            fake_dataset = torch.utils.data.TensorDataset(fake_data)
            fake_dataloader = torch.utils.data.DataLoader(fake_dataset, 
                                                batch_size=batch_size)
        discriminator = Discriminator().to(DEVICE)
        optimizer_d = torch.optim.Adam(discriminator.parameters(), 
                                            lr=lr, betas=(0.5, 0.999))

        blur_operator = torchvision.transforms.GaussianBlur(5, sigma=(0.001, 0.3))
        for epoch in range(epochs):
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
            if fake_data is not None:
                fake_iterator = iter(fake_dataloader)
            for i, data in progress_bar:
                real_images = data[0].to(DEVICE)
                real_images = real_images

                ##############################################
                #  Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ##############################################
                discriminator.zero_grad()

                # Train with real
                real_output = discriminator(real_images)
                errD_real = torch.mean(real_output)
                D_x = real_output.mean().item()

                if fake_data is not None:
                    fake_images = next(fake_iterator)[0].to(DEVICE)
                else:
                    fake_images = blur_operator(real_images)
                noise_lvl = 0.2 * torch.rand((1))[0]
                fake_images += noise_lvl * torch.randn_like(real_images)

                # Train with fake
                fake_output = discriminator(fake_images.detach())
                errD_fake = torch.mean(fake_output)
                D_G_z1 = fake_output.mean().item()

                # Calculate W-div gradient penalty
                gradient_penalty = calculate_gradient_penalty(discriminator,
                                    real_images.data, fake_images.data,
                                    DEVICE)

                # Add the gradients from the all-real and all-fake batches
                errD = -errD_real + errD_fake + gradient_penalty * mu
                errD.backward()
                # Update D
                optimizer_d.step()
                progress_bar.set_description(
                    f"[{epoch + 1}/{epochs}][{i + 1}/{len(dataloader)}] "
                    f"Loss_D: {errD.item():.3f} "
                    f"D(x_true): {D_x:.3f} D(x_corrupt): {D_G_z1:.3f}")

        torch.save({'state_dict': discriminator.state_dict(), 'patch': self.patch_size},
                   f'{save_name}')
        return

    def load_weights(self, weight_file):
        try:
            weights = torch.load(weight_file)['state_dict']
            self.localAReg_CNN.load_state_dict(weights)
        except:
            raise Exception('Model weights cannot be found. Please ensure \
                            that weights are named correctly.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type = bool, default = False,
                        help='Set flag to True if you want to retrain the ALR')
    parser.add_argument('--epochs', type = int, default = 10,
                        help='Number of optimizer steps for training')
    parser.add_argument('--batch', type = int, default = 64,
                        help='Batch size for training')
    parser.add_argument('--patch', type = int, default = 6,
                        help='Patch size for training')
    parser.add_argument('--use_bicubic_fake', type = bool, default = False,
                        help='Use patches from corrupted image instead \
                        of corrupted patches')
    parser.add_argument('--dataset', type = str, default = 'material', 
                        choices=['material','ct'],
                        help='Choose dataset for training. You have the \
                        choice between material data (material) \
                        and CT data (ct).')
    args = parser.parse_args()

    if args.train:
        im2pat = utils.patch_extractor(patch_size=args.patch)
        localAR = LocalA_regularizer(patch_size=args.patch)
        if args.dataset == 'material':
            example_img = utils.imread('data/img_learn_material.png')
            learn = im2pat(example_img)
            if args.use_bicubic_fake:
                fake = F.interpolate(F.interpolate(example_img, 
                                    scale_factor=1/4, mode='nearest'),
                                    scale_factor=4, mode='bicubic')
                fake = im2pat(fake)
            else:
                fake = None
            localAR.train(data=learn,fake_data=fake,epochs=args.epochs,
                        batch_size=args.batch, patch_size=args.patch,
                        save_name=f'model_weights/ALR_material_p{args.patch}.pth')
        elif args.dataset == 'ct':
            from dival import get_standard_dataset
            from odl.contrib.torch import OperatorModule
            import odl
            dataset = get_standard_dataset('lodopab', impl='astra_cuda')
            train = dataset.create_torch_dataset(part='train',
                                    reshape=((1,1,) + dataset.space[0].shape,
                                    (1,1,) + dataset.space[1].shape))
            example_img = torch.cat([train[3][1],train[5][1],train[8][1],
                                train[11][1],train[37][1],train[75][1]]).to(DEVICE)
            learn = torch.tensor([],device=DEVICE)
            ray_trafo = dataset.ray_trafo
            #load operator and FBP
            operator = OperatorModule(ray_trafo).to(DEVICE)
            fbp = odl.tomo.analytic.filtered_back_projection.fbp_op(ray_trafo,
	                    filter_type = 'Hann', frequency_scaling = 0.641025641025641)
            fbp = OperatorModule(fbp)
            if args.use_bicubic_fake:
                sino = torch.cat([train[3][0],train[5][0],train[8][0],
                                train[11][0],train[37][0],train[75][0]]).to(DEVICE)
                fake = fbp(sino)
            else:
                fake = None
            for i in range(len(example_img)):
                learn = torch.cat([learn,im2pat(example_img[i].unsqueeze(0))],dim=0)
            localAR.train(data=learn,save_name=f'model_weights/ALR_ct_p{args.patch}.pth',
                                epochs=args.epochs, batch_size=args.batch, fake = fake)
