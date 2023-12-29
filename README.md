# Learning from small data sets: Patch-based regularizers in inverse problems for image reconstruction

This code belongs to the paper [8] available at https://arxiv.org/abs/2312.16611. Please cite the paper, if you use this code.

The repository contains an implementation of several patch-based regularizers in an unified framework. Numerical examples like Computed Tomography (CT) in Section 7.2, Super-resolution in Section 7.3 and Image Inpainting in Section 7.4 and Posterior sampling in Section 7.5 are implemented. 

The folder `data` contains different datasets used for the numerical examples. In the folder `model_weights`, pretrained versions of EPLL, patchNR and ALR are included. The folder `regularizer` is the core of the repository and contains the implementation of the different regularizers.

For questions and bug reports, please contact Moritz Piening (piening@math.tu-berlin.de) or Fabian Altekrüger (fabian.altekrueger@hu-berlin.de).

## CONTENTS

1. ENVIRONMENT SETTINGS
2. USAGE AND EXAMPLES
3. HYPERPARAMETERS
4. REFERENCES

## 1. ENVIRONMENT SETTINGS

The code requires several Python packages. You can create a conda environment using `environment.yaml` by the command
```python
conda env create --file=environment.yaml
```
Additionally, you have to install the latest version of ODL via pip:
```python
pip install https://github.com/odlgroup/odl/archive/master.zip --upgrade
```

## 2. USAGE AND EXAMPLES

The repository contains PyTorch implementations of the 
- expected patch log-likelihood (EPLL) [10], 
- patch normalizing flow regularizer (patchNR) [1], 
- adversarial local regularizer (ALR) [9],
- Wasserstein patch prior (WPP) [3],
- Sinkhorn patch prior ($`WPP_\varepsilon`$) [7] and 
- semi-unbalanced Sinkhorn patch prior ($`WPP_{\varepsilon,\rho}`$) [7].

The regularizer EPLL, patchNR and ALR need to be trained on given reference data for the experiments `super-resolution on material data` and `ct`. Pretrained versions are already included in the folder `model_weights`. As an example, if you want to retrain the regularizer `patchNR` for the dataset material and a patch size of 6, you need to call the file `NFReg.py` and set the flag `train=True`, the flag `dataset=material` and the flag `patch=6`.
```python
python -m regularizer.NFReg --train=True --patch=6 --dataset='material'
```
In general, the experiments can be reproduced by calling the script `run_xxx.py`. Here you can choose between the different regularizers by setting the flag `reg` to the corresponding regularizer. You have the choice between EPLL (`--reg='epll'`), patchNR (`--reg='nf'`), ALR (`--reg='alr'`), WPP (`--reg='w'`), $WPP_\varepsilon$ (`--reg='s'`) and $WPP_{\varepsilon, \rho}$ (`--reg='sus'`). 

### CT IMAGING

The script `run_CT.py` is the implementation of the CT example in [8, Section 7.2]. The used data is from the LoDoPaB dataset [5], which is available at https://zenodo.org/record/3384092##.Ylglz3VBwgM. Here we assume that we are given 6 training images.
You can choose between the different regularizers and between the low-dose and the limited-angle settings.
As an example, if you want to reconstruct with the regularizer `patchNR` in the low-dose setting, you need to set the flag `reg=nf` and the flag `angle=full`.
```python
python run_ct.py --reg='nf' --angle='full'
```

### SUPER-RESOLUTION

The script `run_sr_xxx.py` is the implementation of the superresolution example in [8, Section 7.3]. On the one hand, we have an example of super-resolution of material data: The used images of material microstructures have been acquired in the frame of the EU Horizon 2020 Marie Sklodowska-Curie Actions Innovative Training Network MUMMERING (MUltiscale, Multimodal and Multidimensional imaging for EngineeRING, Grant Number 765604) at the beamline TOMCAT of the SLS by A. Saadaldin, D. Bernard, and F. Marone Welford. We assume that we are given a high-resolution reference image. You can choose between the different regularizers.
As an example, if you want to reconstruct with the regularizer `patchNR`, you need to set the flag `reg=nf`.
```python
python run_sr_material.py --reg='nf'
```

On the other hand, we have an example of zero-shot super-resolution of natural images from the BSD68 dataset [6]. More precisely, we assume that no reference data is available and we need to extract the prior information from the given low-resolution observation. Thus, at test time we extract the patches of the observation and train EPLL, ALR, and patchNR with them. Moreover, for WPP, $WPP_\varepsilon$ and $WPP_{\varepsilon, \rho}$ the low-resolution patches are used for the reference patch distribution. 

Before you can run the script, you need to download the BSD68 dataset. This can be done, e.g., by
```python
wget -c https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz && tar -xf BSDS300-images.tgz && rm BSDS300-images.tgz && mv BSDS300/images/test/253027.jpg data/bsd_img.jpg
```
where you also moved the first BSD68 image illustrated in the paper to the folder `data`.
As an example, if you want to reconstruct with the regularizer `patchNR`, you need to set the flag `reg=nf`.
```python
python run_sr_zeroshot.py --reg='nf'
```

### INPAINTING

The script `run_inpainting.py` is the implementation of the inpainting example in [8, Section 7.4]. We consider the image `butterfly` from the Set5 [2]. First, you need to download the image in copy it into the folder `data`. This can be done, e.g., by
```python
wget -c https://huggingface.co/datasets/eugenesiow/Set5/resolve/main/data/Set5_HR.tar.gz && tar -xf Set5_HR.tar.gz && rm Set5_HR.tar.gz && mv Set5_HR/butterfly.png data 
```
We assume that no reference data is available, such that we use a small neighborhood of the missing part for extracting the reference patches.
You can choose between the different regularizers.
As an example, if you want to reconstruct with the regularizer `patchNR`, you need to set the flag `reg=nf`.
```python
python run_inpainting.py --reg='nf'
```

### LANGEVIN

We provide two examples of posterior sampling via the unadjusted Langevin algorithm.
In the script `run_langevin_ct.py` we implement the Langevin CT example from [8, Section 7.5]. The setting is similar to the CT MAP estimation setting.
As an example, if you want to reconstruct with the regularizer `patchNR` in the low-dose setting, you need to set the flag `reg=nf` and the flag `angle=full`.
```python
python run_ct.py --reg='nf' --angle='full'
```

Moreover, the script `run_langevin_inpainting.py` is the implementation of the inpainting example from [8, Section 7.5]. We consider the same setting as for the MAP estimation for inpainting.
As an example, if you want to reconstruct with the regularizer `patchNR`, you need to set the flag `reg=nf`.
```python
python run_langevin_inpainting.py --reg='nf'
```

## 3. HYPERPARAMETERS

Here we provide the used hyperparameters for our experiments. They were determined by a grid search. You can play with the parameters and see how they affect the result.

### EPLL

The EM algorithm is used to determine the means and covariance matrices of the corresponding modes and we stop when a predefined tolerance is achieved. We use the Adam optimizer [4] reconstruction. In the reconstruction, we do not use all patches, but a stochastic gradient descent with batch size $N_p$ to speed up the numerical computations.

|            |  low-dose CT | limited-angle CT | SR material | SR zeroshot | Inpainting | Langevin CT | Langevin Inpainting |
| ---        | ---          |              --- |         --- |         --- |        --- | ---         | ---                 |
| `training` |              |                  |             |             |            |             |                     |
| pretrained | True         | True             | True        | False       | False      | True        | False               |
| components | 200          | 200              | 200         | 50          | 50         |  200        | 50                  |
| maximal iterations | 1000 | 1000             | 1000        | 500         | 500        |  1000       | 500                 |
| tolerance  | 1e-4         |  1e-4            |  1e-4       | 1e-4        | 5e-5       | 1e-4        | 5e-5                |
| `reconstruction` |        |                  |             |             |            |             |                     |
| $\lambda$  | 1000         | 1000             |  0.35       | 0.35        | 1          |  7000       |  500                |
| iterations | 300          | 3000             |  300        | 200         | 10000      | 20000       | 25000               |
| learning rate | 0.01      | 0.01             |  0.03       | 0.03        | 0.01       |  5e-7       |  0.002              |
| $N_p$      | 30000        | 30000            |  30000      | 30000       | all        |  30000      |  all                |


### patchNR

We use 5 Glow Coupling blocks and permutations in an alternating manner. The three-layer subnetworks are fully connected with ReLU activation functions and 512 nodes. We use the Adam optimizer for training and reconstruction. The learning rate for training the patchNR is set to 1e-4. In the reconstruction, we do not use all patches, but a stochastic gradient descent with batch size $N_p$ to speed up the numerical computations.

|            |  low-dose CT | limited-angle CT | SR material | SR zeroshot | Inpainting | Langevin CT | Langevin Inpainting |
| ---        | ---          |              --- |         --- |         --- |        --- | ---         | ---                 |
| `training` |              |                  |             |             |            |             |                     |
| pretrained | True         | True             | True        | False       | False      | True        | False               |
| optimizer steps | 750000  | 750000           | 750000      | 5000        | 10000      | 750000      | 40000               |
| batch size | 32           |  32              |  32         | 512         | 512        | 32          |  512                |
| `reconstruction` |        |                  |             |             |            |             |                     |
| $\lambda$  | 700          | 700              |  0.15       | 0.25        | 100        | 6000        |  500                |
| iterations | 300          | 3000             |  300        | 200         | 15000      | 50000       | 25000               |
| learning rate | 0.005     | 0.005            |  0.03       | 0.03        | 0.03       | 2e-7        |  0.002              |
| $N_p$      | 40000        | 40000            |  130000     | 70000       | all        | 40000       |   all               |


### ALR

The discriminator is a CNN with 3 convolutional layers and a LeakyReLU as activation function. We use the Adam optimizer for training and reconstruction. The learning rate for training the ALR is set to 1e-4. In the reconstruction, we do not use all patches, but a stochastic gradient descent with batch size $N_p$ to speed up the numerical computations.

|            |  low-dose CT | limited-angle CT | SR material | SR zeroshot | Inpainting | Langevin CT | Langevin Inpainting |
| ---        | ---          |              --- |         --- |         --- |        --- | ---         | ---                 |
| `training` |              |                  |             |             |            |             |                     |
| pretrained | True         | True             | True        | False       | False      | True        |  False              |
| epochs     | 10           | 10               | 10          | 10          | 50         | 10          | 50                  |
| batch size | 64           |  64              |  64         | 64          | 128        |  64         |  128                |
| `reconstruction` |        |                  |             |             |            |             |                     |
| $\lambda$  | 5.5e+5       | 5.5e+5           |  70         | 70          | 1          | 4e+6        |  500                |
| iterations | 300          | 3000             |  100        | 100         | 400        | 30000       |  250000             |
| learning rate | 0.005     | 0.005            |  0.03       | 0.03        | 0.03       | 5e-7        |  0.002              |
| $N_p$      | 40000        | 40000            |  100000     | 100000      | all        | 40000       |  all                |


### WPP

We use an Adam optimizer for finding the dual maximizer $\psi$ and for reconstruction. In the reconstruction, we use all patches.

|            |  low-dose CT | limited-angle CT | SR material | SR zeroshot | Inpainting | Langevin CT | Langevin Inpainting |
| ---        | ---          |              --- |         --- |         --- |        --- | ---         | ---                 |
| `training` |              |                  |             |             |            |             |                     |
| pretrained | False        | False            | False       | False       | False      | False       |  False              |
| `reconstruction` |        |                  |             |             |            |             |                     |
| $\lambda$  | 4e+6         | 4e+6             |  166        | 106         | 100        | 5e+7        | 500                 |
| iterations | 300          | 3000             |  200        | 300         | 800        | 20000       |  25000              |
| learning rate | 0.001     | 0.001            |  0.03       | 0.03        | 0.03       | 5e-7        |   0.002             |

### $WPP_\varepsilon$

In the reconstruction, we do not use all patches, but a stochastic gradient descent with batch size $N_p$ to speed up the numerical computations. We use an entropic regularization of $\varepsilon=0.01**2$.

|            |  low-dose CT | limited-angle CT | SR material | SR zeroshot | Inpainting | Langevin CT | Langevin Inpainting |
| ---        | ---          |              --- |         --- |         --- |        --- | ---         | ---                 |
| `training` |              |                  |             |             |            |             |                     |
| pretrained | False        | False            | False       | False       | False      | False       |  False              |
| `reconstruction` |        |                  |             |             |            |             |                     |
| $\lambda$  | 1e+7         | 1e+7             |  206        | 106         | 1          |  7e+7       |  500                |
| iterations | 100          | 1000             |  100        | 100         | 1000       |  10000      |   25000             |
| learning rate | 0.001     | 0.001            |  0.03       | 0.01        | 0.03       |  5e-7       | 0.002               |
| $N_p$      | 40000        | 40000            |  60000      | 50000       | 5000       |  40000      |  5000               |

### $WPP_{\varepsilon,\rho}$

In the reconstruction, we do not use all patches, but a stochastic gradient descent with batch size $N_p$ to speed up the numerical computations. 

|            |  low-dose CT | limited-angle CT | SR material | SR zeroshot | Inpainting | Langevin CT | Langevin Inpainting |
| ---        | ---          |              --- |         --- |         --- |        --- | ---         | ---                 |
| `training` |              |                  |             |             |            |             |                     |
| pretrained | False        | False            | False       | False       | False      | False       |   False             |
| `reconstruction` |        |                  |             |             |            |             |                     |
| $\lambda$  | 1e+7         | 1e+7             |  236        | 106         | 1          |  1e+8       |   500               |
| $\varepsilon | 0.01**2    | 0.01**2          |  0.01**2    | 0.05**2     | 0.001**2   | 0.01**2     |   0.001**2          |
| $\rho$     | 0.005**2     | 0.005**2         |  0.005**2   | 306         | 10**2      | 0.005**2    |    20**2            |
| iterations | 200          | 500              |  200        | 100         | 400        | 10000       |    25000            |
| learning rate | 0.002     | 0.002            |  0.03       | 0.003       | 0.03       | 5e-7        |   0.002             |
| $N_p$      | 40000        | 40000            |  50000      | 30000       | 5000       | 40000       |   5000              |

## 4. REFERENCES

[1] F. Altekrüger, A. Denker, P. Hagemann, J. Hertrich, P. Maass and G. Steidl.  
PatchNR: Learning from Very Few Images by Patch Normalizing Flow Regularization.   
Inverse Problems, vol. 39, no. 6, 2023.

[2] M. Bevilacqua, A. Roumy, C. M. Guillemot and M.-L. Alberi-Morel.  
Low-complexity single-image super-resolution based on nonnegative neighbor embedding.  
British Machine Vision Conference, 2012.

[3] J. Hertrich, A. Houdard, and C. Redenbach.  
Wasserstein patch prior for image superresolution.  
IEEE Transactions on Computational Imaging, vol. 8, 2022.

[4] D. Kingma and J. Ba.  
Adam: A method for stochastic optimization.  
International Conference on Learning Representations, 2015.

[5] J. Leuschner, M. Schmidt, D. O. Baguer and P. Maass.  
LoDoPaB-CT, a benchmark dataset for low-dose computed tomography reconstruction.  
Scientific Data, 9(109), 2021.

[6] D. Martin, C. Fowlkes, D. Tal, and J. Malik.  
A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics.  
Proceedings Eighth IEEE International Conference on Computer Vision, volume 2, 2001.

[7] S. Mignon, B. Galarne, M. Hidane, C. Louchet and J. Mille.  
Semi-unbalanced regularized optimal transport for image restoration.  
European Signal Processing Conference, 2023.

[8] M. Piening, F. Altekrüger, J. Hertrich, P. Hagemann, A. Walther and G. Steidl.  
Learning from small data sets: Patch-based regularizers in inverse problems for image reconstruction.  
ArXiv preprint:2312.16611, 2023.

[9] J. Prost, A. Houdard, A. Almansa and N. Papadakis.  
Learning local regularization for variational image restoration.  
International Conference on Scale Space and Variational Methods in Computer Vision, 2021. 

[10] D. Zoran and Y. Weiss.  
From learning models of natural image patches to whole image restoration.  
International Conference on Computer Vision, 2011.


## CITATION

```python
@article{PAHHWS2023,
    author    = {Piening, Moritz and Altekrüger, Fabian and Hertrich, Johannes and Hagemann, Paul and Walther, Andrea and Steidl, Gabi},
    title     = {Learning from small data sets: {P}atch-based regularizers in inverse problems for image reconstruction},
    journal   = {arXiv preprint arXiv:2312.16611},
    year      = {2023}
}
```
