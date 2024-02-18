from .base import BaseVAE
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import math
from typing import List, Callable, Union, Any, TypeVar, Tuple
import cv2   
from .utils import tensor2float, get_powers, NormalizeNumpy, image_checkerer, get_output_resolution


Tensor = TypeVar('torch.tensor')

class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet18Enc(nn.Module):

    def __init__(self, 
                 num_Blocks=[2,2,2,2], 
                 z_dim=10, 
                 nc=1, 
                 **kwargs):
        super().__init__()
        self.in_planes = 64
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
        self.linear = nn.Linear(512, 2 * z_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        return mu, logvar

class ResNet18Dec(nn.Module):

    def __init__(self, 
                 num_Blocks=[2,2,2,2], 
                 z_dim=10, 
                 nc=1,
                 **kwargs):
        super().__init__()
        self.in_planes = 512
        self.nc = nc
        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=4)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), self.nc, 64, 64)
        return x


class RESNET18_BETATC_VAE(BaseVAE):
    num_iter = 0 # Global static variable to keep track of iterations
    def __init__(self, 
                 latent_dim: int,
                 kld_weight:float = 0.001,
                 anneal_steps: int = 200,
                 alpha_mi_loss: float = 1.,
                 beta_tc_loss: float =  6.,
                 device:str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 debug:int = 0,
                 **kwargs) -> None:
        super().__init__()
        
        self.device = device
        self.debug = debug
        self.kld_weight = kld_weight
        Cin = kwargs['in_channels']

        self.anneal_steps = anneal_steps
        self.alpha_mi_loss = alpha_mi_loss
        self.beta_tc_loss = beta_tc_loss  
        
        self.encoder = ResNet18Enc(z_dim=latent_dim, nc=Cin, **kwargs)
        self.decoder = ResNet18Dec(z_dim=latent_dim, nc=Cin, **kwargs)

    def forward(self, sample: tuple):
        x, _ = sample
        # mean, logvar = self.encoder(x.repeat(1,3,1,1))
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decoder(z)
        return [x_hat, x, mean, logvar, z]
    
    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2) # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean

    def log_density_gaussian(self, x: Tensor, mu: Tensor, logvar: Tensor):
        """
        Computes the log pdf of the Gaussian with parameters mu and logvar at x
        :param x: (Tensor) Point at whichGaussian PDF is to be evaluated
        :param mu: (Tensor) Mean of the Gaussian distribution
        :param logvar: (Tensor) Log variance of the Gaussian distribution
        :return:
        """
        norm = - 0.5 * (math.log(2 * math.pi) + logvar)
        log_density = norm - 0.5 * ((x - mu) ** 2 * torch.exp(-logvar))
        return log_density



    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """            
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        z = args[4]
        
        _, labels = args[5] # [B]
        current_epoch = args[6]
        dataset_size = args[7]
        
        weight = 1
        _, C, H, W = input.shape 
        B, Ldim = z.shape
        
        # Ref: https://arxiv.org/pdf/1802.04942.pdf
        # https://github.com/rtqichen/beta-tcvae
        # https://github.com/YannDubs/disentangling-vae
        
        # Reconstruction loss: log_px (averaged over C,W,H and scaled to image dims)        
        # recons_loss =F.mse_loss(recons, input, reduction='sum') / B    # Orig 
        recons_loss = torch.sum(F.mse_loss(recons, input, reduction='none'), dim=(2,3)) # [B,1]
        
        # calculate log q(z|x)
        log_qz_condx = self.log_density_gaussian(z, mu, log_var).sum(dim = 1) # [B] orig
        # log_qz_condx = self.log_density_gaussian(z, mu, log_var) # [B, Ldims]

        # calculate log p(z) , mean and log var is 0
        zeros = torch.zeros_like(z)
        # log_pz = self.log_density_gaussian(z, zeros, zeros).sum(dim = 1) # [B] orig
        log_pz = self.log_density_gaussian(z, zeros, zeros) # [B, Ldims]

        

        # log of latent space Gaussian (for each latent space dims i.e matrix)
        mat_log_qz = self.log_density_gaussian(z.view(B, 1, Ldim),
                                                mu.view(1, B, Ldim),
                                                log_var.view(1, B, Ldim)) # [B,B,Ldims]
        
        # log_importance_weight_matrix with stratification
        M, N = B - 1, dataset_size
        strat_weight = (N - M) / (N * M) # Account for the minibatch samples from the dataset
        importance_weights = torch.Tensor(B, B).fill_(1 / M).to(self.device) # [B, B]
        importance_weights.view(-1)[::M+1] = 1 / N
        importance_weights.view(-1)[1::M+1] = strat_weight
        importance_weights[M-1, 0] = strat_weight    
        log_importance_weights = importance_weights.log() # [B, B]

        mat_log_qz += log_importance_weights.view(B, B, 1) # [B, B, Ldims] orig

        # calculate log q(z)
        log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False) # [B] orig
        
        # log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(1) # [B] orig
        log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False) # [B, Ldims]

        # Elbo
        elbo = recons_loss.squeeze() - log_pz.sum(1) + log_qz_condx

        # MI loss: # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
        # Mutual information between data variable and latent variable based on the empirical data distribution 
        mutual_info_loss  = (log_qz_condx - log_qz) # [B] orig
        # mutual_info_loss  = (log_qz_condx - log_qz).abs() # [B] oli
        # mutual_info_loss  = (log_qz_condx - log_qz.view(B,1) * Ldim) # [B, Ldims] oli
        
        # Total Correlation loss: TC[z] = KL[q(z)||\prod_i z_i]
        # Measure of dependence between variables. Here, it forces the model to find statistically independent factors in the data distribution
        # total_correlation_loss = (log_qz - log_prod_qzi) # [B] orig
        total_correlation_loss = -(log_qz - log_prod_qzi.sum(1)) # [B] 
        # total_correlation_loss = (log_qz - log_prod_qzi).abs() # [B] oli
        # total_correlation_loss = (log_qz.view(B,1) * Ldim - log_prod_qzi ) # [B, Ldims] oli
        
        # Dimension-wise KLD loss (different from usual VAE KLD loss)
        # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
        # Here, mainly prevents individual latent dimensions from deviating too far from their corresponding priors. 
        # acts as a complexity penalty on the aggregate posterior
        kld_loss = (log_prod_qzi - log_pz)  # [B] or [B, Ldims]
        if torch.isnan(kld_loss).any().item():
            print ("NaN deteced")

        # Evaluate annelaing rate to weight-in kld_loss
        if self.training:
            self.num_iter += 1
            anneal_rate = min(0 + 1 * self.num_iter / self.anneal_steps, 1)
        else:
            anneal_rate = 1.

        loss = recons_loss.mean() \
                + self.alpha_mi_loss * mutual_info_loss.mean() \
                + self.beta_tc_loss * total_correlation_loss.mean() \
                + self.kld_weight * anneal_rate * kld_loss.sum(1).mean()
        
        loss_dict = {'loss': loss,
                    'Recons_Loss': recons_loss.detach(),
                    'KLD_Loss': kld_loss.detach(),
                    'MI_Loss': mutual_info_loss.detach(),
                    'Other_Loss': total_correlation_loss.detach(),
                    }

        scalar_outputs = {"full_loss": tensor2float(loss.mean()),
                          "recons_loss": tensor2float(recons_loss.mean()),
                          "KL_loss": tensor2float(self.kld_weight * anneal_rate * kld_loss.sum(1).mean()),   
                          "MI_loss": tensor2float(self.alpha_mi_loss * mutual_info_loss.mean()),                       
                          "TC_loss": tensor2float(self.beta_tc_loss * total_correlation_loss.mean()),
                          }
        
        return loss_dict, scalar_outputs
        
        