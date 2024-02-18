import torch
from .base import BaseVAE
from torch import nn
from torch.nn import functional as F
import math
from typing import List, Callable, Union, Any, TypeVar, Tuple
import cv2   
from .utils import tensor2float, get_powers, NormalizeNumpy, image_checkerer, get_output_resolution

import numpy as np


Tensor = TypeVar('torch.tensor')

class BetaTC_VAE(BaseVAE):
    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 img_res:tuple = (64,64),
                 in_channels: int = 1,
                 latent_dim: int = 64,
                 hidden_dims: str = "32,32,32,32",
                 kernels_dims: str = "4,4,4,4",
                 stride: str = "2,2,2,2",
                 padding: str = "1,1,1,1",
                 kld_weight:float = 0.001,
                 anneal_steps: int = 200,
                 alpha_mi_loss: float = 1.,
                 beta_tc_loss: float =  6.,
                 device:str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 debug:int = 0,
                 **kwargs) -> None:
        super(BetaTC_VAE, self).__init__()

        hidden_dims = [int(n) for n in hidden_dims.split(",")]
        kernels_dims = [int(n) for n in kernels_dims.split(",")]
        stride = [int(n) for n in stride.split(",")]
        padding = [int(n) for n in padding.split(",")]        
        
        self.Cin = in_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.kernels_dims = kernels_dims
        self.stride = stride
        self.padding = padding
        self.device = device
        self.debug = debug
        self.kld_weight = kld_weight

        self.anneal_steps = anneal_steps
        self.alpha_mi_loss = alpha_mi_loss
        self.beta_tc_loss = beta_tc_loss

        # Build Encoder - TODO: must ensure that last layer has 1x1 resolution
        #############
        modules = []
        for h_dim, k_size, strid, pad in zip(hidden_dims,kernels_dims,stride,padding):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, 
                              out_channels = h_dim,
                              kernel_size = k_size, 
                              stride = strid, 
                              padding = pad),
                    # nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim 
        self.encoder = nn.Sequential(*modules)
        
        # Evaluate encoder last layer resolution 
        self.LLR = get_output_resolution(img_res, kernels_dims, stride, padding)
        print(f"Last encoder layer resolution: {self.LLR}")
        
        # Latent space Gaussian
        #############
        self.fc = nn.Linear(hidden_dims[-1] * self.LLR[0] * self.LLR[1], 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)
        # self.fc_mu = nn.Linear(hidden_dims[-1] * np.prod(self.LLR), latent_dim)
        # self.fc_var = nn.Linear(hidden_dims[-1] * np.prod(self.LLR), latent_dim)

        # Build Decoder
        #############
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * np.prod(self.LLR))
        # self.decoder_input = nn.Linear(latent_dim, 256 * 2)
        print("Decoder layers parameters: (in, out,kernel, stride, pad)")
        for h_dim_in, h_dim_out, k_size, strid, pad in zip(hidden_dims[-1:0:-1], hidden_dims[-2::-1],kernels_dims[-1:0:-1], stride[-1:0:-1], padding[-1:0:-1]):
            print(h_dim_in, h_dim_out, k_size, strid, pad)
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(h_dim_in,
                                       h_dim_out,
                                       kernel_size = k_size-1,
                                       stride = strid,
                                       padding = pad,
                                       output_padding = pad),
                    # nn.BatchNorm2d(h_dim_out),
                    nn.LeakyReLU())
            )
            
        self.decoder = nn.Sequential(*modules)
        
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[0],
                                               hidden_dims[0],
                                               kernel_size = kernels_dims[0]-1,
                                               stride = stride[0],
                                               padding = padding[0],
                                               output_padding = padding[0]),
                            # nn.BatchNorm2d(hidden_dims[0]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[0], 
                                      out_channels= self.Cin, 
                                      kernel_size= kernels_dims[0]-1, 
                                      padding= padding[0]),
                            # nn.Tanh())
                            nn.Sigmoid())
        
    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        result = self.fc(result)
        
        # Split the result into mu and var components of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], self.LLR[0], self.LLR[1])
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, sample: Tuple, **kwargs) -> List[Tensor]:
        input, _ = sample
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var, z]

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

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        return self.decode(z)

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]