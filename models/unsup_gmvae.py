import torch
import random
import cv2                
from .base import BaseVAE
from .utils import tensor2float, get_powers, image_checkerer, NormalizeNumpy, get_output_resolution, cluster_accuracy, nmi_score
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
import numpy as np
import math
# import sys
# sys.path.insert(0, '..')
# from utils import get_powers # not clean but work and avoid duplicating get_powers


Tensor = TypeVar('torch.tensor')



class Qy_x(nn.Module):
    """Conditional distribution q(y|x) represented by a neural network.
    Args:   encoder (nn.Module): The encoder module used to process the input data.
            enc_out_dim (int): The output dimension of the encoder module.
            k (int): Number of components in the Gaussian mixture prior.
    Attributes:
        encoder (nn.Module): The encoder module used to process the input data.
        qy_logit (nn.Linear): Linear layer for predicting the logit of q(y|x).
        qy (nn.Softmax): Softmax activation function for q(y|x).
    """
    def __init__(self, encoder, enc_out_dim, k):
        super(Qy_x, self).__init__()
        self.enc = encoder
        self.qy_logit = nn.Linear(enc_out_dim, k)
        self.qy = nn.Softmax(dim=1)

    def forward(self, x):
        """Perform the forward pass for q(y|x).
        Args:  x (torch.Tensor): Input data tensor.
        Returns:  tuple: A tuple containing the logit and softmax outputs of q(y|x).
        """
        x = self.enc(x)                     # [B, Nfeat=512, H=2, W=2])
        x = torch.flatten(x, start_dim =1)  # [B, 2048])
        qy_logit = self.qy_logit(x)         # [B,k_clusters] 
        qy = self.qy(qy_logit)              # [B,k_clusters] proba
        return qy_logit, qy


class Qz_xy(nn.Module):
    """Conditional distribution q(z|x, y) represented by a neural network.
    Args:
        k (int): Number of components in the Gaussian mixture prior.
        encoder (nn.Module): encoder module 
        enc_out_dim (int): encoder module output dimension (flattened if CNNs)
        hidden_size (int): Number of units in the hidden layer(s).
        latent_dim (int): Dimensionality of the latent space.
    Attributes:
        enc (nn.Module): encoder module network
        h2 (nn.Sequential): neural network (with 2 hidden layers) of q(z|x, y).
        z_mean (nn.Linear): Linear layer for predicting the mean of q(z|x, y).
        zlogvar (nn.Linear): Linear layer for predicting the log variance of q(z|x, y).
    """
    def __init__(self, k, encoder, enc_out_dim, hidden_size, latent_dim):
        super(Qz_xy, self).__init__()
        self.enc = encoder
        self.h2 = nn.Sequential(nn.Linear(enc_out_dim + k, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU()
                                )
        self.z_mean = nn.Linear(hidden_size, latent_dim)
        self.zlogvar = nn.Linear(hidden_size, latent_dim)

    def gaussian_sample(self, z_mean, z_logvar):
        z_std = torch.sqrt(torch.exp(z_logvar))
        eps = torch.randn_like(z_std)
        z = z_mean + eps*z_std
        return z

    def forward(self, x, y):
        """ forward pass for q(z|x, y).
        Args:   x (torch.Tensor): Input data tensor.
                y (torch.Tensor): One-hot encoded tensor representing the class labels.
        Returns: tuple: A tuple containing the latent variables, mean, and log variance of q(z|x, y).
        """
        x = self.enc(x)                     # [B, Nfeat=512, H=2, W=2])
        x = torch.flatten(x, start_dim=1)   # [B, EncOutDim=2048]
        xy = torch.cat((x, y), dim=1)
        h2 = self.h2(xy)                    # [B, HiddenSize]
        # q(z|x, y)
        z_mean = self.z_mean(h2)
        z_logvar = self.zlogvar(h2)
        z = self.gaussian_sample(z_mean, z_logvar)
        return z, z_mean, z_logvar


class Px_z(nn.Module):
    """Conditional distribution p(x|z) represented by a neural network.
    Args: decoder (nn.Module): The decoder module used to reconstruct the data.
          k (int): Number of components in the Gaussian mixture prior.
    Attributes: decoder (nn.Module): The decoder module used to reconstruct the data.
                decoder_hidden (int): Number of units in the hidden layer of the decoder.
                latent_dim (int): Dimensionality of the latent space.
                z_mean (nn.Linear): Linear layer for predicting the mean of p(z|y).
                zlogvar (nn.Linear): Linear layer for predicting the log variance of p(z|y).
    """
    def __init__(self, decoder, hidden_size, latent_dim, k):
        super(Px_z, self).__init__()
        self.decoder = decoder
        self.decoder_hidden = hidden_size
        self.latent_dim = latent_dim
        
        self.z_mean = nn.Linear(k, self.latent_dim)
        self.zlogvar = nn.Linear(k, self.latent_dim)
        self.relu = nn.ReLU()
        
    def forward(self, z, y):
        """Perform the forward pass for p(x|z) and p(z|y).
        Args:   z (torch.Tensor): Latent variable tensor. 
                y (torch.Tensor): One-hot encoded tensor representing the class labels.
        Returns:  tuple: A tuple containing the prior mean, prior log variance, and reconstructed data.
        """
        # p(z|y)
        z_mean = self.z_mean(y)     # [B, Ldims]
        zlogvar = self.zlogvar(y)   # [B, Ldims]

        # p(x|z)
        zy = z_mean + torch.sqrt(F.softplus(zlogvar)) * z  # new: non-degenerate Gaussian mixture latent layer from 
        x_hat = self.decoder(self.relu(zy)) # new
        # x_hat = self.decoder(z)
        return z_mean, zlogvar, x_hat
    
    
class UNSUP_GMVAE(BaseVAE):
    """Based on GMVAE improvements from https://github.com/RuiShu/vae-clustering
    """
    def __init__(self,
                 img_res:tuple = (64,64),
                 in_channels: int = 1,
                 latent_dim: int  = 128,
                 hidden_dims: str = "32, 64, 128, 256, 512",
                 kernels_dims: str = "3,3,3,3,3",
                 stride: str = "2,2,2,2,2",
                 padding: str = "1,1,1,1,1",
                 kld_weight:float = 0.001,
                 Nclusters: int = 10,
                 hidden_size: int = 256,
                 device:str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 debug:int = 0,
                 **kwargs) -> None:
        super(UNSUP_GMVAE, self).__init__()

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
        self.k = Nclusters
        self.hidden_size = hidden_size

        # Build Encoder 
        #############
        modules = []
        for h_dim, k_size, strid, pad in zip(hidden_dims,kernels_dims,stride,padding):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, 
                              out_channels=h_dim,
                              kernel_size= k_size, 
                              stride= strid, 
                              padding  = pad),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules) 
        
        # Evaluate encoder last layer resolution 
        self.enc_out_dim = get_output_resolution(img_res, kernels_dims, stride, padding)
        print(f"Last encoder layer resolution: {self.enc_out_dim}")
        
        # Build Decoder
        #############
        modules = []
        modules.append(nn.Linear(latent_dim, hidden_dims[-1] * np.prod(self.enc_out_dim)))
        modules.append(nn.Unflatten(1, (512,2,2))) 
        for h_dim_in, h_dim_out, k_size, strid, pad in zip(hidden_dims[-1:0:-1], hidden_dims[-2::-1],kernels_dims[-1:0:-1], stride[-1:0:-1], padding[-1:0:-1]):
            # print(h_dim_in, h_dim_out, k_size, strid, pad)
            modules.append( nn.Sequential(  nn.ConvTranspose2d(h_dim_in,
                                                            h_dim_out,
                                                            kernel_size = k_size,
                                                            stride = strid,
                                                            padding = pad,
                                                            output_padding = pad),
                                            nn.BatchNorm2d(h_dim_out),
                                            nn.LeakyReLU()
                                        )
                            )
        modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[0],
                                                        hidden_dims[0],
                                                        kernel_size = kernels_dims[0],
                                                        stride = stride[0],
                                                        padding = padding[0],
                                                        output_padding = padding[0]),
                                    nn.BatchNorm2d(hidden_dims[0]),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(hidden_dims[0], out_channels= self.Cin, kernel_size= kernels_dims[0], padding= padding[0]),
                                    # nn.Tanh())
                                    nn.Sigmoid()
                                    )
                        )        
        self.decoder = nn.Sequential(*modules)
        
        enc_output_dims = np.prod(self.enc_out_dim)*hidden_dims[-1]
        self.qy_x = Qy_x(self.encoder, enc_output_dims, self.k)
        self.qz_xy = Qz_xy(self.k, self.encoder, enc_output_dims, self.hidden_size, self.latent_dim)
        self.px_z = Px_z(self.decoder, self.hidden_size, self.latent_dim, self.k)


    def forward(self, sample):
        """Perform forward pass through the GMVAE model.
        Args: x (torch.Tensor): Input data tensor.
        Returns:
            tuple: A tuple containing two dictionaries. 
            The first dictionary contains training-related outputs such as z, zm, zv, zm_prior, zv_prior, qy_logit, qy, and px. 
            The second dictionary contains inference-related outputs such as y_hat, z_hat, x_hat, and qy.
        """
        x, labels = sample
        k = self.k
        batch_size = x.shape[0]
        y_ = torch.zeros([batch_size, k]).to(x.device) # [B, k]
        
        # Eval conditional proba of a class y given x: qy_x
        qy_logit, qy = self.qy_x(x) # [B,K] and [B,K] proba
        
        # Evaluate mu & Sigma priors from px_z (based on reconstruction from cluster latent variables)
        # Evaluate mu & Sigma from qz_xy (based on input and infered cluster)
        z, zm, zv, zm_prior, zv_prior, px = [[None] * k for i in range(6)]
        for i in range(k):
            y = y_ + torch.eye(k).to(x.device)[i]                   # [B, k]
            z[i], zm[i], zv[i] = self.qz_xy(x, y)                   # all [B, Ldims] 
            zm_prior[i], zv_prior[i], px[i] = self.px_z(z[i], y)    # [B, Ldims], [B, Ldims], [B,C,H,W]

        # Inference for x_hat:
        # class label assigned to the cluster based on the most-frequently-occuring label within a cluster?
        with torch.no_grad():
            y_hat = torch.argmax(qy, dim=-1)                        # [B]
            y_temp = torch.zeros(batch_size, k)                     # [B, k]
            y_temp = torch.scatter(y_, 1, y_hat.unsqueeze(1), 1)
            z_hat, *_ = self.qz_xy(x, y_temp)  # [B, Ldims] infer latent variables based on most likely cluster
            *_, x_hat = self.px_z(z_hat, y_temp) # [B,1,H,W] reconstructi from latent variables based on most likely cluster

        out_train = {"z": z,
                     "zm": zm,
                     "zv": zv,
                     "zm_prior": zm_prior,
                     "zv_prior": zv_prior,
                     "qy_logit": qy_logit,
                     "qy": qy,
                     "px": px
                     }
        out_infer = {"y_hat": y_hat,
                     "z_hat": z_hat,
                     "x_hat": x_hat,
                     "qy": qy
                     }
        
        return  [x_hat, x, out_train, out_infer]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        generative process:
            p_theta(x, y, z) = p(y) p_theta(z|y) p_theta(x|z,c)   # Multiplication rule
            p_theta(x, y, z) = p(y) p_theta(z|y) p_theta(x|z)     #  z and c dependents
        with
            y ~ Cat(y|1/k)
            z|y ~ N(z; mu_z_theta(y), sigma^2_z_theta(y))
            x|z ~ Bern(x; mu_x_theta(z)) or ~N(x; mu_x_theta(z), sigma^2_x_theta(z))

        The goal of GMVAE is to estimate the posterior distribution p(z, y|x), which is usually difficult to compute directly.
        Instead, a factorized posterior, known as the inference model, is commonly used as an approximation:

            q_phi(z, y|x) = q_phi(z|x, y) q_phi(y|x)  # Multiplication rule
        with
            y|x ~ Cat(y|pi_phi(x))
            z|x, y ~ N(z; mu_z_phi(x, y), sigma^2z_phi(x, y))  # priors

        ELBO = + Eq_phi(z|x,y) [log p_theta(x|z)]    # reconstruction
                -KL(q_phi(z|x, y) || p_theta(z|y))   # KLD between variational posterior q(z|c,x) and GM prior p(z|x,c) 
                    - KL(q_phi(y|x) || p(y))         # KLD 

        Returns:
            dict: _description_
        """
        # Get variables from args and dicts
        recons = args[0] # [B,C,H,W]
        input = args[1]  # [B,C,H,W]
        out_train = args[2] 
        out_infer = args[3] 
        
        _, true_labels = args[4] 
        current_epoch = args[5]
        dataset_size = args[6]
        
        B, C, H, W = input.shape
        
        # Gen Model outputs
        qy_logit, qy = out_train["qy_logit"],out_train["qy"]   # [B,K] logits and corresponding proba (via softmax) 
        px = out_train["px"]                                    # reconstruction img for each class: list of [B,1,H,W]
        z = out_train["z"]                                                  # [B, Ldims] latent space variable
        zm_prior, zv_prior = out_train["zm_prior"],  out_train["zv_prior"]  # list: 10 Tensor[B, Ldims]  px_z(z[i],y) class prior 
        zm, zv = out_train["zm"], out_train["zv"]                           # list: 10 Tensor[B, Ldims]  qz_xy(x, y)  

        # Infer outputs
        y_hat = out_infer["y_hat"]
        
        # conditional entropy loss (categorical loss)
        # loss_qy = torch.sum(F.binary_cross_entropy_with_logits(qy_logit, qy, reduction="none"), dim=1)     # [B]
        # loss_qy = torch.sum(qy * torch.nn.LogSoftmax(1)(qy_logit), 1)
        loss_qy = self.neg_entropy_from_logit(qy, qy_logit)
        
        # loss per cluster
        losses_i = []        
        recon_loss_i = []
        kld_loss_i = []
        for i in range(self.k):
            rec_loss = torch.sum(F.mse_loss(input, px[i], reduction='none'), dim=(1,2,3))   # [B]
            # rec_loss = self.BCELogits(input, px[i], eps=1E-8)                                       # [B]
            # rec_loss = self.log_bernoulli_with_logits(input, px[i], eps=1E-8)   
            
            log_norm = self.log_normal(z[i], zm[i], torch.exp(zv[i]), axis=None)                    # [B, Ldims]
            log_norm_prior = self.log_normal(z[i], zm_prior[i], torch.exp(zv_prior[i]), axis=None)  # [B, Ldims]            
            KL_loss = log_norm - log_norm_prior - np.log(1/self.k)                                  # [B, Ldims]
            
            loss_per_cluster = rec_loss + self.kld_weight * KL_loss.sum(1)           # [B]
        
            recon_loss_i.append(rec_loss)
            kld_loss_i.append(KL_loss)
            losses_i.append(loss_per_cluster)
        
        recons_loss = torch.stack([qy[:, i] * recon_loss_i[i] for i in range(self.k)]).sum(0)           # [B]
        kld_loss = torch.stack([qy[:, i].unsqueeze(1) * kld_loss_i[i] for i in range(self.k)]).sum(0)   # [B, Ldims]
        
        loss = torch.stack([loss_qy] + [qy[:, i] * losses_i[i] for i in range(self.k)]).sum(0)          # [B]        
        # loss =  torch.sum(torch.mul(torch.stack(losses_i), torch.transpose(qy, 1, 0)), dim=0) # Alternative way to calculate loss:
        
        # Metrics 
        # Warning: use numpy so vaiables must be transfered to cpu hence huge bottleneck during training  # TODO: rewrite for Tensors
        # Note: a bit useless in unsupervised training anyway
        # NMI = nmi_score(y_hat.cpu(), labels.cpu())
        # cluster_acc = cluster_accuracy(y_hat.cpu(), labels.cpu())
        
        # Wraping up
        loss_dict = {'loss': loss, 
                     'Recons_Loss': recons_loss.detach(), 
                     'KLD_Loss': kld_loss.detach(), 
                     'Other_Loss': loss_qy.detach()
                     }
        
        scalar_outputs = {"full_loss": tensor2float(loss.mean()),
                          "recons_loss": tensor2float(recons_loss.mean()),
                          "KL_loss": tensor2float(self.kld_weight * kld_loss.sum(1).mean()),
                          "Other_Loss": tensor2float(loss_qy.mean()),
                          "pred_cluster": tensor2float(y_hat[0]),
                        }
                        #   "cluster_accuracy": tensor2float(cluster_acc),
                        #   "nmi": tensor2float(NMI),
                          
     
        for i in range(self.k):
            scalar_outputs
  
        return loss_dict, scalar_outputs

    def BCELogits(self, x, px_logits, eps=1E-8):
        batch_size = x.shape[0]
        if eps > 0.0:
            max_val = np.log(1.0 - eps) - np.log(eps)
            px_logits = torch.clamp(px_logits, -max_val, max_val)
        loss = torch.sum(F.binary_cross_entropy(px_logits, x, reduction='none'), dim=(1,2,3))   # [B]
        return loss
    
    def log_bernoulli_with_logits(self, x, px_logits, eps=0.0):
        if eps > 0.0:
            max_val = np.log(1.0 - eps) - np.log(eps)
            px_logits = torch.clamp(px_logits, -max_val, max_val)
        loss = torch.sum(torch.nn.MultiLabelSoftMarginLoss(reduction='none')(px_logits, x),  dim=(1,2))   # [B]
        return loss

    def neg_entropy_from_logit(self, qy, qy_logit):
        """
        Definitions:
            Entropy(qy, qy) = - ∑qy * log qy
            Entropy(qy, qy_logit) = - ∑qy * log p(qy_logit)
            p(qy_logit) = softmax(qy_logit)
            Entropy(qy, qy_logit) = - ∑qy * log softmax(qy_logit)
        """
        return torch.sum(qy * torch.nn.LogSoftmax(1)(qy_logit), 1)

    def log_normal(self, x, mu, var, eps=1E-8, axis=-1):
        """
        Logarithm of normal distribution with mean=mu and variance=var
         log(x|μ, σ^2) = loss = -0.5 * Σ log(2π) + log(σ^2) + ((x - μ)/σ)^2
        Args:
            x (torch.Tensor): Input tensor.
            mu (torch.Tensor): Mean tensor of the normal distribution.
            var (torch.Tensor): Variance tensor of the normal distribution.
            eps (float, optional): A small value added to the variance to avoid numerical instability. Defaults to 0.0.
            axis (int, optional): The axis along which the log probabilities are summed. Defaults to -1.
        Returns:
            torch.Tensor: The computed log probability of the normal distribution.
        """
        if eps > 0.0:
            var = torch.add(var, eps)
        if axis == None:
            log_norm = -0.5 * (np.log(2 * math.pi) + torch.log(var) + torch.pow(x - mu, 2) / var) # [B, Ldims]
        else:
            log_norm = -0.5 * torch.sum(np.log(2 * math.pi) + torch.log(var) + torch.pow(x - mu, 2) / var, axis) # [B]            
        return log_norm



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
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

