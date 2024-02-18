import torch
import random
import cv2                
from .base import BaseVAE
from .utils import tensor2float, get_powers, image_checkerer, NormalizeNumpy, get_output_resolution
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
import numpy as np
# import sys
# sys.path.insert(0, '..')
# from utils import get_powers # not clean but work and avoid duplicating get_powers


Tensor = TypeVar('torch.tensor')

class SVDD_VAE(BaseVAE):
    def __init__(self,
                 img_res:tuple = (64,64),
                 in_channels: int = 1,
                 latent_dim: int  = 128,
                 hidden_dims: str = "32, 64, 128, 256, 512",
                 kernels_dims: str = "3,3,3,3,3",
                 stride: str = "2,2,2,2,2",
                 padding: str = "1,1,1,1,1",
                 svdd_nu:float = 0.1,
                 kld_weight:float = 0.001,
                 device:str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 debug:int = 0,
                 **kwargs) -> None:
        super(SVDD_VAE, self).__init__()

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
        self.debug = debug
        self.kld_weight = kld_weight
        self.svdd_R = torch.zeros(latent_dim, device=device)
        self.svdd_c = torch.randn(latent_dim, device=device)
        self.svdd_nu = svdd_nu  
        
        # Selection of dimensions to be shifted 
        mu_shift = kwargs["mu_shift"]
        if mu_shift != 0.0:
            s1, s2 = kwargs["mu_shift_dims"].split(':')
            if int(s2) == latent_dim: # ALL dims                
                self.mu_shift_dims = [i for i in range(latent_dim)]
                print(f"Selected latent space dimensions for Mu shift: ALL")
            elif s2 != "0":  # randomly a certain number of dimensions to be shifted
                self.mu_shift_dims = random.sample(range(latent_dim), int(s2)) 
                print(f"Selected latent space dimensions for Mu shift: {self.mu_shift_dims}")
            else:  # use specified dimensions
                self.mu_shift_dims = [int(n) for n in s1.split(",")]  #  pick the ones already selected
                print(f"Selected latent space dimensions for Mu shift: {self.mu_shift_dims}")
        print(f"Latent space dimensions forced shift: {mu_shift}")
        
        def hookinput(self, input, output, debug=self.debug):
        # https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html
            class_name = self.__class__.__name__
            if "1" in get_powers(debug): # add 2
                B, F, H, W = input[0].shape
                in_img = input[0].data
                cv2.imshow(f"[INPUT] {class_name} Res:{W}x{H}",in_img[0,0].detach().cpu().numpy())

        def hook(self, input, output, debug=self.debug):
        # https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html
            class_name = self.__class__.__name__
            if "0" in get_powers(debug): # add 1
                print('')
                print('Inside ' + class_name )
                print('input: ', type(input))
                print('input[0]: ', type(input[0]), input[0].shape)
                print('output: ', type(output), output.shape)
        
            if "1" in get_powers(debug): # add 2
                # plot input image of a given layer
                B, F, H, W = input[0].shape
                in_img = input[0].data
                cv2.imshow(f"[ENC-IN] {class_name} Res:{W}x{H}",in_img[0,0].detach().cpu().numpy())                
                # plot output Features of a given layer
                B, Nfeat, H, W = output.shape
                images = []
                for n in range(0,Nfeat, Nfeat//16):
                    images.append(NormalizeNumpy(output.data[0,n].detach().cpu().numpy()))
                img = image_checkerer(images)
                cv2.imshow(f"[ENC] {class_name} Feat:{int(Nfeat / (Nfeat//16))}/{Nfeat}", img)                
                # for filter in range(0, F, F//12):
                #     cv2.imshow(f"[ENC-OUT] {class_name} filt:{filter} Res:{W}x{H}",out_img[0,filter].detach().cpu().numpy())
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Build Encoder - TODO: must ensure that last layer has 1x1 resolution
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
        self.LLR = get_output_resolution(img_res, kernels_dims, stride, padding, verbose=True)
        print(f"Last encoder layer resolution: {self.LLR}")
        
        # Latent space Gaussian
        #############
        self.fc_mu = nn.Linear(hidden_dims[-1] * np.prod(self.LLR), latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * np.prod(self.LLR), latent_dim)
        
        # Register forward hooks on encoder layer TODO: make layer selectable?
        if self.debug != 0:  
            for i in range(len(hidden_dims)):
                print(f"REG FWD: {i}")
                # self.encoder[0].register_forward_hook(hookinput) # whole module
                self.encoder[i][0].register_forward_hook(hookinput) # Conv2D
                # self.encoder[i][1].register_forward_hook(hook) # Batch
                self.encoder[i][2].register_forward_hook(hook) # LeakyReLu
        

        # Build Decoder
        #############
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * np.prod(self.LLR))
        print("Decoder layers parameters: (in, out,kernel, stride, pad)")
        for h_dim_in, h_dim_out, k_size, strid, pad in zip(hidden_dims[-1:0:-1], hidden_dims[-2::-1],kernels_dims[-1:0:-1], stride[-1:0:-1], padding[-1:0:-1]):
            print(h_dim_in, h_dim_out, k_size, strid, pad)
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(h_dim_in,
                                       h_dim_out,
                                       kernel_size = k_size,
                                       stride = strid,
                                       padding = pad,
                                       output_padding = pad),
                    nn.BatchNorm2d(h_dim_out),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[0],
                                               hidden_dims[0],
                                               kernel_size = kernels_dims[0],
                                               stride = stride[0],
                                               padding = padding[0],
                                               output_padding = padding[0]),
                            nn.BatchNorm2d(hidden_dims[0]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[0], out_channels= self.Cin, kernel_size= kernels_dims[0], padding= padding[0]),
                            # nn.Tanh())
                            nn.Sigmoid())
        
        
    def get_svdd_R(self):
        return self.svdd_R.detach()
    
    def set_svdd_R(self, svdd_R):
        self.svdd_R = svdd_R
    
    def get_svdd_c(self):
        return self.svdd_c.detach()
    
    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

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

    def reparameterize(self, mu: Tensor, logvar: Tensor, N = 1) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        # std = torch.exp(0.5 * logvar)
        # eps = torch.randn_like(std)
        # z = mu + std * eps 
        mean = mu.unsqueeze(2).repeat(1, 1, N).squeeze()
        std = torch.exp(0.5 * logvar).unsqueeze(2).repeat(1, 1, N).squeeze()
        eps = torch.randn_like(std)
        return mean + std * eps 

    def forward(self, sample: Tuple, **kwargs) -> List[Tensor]:      
        input, _ = sample  
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]


    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE losses        
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0] # [B,C,H,W]
        input = args[1]  # [B,C,H,W]
        mu = args[2]      # [B,LS]
        log_var = args[3] # [B,LS]
        
        _, labels = args[4]  # [B]
        current_epoch = args[5]
        
        svdd_flag = kwargs["svdd_flag"]
        svdd_epoch = kwargs["svdd_epoch"]
        mu_shift = kwargs["mu_shift"]
        B, C, H, W = input.shape
        
        # shift mean of samples labelled as anomalies
        if mu_shift != 0:
            if len(self.mu_shift_dims) == self.latent_dim:
                mu += mu_shift * labels.unsqueeze(1) # [B, Ldim]
            else:
                for dim in self.mu_shift_dims:
                    mu[:,dim] += mu_shift * labels # [B, Ldim]
        
        # Reconstruction loss 
        recons_loss = torch.mean(F.mse_loss(recons, input, reduction='none'), dim=(2,3))* H * W # [B,1]
            
        # KL loss for each latent dimension
        kld_loss =  -0.5 * (1 + log_var - mu ** 2 - log_var.exp() ) * H * W   # [B, Ldims]
        
        # SVDD loss
        if svdd_flag and current_epoch > svdd_epoch:
            # evaluate
            z = self.reparameterize(mu, log_var)
            z_bunch = self.reparameterize(mu, log_var, N= 100) # [B, Ldims, N]
            self.svdd_c = z_bunch.mean(dim=(2)) # [B,  Ldims]
            
            dist = (z - self.svdd_c ) ** 2      # [B, Ldims]
            scores = dist - self.svdd_R ** 2    # [B, Ldims]
            svdd_loss = (self.svdd_R ** 2 + (1 / self.svdd_nu) * torch.max(torch.zeros_like(scores), scores) ) # [B, Ldims]            
            # dist = torch.mean( (z - self.svdd_c)**2, dim=1)      # [B]
            # scores = dist - self.svdd_R ** 2    # [B]
            # svdd_loss = self.svdd_R ** 2 + (1 / self.svdd_nu) * torch.mean(torch.max(torch.zeros_like(scores), scores) ) # [B, Ldims]
            
            # Update
            self.svdd_R = torch.quantile(dist.detach().sqrt(), 1-self.svdd_nu, dim=0)
            
        else:
            svdd_loss = torch.zeros_like(kld_loss)
                    
        # Total loss # [B, Ldims]
        loss = recons_loss \
                + self.kld_weight * kld_loss \
                + svdd_loss 
        
        loss_dict = {'loss': loss, 
                     'Recons_Loss': recons_loss.detach(), 
                     'KLD_Loss': kld_loss.detach(), 
                     'Other_Loss': svdd_loss.detach()}
        
        scalar_outputs = {"full_loss": tensor2float(loss.mean()),
                          "recons_loss": tensor2float(recons_loss.mean()),
                          "KL_loss": tensor2float(self.kld_weight*kld_loss.mean()),
                          }
        
        if svdd_flag and current_epoch > svdd_epoch:          
                    scalar_outputs["SVDD_loss"] = tensor2float(svdd_loss.mean())
                    scalar_outputs["SVDD_R"] = tensor2float(self.get_svdd_R().mean())
                    scalar_outputs["SVDD_centroid"] = tensor2float(self.get_svdd_c().mean())
                    
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
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples



# Input size: Cin=1 H=64 W=64
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1           [-1, 32, 32, 32]             320
#        BatchNorm2d-2           [-1, 32, 32, 32]              64
#          LeakyReLU-3           [-1, 32, 32, 32]               0
#             Conv2d-4           [-1, 64, 16, 16]          18,496
#        BatchNorm2d-5           [-1, 64, 16, 16]             128
#          LeakyReLU-6           [-1, 64, 16, 16]               0
#             Conv2d-7            [-1, 128, 8, 8]          73,856
#        BatchNorm2d-8            [-1, 128, 8, 8]             256
#          LeakyReLU-9            [-1, 128, 8, 8]               0
#            Conv2d-10            [-1, 256, 4, 4]         295,168
#       BatchNorm2d-11            [-1, 256, 4, 4]             512
#         LeakyReLU-12            [-1, 256, 4, 4]               0
#            Conv2d-13            [-1, 512, 2, 2]       1,180,160
#       BatchNorm2d-14            [-1, 512, 2, 2]           1,024
#         LeakyReLU-15            [-1, 512, 2, 2]               0
#            Linear-16                  [-1, 128]         262,272
#            Linear-17                  [-1, 128]         262,272
#            Linear-18                 [-1, 2048]         264,192
#   ConvTranspose2d-19            [-1, 256, 4, 4]       1,179,904
#       BatchNorm2d-20            [-1, 256, 4, 4]             512
#         LeakyReLU-21            [-1, 256, 4, 4]               0
#   ConvTranspose2d-22            [-1, 128, 8, 8]         295,040
#       BatchNorm2d-23            [-1, 128, 8, 8]             256
#         LeakyReLU-24            [-1, 128, 8, 8]               0
#   ConvTranspose2d-25           [-1, 64, 16, 16]          73,792
#       BatchNorm2d-26           [-1, 64, 16, 16]             128
#         LeakyReLU-27           [-1, 64, 16, 16]               0
#   ConvTranspose2d-28           [-1, 32, 32, 32]          18,464
#       BatchNorm2d-29           [-1, 32, 32, 32]              64
#         LeakyReLU-30           [-1, 32, 32, 32]               0
#   ConvTranspose2d-31           [-1, 32, 64, 64]           9,248
#       BatchNorm2d-32           [-1, 32, 64, 64]              64
#         LeakyReLU-33           [-1, 32, 64, 64]               0
#            Conv2d-34            [-1, 1, 64, 64]             289
#              Tanh-35            [-1, 1, 64, 64]               0
# ================================================================
# Total params: 3,936,481
# Trainable params: 3,936,481
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.02
# Forward/backward pass size (MB): 5.94
# Params size (MB): 15.02
# Estimated Total Size (MB): 20.97
# ----------------------------------------------------------------