import torch
from .base import BaseVAE
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
import cv2   
from .utils import tensor2float, get_powers, NormalizeNumpy, image_checkerer, get_output_resolution
from torchvision.models import vgg19_bn, VGG19_BN_Weights
from torchvision.models import Inception3
from torchvision.models import resnet18, ResNet18_Weights


Tensor = TypeVar('torch.tensor')

class DFC_VAE(BaseVAE):
# Deep Feature Consistent Variational Autoencoder, Xianxu Hou (https://arxiv.org/abs/1610.00291)
    def __init__(self,
                 img_res:tuple = (64,64),
                 in_channels: int = 1,
                 latent_dim: int = 128,
                 hidden_dims: str = "32, 64, 128, 256, 512",
                 kernels_dims: str = "3,3,3,3,3",
                 stride: str = "2,2,2,2,2",
                 padding: str = "1,1,1,1,1",
                 feat_model:str = "vgg19",
                 feat_layers:str = "14,24,34,43:0",
                 kld_weight:float = 0.001,
                 alpha_recons_loss:float = 1,
                 beta_feat_loss:float = 0.5,
                 device:str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 debug:int = 0,
                 **kwargs) -> None:
        super(DFC_VAE, self).__init__()

        hidden_dims = [int(n) for n in hidden_dims.split(",")]
        kernels_dims = [int(n) for n in kernels_dims.split(",")]
        stride = [int(n) for n in stride.split(",")]
        padding = [int(n) for n in padding.split(",")]        
        feat_layers = [str(n) for n in feat_layers.split(':')[0].split(",")]
        
        self.Cin = in_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.kernels_dims = kernels_dims
        self.stride = stride
        self.padding = padding
        self.device = device
        self.debug = debug
        self.kld_weight = kld_weight
        
        self.feat_layers = feat_layers
        self.alpha_recons_loss = alpha_recons_loss
        self.beta_feat_loss = beta_feat_loss

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
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        
        # Evaluate encoder last layer resolution 
        self.LLR = get_output_resolution(img_res, kernels_dims, stride, padding)
        print(f"Last encoder layer resolution: ({self.LLR})")
        
        # Latent space Gaussian
        #############
        # self.fc = nn.Linear(hidden_dims[-1] * self.LLR[0] * self.LLR[1], 256)
        # self.fc_mu = nn.Linear(256, latent_dim)
        # self.fc_var = nn.Linear(256, latent_dim)
        self.fc_mu = nn.Linear(hidden_dims[-1] * self.LLR[0] * self.LLR[1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * self.LLR[0] * self.LLR[1], latent_dim)

        # Build Decoder
        #############
        modules = [] 
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * self.LLR[0] * self.LLR[1]) 
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

        # Pretrained model 
        #############
        if feat_model == "vgg19":
            # self.feature_network = vgg19_bn(pretrained=True)
            self.feature_network = vgg19_bn(weights="DEFAULT") # "IMAGENET1K_V1"
            
        # elif feat_model == "inception3": # TODO
        #     self.feature_network = Inception3()
        #     self.feature_network.features = self.feature_network._modules
            
        # elif feat_model == "resnet18": # TODO
        #     self.feature_network = resnet18(pretrained=True) #weights="ResNet18_Weights")
            
        else:
            print("No valid feature network.")
            exit()
        
        # Freeze the pretrained feature network
        for param in self.feature_network.parameters():
            param.requires_grad = False
        self.feature_network.eval()


    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # result = self.fc(result)
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
        recons = self.decode(z)

        input_features = self.extract_features(input, self.feat_layers)
        recons_features = self.extract_features(recons, self.feat_layers)

        return  [recons, input, mu, log_var, recons_features, input_features]

    def extract_features(self,
                         input: Tensor,
                         feature_layers: List = None) -> List[Tensor]:
        """
        Extracts the features from the pretrained model at the layers indicated by feature_layers.
        :param input: (Tensor) [B x C x H x W]
        :param feature_layers: List of string of IDs
        :return: List of the extracted features
        """
        if feature_layers is None:
            feature_layers = ['14', '24', '34', '43']
        features = []
        result = input.expand([-1,3,-1,-1])
        for (key, module) in self.feature_network.features._modules.items():
            result = module(result)
            if(key in feature_layers):
                features.append(result)
                ### DEBUG
                if "0" in get_powers(self.debug): # add 1: display feature
                    Nfeat = result.shape[1]
                    print(f"[DEBUG] layer:{key} => {module} ") 
                    images = []
                    for n in range(0,Nfeat, Nfeat//16):
                        images.append(NormalizeNumpy(result[0,n].detach().cpu().numpy()))
                    img = image_checkerer(images)
                    cv2.imshow(f"Layer:{key} Feat:{int(Nfeat / (Nfeat//16))}/{Nfeat}", img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

        return features

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
        recons_features = args[4] # list: [torch([1,256,16,16]), torch([1,256,16,16]), torch([1,512,8,8]), torch([1,512,4,4]) ]
        input_features = args[5]
        
        _, labels = args[6] # [B]
        current_epoch = args[7]
        
        B, C, H, W = input.shape
        # kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        
        # Reconstruction loss (averaged over B,C,H,W)
        # recons_loss =F.mse_loss(recons, input)        
        
        # Reconstruction loss (averaged over C,W,H and scaled to image dims)
        recons_loss = torch.mean(F.mse_loss(recons, input, reduction='none'), dim=(2,3)) * H*W # [B,1]

        # Features loss (averaged over B,C,H,W necessary since done on VGG16 net)
        feature_loss = 0.0
        for (recons_feature, input_feature) in zip(recons_features, input_features):
            feature_loss += F.mse_loss(recons_feature, input_feature) 
            
        # Features loss (averaged over C,H,W only and scaled to image dims)
        # feature_loss = torch.zeros(self.latent_dim, device=self.device)        
        # for (recons_feature, input_feature) in zip(recons_features, input_features):
        #     feature_loss += torch.mean(F.mse_loss(recons_feature, input_feature, reduction='none'), dim=(2,3)) *H*W

        # KL loss for each latent dimension
        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld_loss = -0.5 * (1 + log_var - mu ** 2 - log_var.exp() ) * H*W  # [B, Ldims]
        

        loss = self.alpha_recons_loss * recons_loss.mean() \
                + self.beta_feat_loss * feature_loss \
                + self.kld_weight * torch.mean(kld_loss, dim = 1).mean()
        
        loss_dict = {'loss': loss, 
                     'Recons_Loss': recons_loss.detach(), 
                     'KLD_Loss': kld_loss.detach(),
                     'Other_Loss': torch.zeros_like(kld_loss),
                     }
        
        scalar_outputs = {"full_loss": tensor2float(loss.mean()),
                          "recons_loss": tensor2float(self.alpha_recons_loss*recons_loss.mean()),
                          "Feat_loss": tensor2float(self.beta_feat_loss*feature_loss),
                          "KL_loss": tensor2float(self.kld_weight*torch.mean(kld_loss, dim = 1).mean()),
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
        z = torch.randn(num_samples,self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
    
    
    
    
## VGG19 
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
#            Linear-16                  [-1, 256]         524,544
#            Linear-17                  [-1, 256]         524,544
#            Linear-18                 [-1, 2048]         526,336
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
#            Conv2d-36           [-1, 64, 64, 64]           1,792
#       BatchNorm2d-37           [-1, 64, 64, 64]             128
#              ReLU-38           [-1, 64, 64, 64]               0
#            Conv2d-39           [-1, 64, 64, 64]          36,928
#       BatchNorm2d-40           [-1, 64, 64, 64]             128
#              ReLU-41           [-1, 64, 64, 64]               0
#         MaxPool2d-42           [-1, 64, 32, 32]               0
#            Conv2d-43          [-1, 128, 32, 32]          73,856
#       BatchNorm2d-44          [-1, 128, 32, 32]             256
#              ReLU-45          [-1, 128, 32, 32]               0
#            Conv2d-46          [-1, 128, 32, 32]         147,584
#       BatchNorm2d-47          [-1, 128, 32, 32]             256
#              ReLU-48          [-1, 128, 32, 32]               0
#         MaxPool2d-49          [-1, 128, 16, 16]               0
#            Conv2d-50          [-1, 256, 16, 16]         295,168
#       BatchNorm2d-51          [-1, 256, 16, 16]             512
#              ReLU-52          [-1, 256, 16, 16]               0
#            Conv2d-53          [-1, 256, 16, 16]         590,080
#       BatchNorm2d-54          [-1, 256, 16, 16]             512
#              ReLU-55          [-1, 256, 16, 16]               0
#            Conv2d-56          [-1, 256, 16, 16]         590,080
#       BatchNorm2d-57          [-1, 256, 16, 16]             512
#              ReLU-58          [-1, 256, 16, 16]               0
#            Conv2d-59          [-1, 256, 16, 16]         590,080
#       BatchNorm2d-60          [-1, 256, 16, 16]             512
#              ReLU-61          [-1, 256, 16, 16]               0
#         MaxPool2d-62            [-1, 256, 8, 8]               0
#            Conv2d-63            [-1, 512, 8, 8]       1,180,160
#       BatchNorm2d-64            [-1, 512, 8, 8]           1,024
#              ReLU-65            [-1, 512, 8, 8]               0
#            Conv2d-66            [-1, 512, 8, 8]       2,359,808
#       BatchNorm2d-67            [-1, 512, 8, 8]           1,024
#              ReLU-68            [-1, 512, 8, 8]               0
#            Conv2d-69            [-1, 512, 8, 8]       2,359,808
#       BatchNorm2d-70            [-1, 512, 8, 8]           1,024
#              ReLU-71            [-1, 512, 8, 8]               0
#            Conv2d-72            [-1, 512, 8, 8]       2,359,808
#       BatchNorm2d-73            [-1, 512, 8, 8]           1,024
#              ReLU-74            [-1, 512, 8, 8]               0
#         MaxPool2d-75            [-1, 512, 4, 4]               0
#            Conv2d-76            [-1, 512, 4, 4]       2,359,808
#       BatchNorm2d-77            [-1, 512, 4, 4]           1,024
#              ReLU-78            [-1, 512, 4, 4]               0
#            Conv2d-79            [-1, 512, 4, 4]       2,359,808
#       BatchNorm2d-80            [-1, 512, 4, 4]           1,024
#              ReLU-81            [-1, 512, 4, 4]               0
#            Conv2d-82            [-1, 512, 4, 4]       2,359,808
#       BatchNorm2d-83            [-1, 512, 4, 4]           1,024
#              ReLU-84            [-1, 512, 4, 4]               0
#            Conv2d-85            [-1, 512, 4, 4]       2,359,808
#       BatchNorm2d-86            [-1, 512, 4, 4]           1,024
#              ReLU-87            [-1, 512, 4, 4]               0
#         MaxPool2d-88            [-1, 512, 2, 2]               0
#            Conv2d-89           [-1, 64, 64, 64]           1,792
#       BatchNorm2d-90           [-1, 64, 64, 64]             128
#              ReLU-91           [-1, 64, 64, 64]               0
#            Conv2d-92           [-1, 64, 64, 64]          36,928
#       BatchNorm2d-93           [-1, 64, 64, 64]             128
#              ReLU-94           [-1, 64, 64, 64]               0
#         MaxPool2d-95           [-1, 64, 32, 32]               0
#            Conv2d-96          [-1, 128, 32, 32]          73,856
#       BatchNorm2d-97          [-1, 128, 32, 32]             256
#              ReLU-98          [-1, 128, 32, 32]               0
#            Conv2d-99          [-1, 128, 32, 32]         147,584
#      BatchNorm2d-100          [-1, 128, 32, 32]             256
#             ReLU-101          [-1, 128, 32, 32]               0
#        MaxPool2d-102          [-1, 128, 16, 16]               0
#           Conv2d-103          [-1, 256, 16, 16]         295,168
#      BatchNorm2d-104          [-1, 256, 16, 16]             512
#             ReLU-105          [-1, 256, 16, 16]               0
#           Conv2d-106          [-1, 256, 16, 16]         590,080
#      BatchNorm2d-107          [-1, 256, 16, 16]             512
#             ReLU-108          [-1, 256, 16, 16]               0
#           Conv2d-109          [-1, 256, 16, 16]         590,080
#      BatchNorm2d-110          [-1, 256, 16, 16]             512
#             ReLU-111          [-1, 256, 16, 16]               0
#           Conv2d-112          [-1, 256, 16, 16]         590,080
#      BatchNorm2d-113          [-1, 256, 16, 16]             512
#             ReLU-114          [-1, 256, 16, 16]               0
#        MaxPool2d-115            [-1, 256, 8, 8]               0
#           Conv2d-116            [-1, 512, 8, 8]       1,180,160
#      BatchNorm2d-117            [-1, 512, 8, 8]           1,024
#             ReLU-118            [-1, 512, 8, 8]               0
#           Conv2d-119            [-1, 512, 8, 8]       2,359,808
#      BatchNorm2d-120            [-1, 512, 8, 8]           1,024
#             ReLU-121            [-1, 512, 8, 8]               0
#           Conv2d-122            [-1, 512, 8, 8]       2,359,808
#      BatchNorm2d-123            [-1, 512, 8, 8]           1,024
#             ReLU-124            [-1, 512, 8, 8]               0
#           Conv2d-125            [-1, 512, 8, 8]       2,359,808
#      BatchNorm2d-126            [-1, 512, 8, 8]           1,024
#             ReLU-127            [-1, 512, 8, 8]               0
#        MaxPool2d-128            [-1, 512, 4, 4]               0
#           Conv2d-129            [-1, 512, 4, 4]       2,359,808
#      BatchNorm2d-130            [-1, 512, 4, 4]           1,024
#             ReLU-131            [-1, 512, 4, 4]               0
#           Conv2d-132            [-1, 512, 4, 4]       2,359,808
#      BatchNorm2d-133            [-1, 512, 4, 4]           1,024
#             ReLU-134            [-1, 512, 4, 4]               0
#           Conv2d-135            [-1, 512, 4, 4]       2,359,808
#      BatchNorm2d-136            [-1, 512, 4, 4]           1,024
#             ReLU-137            [-1, 512, 4, 4]               0
#           Conv2d-138            [-1, 512, 4, 4]       2,359,808
#      BatchNorm2d-139            [-1, 512, 4, 4]           1,024
#             ReLU-140            [-1, 512, 4, 4]               0
#        MaxPool2d-141            [-1, 512, 2, 2]               0
# ================================================================
# Total params: 44,793,953
# Trainable params: 4,723,169
# Non-trainable params: 40,070,784
# ----------------------------------------------------------------
# Input size (MB): 0.02
# Forward/backward pass size (MB): 63.35
# Params size (MB): 170.88
# Estimated Total Size (MB): 234.24
# ----------------------------------------------------------------


