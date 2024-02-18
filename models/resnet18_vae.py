from .base import BaseVAE
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from .utils import tensor2float, get_powers, NormalizeNumpy


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


class RESNET18_VAE(BaseVAE):
    def __init__(self, 
                 latent_dim: int,
                 kld_weight:float = 0.001,
                 device:str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 debug:int = 0,
                 **kwargs) -> None:
        super().__init__()
        
        self.device = device
        self.debug = debug
        self.kld_weight = kld_weight
        Cin = kwargs['in_channels']
        
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
        
        B, C, H, W = input.shape
        
        # Reconstruction loss (averaged over B,C,H,W)
        # recons_loss =F.mse_loss(recons, input)        
        
        # Reconstruction loss (averaged over C,W,H and scaled to image dims)
        recons_loss = torch.mean(F.mse_loss(recons, input, reduction='none'), dim=(1,2,3)) * H*W # [B,1]
        
        # KL loss for each latent dimension
        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld_loss = -0.5 * (1 + log_var - mu ** 2 - log_var.exp() ) * H*W  # [B, Ldims]
        

        loss = recons_loss \
                + self.kld_weight * torch.mean(kld_loss, dim = 1)
        
        loss_dict = {'loss': loss, 
                     'Recons_Loss': recons_loss.detach(), 
                     'KLD_Loss': kld_loss.detach(),
                     'Other_Loss': torch.zeros_like(kld_loss),
                     }
        
        scalar_outputs = {"full_loss": tensor2float(loss.mean()),
                          "recons_loss": tensor2float(recons_loss.mean()),
                          "KL_loss": tensor2float(self.kld_weight*torch.mean(kld_loss, dim = 1).mean()),
                          }
        
        return loss_dict, scalar_outputs
    
    
    
    
## ResNet18

# Input size: Cin=1 H=64 W=64
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1           [-1, 64, 32, 32]           1,728
#        BatchNorm2d-2           [-1, 64, 32, 32]             128
#             Conv2d-3           [-1, 64, 32, 32]          36,864
#        BatchNorm2d-4           [-1, 64, 32, 32]             128
#             Conv2d-5           [-1, 64, 32, 32]          36,864
#        BatchNorm2d-6           [-1, 64, 32, 32]             128
#      BasicBlockEnc-7           [-1, 64, 32, 32]               0
#             Conv2d-8           [-1, 64, 32, 32]          36,864
#        BatchNorm2d-9           [-1, 64, 32, 32]             128
#            Conv2d-10           [-1, 64, 32, 32]          36,864
#       BatchNorm2d-11           [-1, 64, 32, 32]             128
#     BasicBlockEnc-12           [-1, 64, 32, 32]               0
#            Conv2d-13          [-1, 128, 16, 16]          73,728
#       BatchNorm2d-14          [-1, 128, 16, 16]             256
#            Conv2d-15          [-1, 128, 16, 16]         147,456
#       BatchNorm2d-16          [-1, 128, 16, 16]             256
#            Conv2d-17          [-1, 128, 16, 16]           8,192
#       BatchNorm2d-18          [-1, 128, 16, 16]             256
#     BasicBlockEnc-19          [-1, 128, 16, 16]               0
#            Conv2d-20          [-1, 128, 16, 16]         147,456
#       BatchNorm2d-21          [-1, 128, 16, 16]             256
#            Conv2d-22          [-1, 128, 16, 16]         147,456
#       BatchNorm2d-23          [-1, 128, 16, 16]             256
#     BasicBlockEnc-24          [-1, 128, 16, 16]               0
#            Conv2d-25            [-1, 256, 8, 8]         294,912
#       BatchNorm2d-26            [-1, 256, 8, 8]             512
#            Conv2d-27            [-1, 256, 8, 8]         589,824
#       BatchNorm2d-28            [-1, 256, 8, 8]             512
#            Conv2d-29            [-1, 256, 8, 8]          32,768
#       BatchNorm2d-30            [-1, 256, 8, 8]             512
#     BasicBlockEnc-31            [-1, 256, 8, 8]               0
#            Conv2d-32            [-1, 256, 8, 8]         589,824
#       BatchNorm2d-33            [-1, 256, 8, 8]             512
#            Conv2d-34            [-1, 256, 8, 8]         589,824
#       BatchNorm2d-35            [-1, 256, 8, 8]             512
#     BasicBlockEnc-36            [-1, 256, 8, 8]               0
#            Conv2d-37            [-1, 512, 4, 4]       1,179,648
#       BatchNorm2d-38            [-1, 512, 4, 4]           1,024
#            Conv2d-39            [-1, 512, 4, 4]       2,359,296
#       BatchNorm2d-40            [-1, 512, 4, 4]           1,024
#            Conv2d-41            [-1, 512, 4, 4]         131,072
#       BatchNorm2d-42            [-1, 512, 4, 4]           1,024
#     BasicBlockEnc-43            [-1, 512, 4, 4]               0
#            Conv2d-44            [-1, 512, 4, 4]       2,359,296
#       BatchNorm2d-45            [-1, 512, 4, 4]           1,024
#            Conv2d-46            [-1, 512, 4, 4]       2,359,296
#       BatchNorm2d-47            [-1, 512, 4, 4]           1,024
#     BasicBlockEnc-48            [-1, 512, 4, 4]               0
#            Linear-49                  [-1, 256]         131,328
#       ResNet18Enc-50     [[-1, 128], [-1, 128]]               0
#            Linear-51                  [-1, 512]          66,048
#            Conv2d-52            [-1, 512, 4, 4]       2,359,296
#       BatchNorm2d-53            [-1, 512, 4, 4]           1,024
#            Conv2d-54            [-1, 512, 4, 4]       2,359,296
#       BatchNorm2d-55            [-1, 512, 4, 4]           1,024
#     BasicBlockDec-56            [-1, 512, 4, 4]               0
#            Conv2d-57            [-1, 512, 4, 4]       2,359,296
#       BatchNorm2d-58            [-1, 512, 4, 4]           1,024
#            Conv2d-59            [-1, 256, 8, 8]       1,179,904
#      ResizeConv2d-60            [-1, 256, 8, 8]               0
#       BatchNorm2d-61            [-1, 256, 8, 8]             512
#            Conv2d-62            [-1, 256, 8, 8]       1,179,904
#      ResizeConv2d-63            [-1, 256, 8, 8]               0
#       BatchNorm2d-64            [-1, 256, 8, 8]             512
#     BasicBlockDec-65            [-1, 256, 8, 8]               0
#            Conv2d-66            [-1, 256, 8, 8]         589,824
#       BatchNorm2d-67            [-1, 256, 8, 8]             512
#            Conv2d-68            [-1, 256, 8, 8]         589,824
#       BatchNorm2d-69            [-1, 256, 8, 8]             512
#     BasicBlockDec-70            [-1, 256, 8, 8]               0
#            Conv2d-71            [-1, 256, 8, 8]         589,824
#       BatchNorm2d-72            [-1, 256, 8, 8]             512
#            Conv2d-73          [-1, 128, 16, 16]         295,040
#      ResizeConv2d-74          [-1, 128, 16, 16]               0
#       BatchNorm2d-75          [-1, 128, 16, 16]             256
#            Conv2d-76          [-1, 128, 16, 16]         295,040
#      ResizeConv2d-77          [-1, 128, 16, 16]               0
#       BatchNorm2d-78          [-1, 128, 16, 16]             256
#     BasicBlockDec-79          [-1, 128, 16, 16]               0
#            Conv2d-80          [-1, 128, 16, 16]         147,456
#       BatchNorm2d-81          [-1, 128, 16, 16]             256
#            Conv2d-82          [-1, 128, 16, 16]         147,456
#       BatchNorm2d-83          [-1, 128, 16, 16]             256
#     BasicBlockDec-84          [-1, 128, 16, 16]               0
#            Conv2d-85          [-1, 128, 16, 16]         147,456
#       BatchNorm2d-86          [-1, 128, 16, 16]             256
#            Conv2d-87           [-1, 64, 32, 32]          73,792
#      ResizeConv2d-88           [-1, 64, 32, 32]               0
#       BatchNorm2d-89           [-1, 64, 32, 32]             128
#            Conv2d-90           [-1, 64, 32, 32]          73,792
#      ResizeConv2d-91           [-1, 64, 32, 32]               0
#       BatchNorm2d-92           [-1, 64, 32, 32]             128
#     BasicBlockDec-93           [-1, 64, 32, 32]               0
#            Conv2d-94           [-1, 64, 32, 32]          36,864
#       BatchNorm2d-95           [-1, 64, 32, 32]             128
#            Conv2d-96           [-1, 64, 32, 32]          36,864
#       BatchNorm2d-97           [-1, 64, 32, 32]             128
#     BasicBlockDec-98           [-1, 64, 32, 32]               0
#            Conv2d-99           [-1, 64, 32, 32]          36,864
#      BatchNorm2d-100           [-1, 64, 32, 32]             128
#           Conv2d-101           [-1, 64, 32, 32]          36,864
#      BatchNorm2d-102           [-1, 64, 32, 32]             128
#    BasicBlockDec-103           [-1, 64, 32, 32]               0
#           Conv2d-104            [-1, 3, 64, 64]           1,731
#     ResizeConv2d-105            [-1, 3, 64, 64]               0
#      ResNet18Dec-106            [-1, 3, 64, 64]               0
# ================================================================
# Total params: 23,910,275
# Trainable params: 23,910,275
# Non-trainable params: 0
# ----------------------------------------------------------------