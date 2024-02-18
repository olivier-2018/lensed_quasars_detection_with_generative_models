import os
from PIL import Image
import numpy as np
import random
  
from astropy.io import fits 
from astropy.visualization import astropy_mpl_style
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from torch.utils.data import  Dataset
from torchvision import transforms as T
import torch


DEBUG = False # local debugging

# std_datasets_list = {'CIFAR10': 
#                         {"name":CIFAR10, feat:}
#                      'CelebA': CelebA,
#                      'FashionMNIST': FashionMNIST,
#                      'STL10': STL10
#                     }

class HSCSSP_Dataset(Dataset):
    """
    Args:
        data_dir: dataset root directory
        cand_fname: candidate list filename (train or test)
    """
    def __init__(self,
                datapath:str = "../dataset",
                dataset_name:str = "HSC-SSP_DR4_james",
                img_resize:tuple = 47,
                list_path:str = "./lists",
                list_names: str = "train.txt",
                **kwargs,
            ):
        super().__init__()

        self.data_dir = os.path.join(datapath, dataset_name)
        self.cand_fname = os.path.join(list_path, list_names)
        self.cand_list = self.build_list()
        self.img_resize = (img_resize, img_resize)
        augment_proba = kwargs["augmentation"]
        
        # Process numpy array    
        # https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html
        # https://pytorch.org/vision/stable/auto_examples/index.html
        # https://pytorch.org/vision/0.15/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
        # RandomPerspective, RandomRotation, GaussianBlur, RandomAffine, ElasticTransform, vRandomCrop, RandomResizedCrop
        # RandomInvert, RandomPosterize , RandomSolarize, RandomAdjustSharpness,RandomAutocontrast, RandomEqualize
        # AutoAugment, RandAugment, TrivialAugmentWide, AugMix, RandomHorizontalFlip, RandomVerticalFlip, RandomApply  
        augment_transform = T.transforms.RandomChoice([
                                                        # T.RandomPerspective(distortion_scale=0.3, p=1.0), 
                                                        # T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                                                        # T.RandomRotation(degrees=(-180, 180), interpolation=T.InterpolationMode.BILINEAR),
                                                        T.RandomRotation(degrees=(90, 90)), 
                                                        T.RandomRotation(degrees=(-90, -90)), 
                                                        T.RandomRotation(degrees=(180, 180)),
                                                        T.RandomHorizontalFlip(p=1), 
                                                        T.RandomVerticalFlip(p=1)
                                                        ])
            
        self.transfo = T.transforms.Compose(
                        [
                            T.ToTensor(), 
                            T.Resize(self.img_resize, antialias=True),
                            T.Lambda(lambda x: (x - x.min() ) /  (x.max() - x.min())),
                            # T.Lambda(lambda x: torch.where(x > torch.rand((x.shape[0], 1)).to(x.device), 1.0, 0.0)),         # binarize   
                            T.RandomApply([augment_transform], p=augment_proba)
                        ])

    def build_list(self):
        # scan candidates list file to retrieve .fits filenames
        cand_list = []
        with open(self.cand_fname) as f:
            for line in f:
                candidate, label = line.strip().split(", ")
                cand_list.append((candidate+"_HSC-I_img.fits", int(label)))
        return cand_list     
    
    def __len__(self):
        return len(self.cand_list)
    
    def NormalizeNumpy(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def read_fits_img(self, filename, id):        
        # Get fits data (see astropy lib.)
        hdulist = fits.open(filename, mode='readonly')
        np_array = hdulist[id].data
        if DEBUG: print (f"[HubbleDataset] hduData min/max: {np_array.min():.2f}/{np_array.max():.2f}")
        info = None #hdulist.info(output = False)
        hdulist.close()
        # Normalize np array
        # torch_array =  torch.from_numpy(np_array)
        # Optional, transform to PIL image to enable torch transforms downstream
        # PIL_img = Image.fromarray(np.uint8(np_array)) #.convert('RGB')                
        return np_array, info

        
    def __getitem__(self, index):
        
        cand_name, cand_label = self.cand_list[index]
        if DEBUG: print (f"[HubbleDataset] index:{index}, IDname: {cand_name}, label: {cand_label}")
        
        # Set relevant data index in fits file (to be checked for each new dataset)
        img_id = 1 
        
        # WARNING - Assumes dataset folder structure, change as required 
        fname = os.path.join(self.data_dir, "fits_data", cand_name)
        
        # Read fits file and extract img data  
        # Nb: info is hdulist.info() i.e of class None, use print to display
        np_array, info = self.read_fits_img(fname, img_id)
   
        processed_tensor = self.transfo(np_array.newbyteorder().byteswap()) # torch.Size([1, 64, 64])
        
        # return processed_tensor, cand_label, cand_name
        return processed_tensor, cand_label
    
        
if __name__ == '__main__':
    
    # Instantiate dataset
    kwargs = {"augmentation": 0.8}
    dataset = HSCSSP_Dataset(data_path = "../dataset", 
                            dataset_name = "HSC-SSP_DR4_james",
                            path_size = (64,64),
                            list_path = "./lists/HSC-SSP_DR4_james/testtrain_split0.2",
                            list_names = "train.txt",
                            **kwargs)
    # Create iterator
    iterator= iter(dataset)
    
    # Print a couple of images
    for _ in range(3):
        img, label = next(iterator)
        # print(type(img))
        
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,  figsize=(16,6))
        fig.suptitle(f" (label={label})")
        img1 = ax1.imshow(img.squeeze().numpy(), cmap='gray')
        fig.colorbar(img1)
        ax1.set_title("Linear color scale")
        img2 = ax2.imshow(img.squeeze().numpy(),  cmap='gray', norm=LogNorm())
        fig.colorbar(img2)
        ax2.set_title("LogNorm color scale")
        plt.show(block=True)
    