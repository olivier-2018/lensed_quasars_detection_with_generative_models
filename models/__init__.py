import importlib
from .svdd_vae import SVDD_VAE
from .dfc_vae import DFC_VAE
from .resnet18_vae import RESNET18_VAE
from .betatc_vae import BetaTC_VAE
from .resnet18_btc_vae import RESNET18_BETATC_VAE
from .unsup_gmvae import UNSUP_GMVAE
from .semisup_gmvae import SEMISUP_GMVAE

# Define model filename and model class name
# key: must be the model filename (ex: "vae" for vae.py)
# item: name of the class defined for the model (ex: VAE_CNN in vae.py)
models_classnames = {'svdd_vae': SVDD_VAE,
                     'dfc_vae': DFC_VAE,
                     'resnet18_vae': RESNET18_VAE,
                     'betatc_vae': BetaTC_VAE,
                     'resnet18_betatc_vae': RESNET18_BETATC_VAE,
                     'unsup_gmvae': UNSUP_GMVAE,
                     'semisup_gmvae': SEMISUP_GMVAE,
                    }

# # find and import model by name, for example model "vae" from vae.py
# def find_model(model_name):
#     module_name = 'models.{}'.format(model_name)
#     module = importlib.import_module(module_name)

#     return getattr(module, model_classnames[model_name])
