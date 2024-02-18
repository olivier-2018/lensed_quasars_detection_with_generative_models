# Lensed Quasar detection with VAEs

This repo presents a semester project at the FHNW (Switzerland) to use generative modeling to learn the probability density distribution of cosmological objects to detect lensed quasar candidtaes from a large imbalance transient survey dataset.

### Installation
```
$ git clone git@github.com:olivier-2018/lensed_quasars_detection_with_generative_models.git
$ cd lensed_quasars_detection_with_generative_models
$ virtualenv -p /usr/bin/python3.8 venv_quasar
$ source venv_quasar/bin/activate
$ pip install -r requirements.txt
```

### Dataset
The original dataset corresponds to the processed data from the HSC-SPP (Hyper SuprimeCam Strategic Survey Program) Data Release 4 and is available on request from Chan et Al. (2023). 

### Dataset convention
Assume dataset folder structure as:
```
<dataset_name>/
├── candidates_list_raw.txt
├── candidates_list_selected.txt
└── fits_data
    ├── J000000.78+045553.67_HSC-I_img.fits
    ├── J000001.41+003704.79_HSC-I_img.fits
    ├── J000001.46-002723.02_HSC-I_img.fits
    ├── ...
```

### Available models
Available models are defined in "models/__init__.py".

Currently available models are:
- SVDD_VAE: further constrained latent space whereby the encoder network also learns an hypersphere around normal data and penalize anomalies outside of the hypershere  (Wang et al.)
- DFC_VAE: an architecture to enfoce deep feature consistency by complementing the VAE reconstruction loss with feature conceptual loss evaluated by comparing features generated with a pretrained VGG19 neural network (Hou et al.)
- RESNET18_VAE: ResNet like encode-decoder architecture i.e deeper predictive power.
- BetaTC_VAE: loss formulation proposed to try and disentangle the latent space by exploiting a measure of the total correlation (TC) between latent variables and the mutual information gap (MIG), a measure of disentanglement (Chen et al)
- RESNET18_BETATC_VAE: combinedResNet18 architecture and BetaTC_VAE loss
- UNSUP_GMVAE: Unsupervised Gaussian Mixture Model
- SEMISUP_GMVAE: Semi-supervised Gaussian Mixture Model

### Training
```
$ python main.py --config=cfgs/<CFG_FILE>>.yaml 
```
or
```
$ bash script/launch_training.sh <Log_dir>
```
or
Use VSCODE debugging capability (see .vscode/.launch.json)

### Analysis / Classification
Set model:mode in CFG file to either "AD_analysis" or "anomaly_classification"<br>
or<br>
Use VSCODE debugging capability (see .vscode/.launch.json)

### Background
Strong gravitational lensing in astrophysics is a phenomenon which occurs when lines of sight to a foreground and background object nearly coincide. <br>
The light the background object may emit is then deflected by the gravitational field of the foreground object, resulting in multiple imaging of the background object. <br>
Although quite rare, it offers an important diagnostic of masses and mass distributions in foreground objects ranging from stars to clusters of galaxies, and in particular, it is sensitive to all kinds of matter thus allowing a further insight into the structure of the universe.

<img src="https://www.eso.org/public/archives/images/screen/eso1313b.jpg" width="600" /><br>
Gravitational lensing of distant star-forming galaxies (Credit: ALMA (ESO/NRAO/NAOJ), L. Calçada (ESO), Y. Hezaveh et al.)

Quasars are relatively rare phenomena and lensed quasars, in which a foreground galaxy provides the gravitational deflection, are correspondingly rare. <br>
The detection and study of lensed Quasars offer a range of advantages in astrophysics such as an access to the redshift universe (i.e an insight into galaxy located billions of light years away), a magnified view of quasars (otherwise very tiny and difficult to detect), and access to various cosmological information. 

<img src="doc/HSC_gradeA.png" width="600" /><br>

With recent technological improvements available on modern telescopes and space program probes such as Hubble or Euclid, a huge number of high resolution images is becoming available and requires fast and reliable methods to detect lensed Quasars. <br>
However, despite recent progress, lensed Quasar detection still often require human visual inspection to validate the correct identification of lensed quasar candidates.

Given the extremely rare occurrence of lensed quasars in cosmological image datasets the task to detect such phenomena falls into anomaly detection in imbalanced unlabelled datasets. The challenge however is to develop and train a deep neural network which can correctly model the probability distribution of such stellar objects and thereafter develop the correct metrics to detect whether a given anomaly from that distribution indeed corresponds to a lensed quasar.

The proposed approach is to use generative modeling to learn the probability density distribution of cosmological objects using the post-processed dataset from Chan et Al. (2023) corresponding to the Data Release 4 of the HSC-SSP transient survey.

### Results

Reconstruction of a few stellar systems: 

<img src="doc/reconstruction.png" width="1000" />

Latent space distribution of normal vs anomaly sets using a ResNet18 encoder-decoder architecture and a betaTC loss function. <br>
Note that the latent dimensions in which the anomaly information is embedded can be reliably discovered using the Chi² metric.  

<img src="doc/resnet18-BTC-aug-vae_metrics.png" width="900" />

Using the Chi² metric to isolate the latent dimensions in which anomaly is embedded, it becomes possible to set a classification threshold whereby new sample images can be classified.

<img src="./doc/resnet18-BTC-aug-vae_threshold_study_5_dims.png.png" width="400" />

A ResNet18-BetaTC-VAE model achieves the best classification results with a precision, recall and F1 score of 65\%, 27\% and 38\% respectively in a hugely imbalanced dataset.
