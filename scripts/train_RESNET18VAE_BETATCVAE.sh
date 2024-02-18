#!/usr/bin/env bash

# Launch script with
#   bash ./scripts/<script_name>.sh <log_dir>
#   
# bash ./scripts/train_RESNET18VAE_BETATCVAE.sh RESNET18VAE_BETATCVAE_split0.2_Dim128_KLw0.001_MI0.01_TC0.02
# bash ./scripts/train_RESNET18VAE_BETATCVAE.sh RESNET18VAE_BETATCVAE_split0.2_Dim128_KLw0.001_MI0.01_TC0.02_bal_aug0.8


# CHKPT="outputs/.../model_58.ckpt.ckpt"

exp=$1
PY_ARGS=${@:2}
LOG_DIR="./outputs/"$exp 
LOG_FILE="log_"$exp".txt"
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

echo "====== Copying run script to running folder ${LOG_DIR}"
cp ./scripts/train_RESNET18VAE.sh ${LOG_DIR}/.

echo "====== Check log in file: tail -f ${LOG_DIR}/${LOG_FILE}"

python main.py \
--config=cfgs/resnet18_betatc_vae.yaml \
model.name "resnet18_betatc_vae" \
model.mode "train" \
model.resume False \
model.resume_force_LR False \
model.loadckpt "None" \
dataset.fixinbalance True \
dataset.augmentation 0.8 \
architecture.latent_dim 128 \
architecture.kld_weight 0.01 \
architecture.KLD_Loss_threshold 0.6 \
architecture.KLD_Loss_dims "all" \
architecture.anneal_steps 200 \
architecture.alpha_mi_loss 0.01 \
architecture.beta_tc_loss 0.02 \
training.max_epochs 75 \
training.lrepochs "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28:1.1" \
training.wd 0.0 \
training.batch_size 32 \
logging.logdir $LOG_DIR \
logging.summary_freq 32 \
debug.debug 0 \
$PY_ARGS &> $LOG_DIR"/"$LOG_FILE &


# architecture.KLD_Loss_dims "all" \
