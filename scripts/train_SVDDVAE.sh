#!/usr/bin/env bash

# Launch script with
#   bash ./scripts/train_SVDDVAE.sh <log_dir>
#   
# bash ./scripts/train_SVDDVAE.sh SVDDVAE_split0.2_Dim128_KLw0.005_noSVDD
# bash ./scripts/train_SVDDVAE.sh SVDDVAE_split0.2_Dim128_KLw0.005_noSVDD_bal
# bash ./scripts/train_SVDDVAE.sh SVDDVAE_split0.2_Dim128_KLw0.005_noSVDD_bal_aug0.9
# bash ./scripts/train_SVDDVAE.sh SVDDVAE_split0.2_Dim128_KLw0.005_noSVDD_bal_aug0.9b (no rotation)

# bash ./scripts/train_SVDDVAE.sh SVDDVAE_split0.2_Dim128_KLw0.005_SVDD
# bash ./scripts/train_SVDDVAE.sh SVDDVAE_split0.2_Dim128_KLw0.005_SVDD_bal
# bash ./scripts/train_SVDDVAE.sh SVDDVAE_split0.2_Dim128_KLw0.005_SVDD_bal_aug0.8

# bash ./scripts/train_SVDDVAE.sh SVDDVAE_split0.2_Dim128_KLw0.005_noSVDD_Mu+5_all
# bash ./scripts/train_SVDDVAE.sh SVDDVAE_split0.2_Dim128_KLw0.005_noSVDD_Mu+5_12dims


# CHKPT="outputs/VAE_CNN_split0-0.2-0.2_Dim256_KLw0.005_SVDDnu0.2/model_58.ckpt.ckpt"

exp=$1
PY_ARGS=${@:2}
LOG_DIR="./outputs/"$exp 
LOG_FILE="log_"$exp".txt"
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

echo "====== Copying run script to running folder ${LOG_DIR}"
cp ./scripts/train_SVDDVAE.sh ${LOG_DIR}/.

echo "====== Check log in file: tail -f ${LOG_DIR}/${LOG_FILE}"

python main.py \
--config=cfgs/svdd_vae.yaml \
model.name "svdd_vae" \
model.mode "train" \
model.resume False \
model.resume_force_LR False \
model.loadckpt "None" \
dataset.fixinbalance True \
dataset.augmentation 0.8 \
architecture.latent_dim 128 \
architecture.kld_weight 0.005 \
architecture.svdd_flag True \
architecture.svdd_epoch 2 \
architecture.svdd_nu 0.1 \
architecture.mu_shift 0.0 \
architecture.mu_shift_dims "0,1,2,3,4,5,6,7,8,9,10,11:0" \
training.max_epochs 45 \
training.lrepochs "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28:1.1" \
training.wd 0.0 \
training.batch_size 32 \
logging.logdir $LOG_DIR \
logging.summary_freq 32 \
debug.debug 0 \
$PY_ARGS &> $LOG_DIR"/"$LOG_FILE &

# model.loadckpt=$CHKPT \
# training.lrepochs="2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28:1.2" \
# training.lrepochs="5,10,15,20,25,30,35,40,45,50,60:1.2" \

# architecture.mu_shift_dims "88,51,92,18,58,75,76,80:0" \
