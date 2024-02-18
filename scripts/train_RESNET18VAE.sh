#!/usr/bin/env bash

# Launch script with
#   bash ./scripts/<script_name>.sh <log_dir>
#   
# bash ./scripts/train_RESNET18VAE.sh RESNET18VAE_split0.2_Dim128_KLw0.005
# bash ./scripts/train_RESNET18VAE.sh RESNET18VAE_split0.2_Dim128_KLw0.005_bal_aug0.8



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
--config=cfgs/resnet18_vae.yaml \
model.name "resnet18_vae" \
model.mode "train" \
model.resume False \
model.resume_force_LR False \
model.loadckpt "None" \
dataset.fixinbalance False \
dataset.augmentation 0. \
architecture.latent_dim 128 \
architecture.kld_weight 0.005 \
architecture.KLD_Loss_threshold 500 \
architecture.KLD_Loss_dims "all" \
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

# architecture.feat_layers "0,3,7,10,14,17,20,23,27,30,33,36,40,43,46,49:0" \ # Con2D
# architecture.feat_layers: "2,5,9,12,16,19,22,25,29,32,35,38,42,45,48,51:0" \ # ReLu
