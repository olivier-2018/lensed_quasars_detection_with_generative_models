import argparse, yaml
import os, sys, time, gc, datetime, io
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchsummary import summary as Msummary
from torch.utils.data import DataLoader, WeightedRandomSampler
from tensorboardX import SummaryWriter
from datasets import HSCSSP_Dataset #, std_datasets_list
from torchvision.datasets import CIFAR10, CelebA, FashionMNIST, STL10
from torchvision import transforms as T
from torchvision.transforms import v2 as T2
from torchvision.transforms import Compose
from models import models_classnames
from models.utils import *
import yaml
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg',force=True)
from PIL import Image 
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, precision_recall_curve,  confusion_matrix, roc_curve, auc, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import pandas as pd
import seaborn as sns
import math, scipy, sklearn
from yacs.config import CfgNode as CN
from yacs.config import load_cfg
import git
import _pickle as pickle


# Improve performance if model does not change and input sizes remain the same 
cudnn.benchmark = True

# Arguments parser
parser = argparse.ArgumentParser(description='Lensed quasar detection with VAEs')
parser.add_argument('--config', dest="filename", metavar='FILE', default='cfgs/base_vae.yaml')
parser.add_argument("opts", default=None,nargs=argparse.REMAINDER,help="Modify config options using the command-line")  


# main functions
def train(model, optimizer, TrainImgLoader, TestImgLoader, start_epoch, tb_logger, cfg):
    
    milestones = [len(TrainImgLoader) * int(epoch_idx) for epoch_idx in cfg.training.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(cfg.training.lrepochs.split(':')[1])

    # LR scheduler
    # https://towardsdatascience.com/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863
    if cfg.training.lr_scheduler == 'MS':
        lr_scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=lr_gamma, warmup_factor=1.0/3, warmup_iters=100,
                                                            last_epoch=len(TrainImgLoader) * start_epoch - 1)

    # Epoch loop
    for epoch_idx in range(start_epoch, cfg.training.max_epochs):        
        print('Epoch {}:'.format(epoch_idx+1))
        
        # Batch loop
        model.train()
        for batch_idx, sample in enumerate(TrainImgLoader):
            
            # Init
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx 
            do_summary = global_step % cfg.logging.summary_freq == 0            

            # FORWARD PASS
            outputs = model(tocuda(sample)) 
            recons_img, input_img = outputs[0], outputs[1]
            # assert not torch.isnan(outputs[]).any(), "Reconstructed image with NaNs"
            
            # LOSS EVALUATION                      
            loss_dict, scalar_outputs = model.loss_function(*outputs, 
                                                            tocuda(sample), 
                                                            epoch_idx, 
                                                            len(TrainImgLoader.dataset),
                                                            **cfg.architecture) # [B, Ldim]
            
            # BACKWARD PROP & UPDATE PARAMETERS
            optimizer.zero_grad()
            loss_dict["loss"].mean().backward()
            optimizer.step()
        
            # Training summary
            if do_summary:
                # TB Logger info                  
                scalar_outputs["LR"] = tensor2float(optimizer.param_groups[0]["lr"])
                    
                image_outputs = {"recons_img": tensor2numpy(recons_img[0]),
                                "input_img": tensor2numpy(input_img[0]),
                                "dummy": tensor2numpy(input_img[0]),
                                } 
                
                save_scalars(tb_logger, 'train', scalar_outputs, global_step)
                save_images(tb_logger, 'train', image_outputs, global_step)
                del scalar_outputs, image_outputs 
                
                # Std output prints
                print(
                    "Epoch:{}/{}, Train iter:{}/{}, lr={:.2E}, loss={:.3f}, KL_loss={:.3f}, Recons_loss={:.3f}, time = {:.3f}".format(
                        epoch_idx+1, cfg.training.max_epochs, batch_idx, len(TrainImgLoader),
                        optimizer.param_groups[0]["lr"], 
                        loss_dict["loss"].mean(),
                        loss_dict["KLD_Loss"].mean(),
                        loss_dict["Recons_Loss"].mean(),
                        time.time() - start_time)
                    )     
                sys.stdout.flush() 
                                    
            # STEP LR (except on last batch, otherwise testing will be done at different LR)
            if not batch_idx == len(TrainImgLoader)-1:
                lr_scheduler.step() 

        ################ checkpoint ################
        if (epoch_idx + 1) % cfg.logging.save_freq == 0:
            # Debug model
            # for param_tensor in model.state_dict():
            #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
            # for var_name in optimizer.state_dict():
            #     print(var_name, "\t", optimizer.state_dict()[var_name])
            
            save_dict ={'epoch': epoch_idx,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }
            if cfg.model.name == "svdd_vae": save_dict['SVDD_R'] = model.get_svdd_R()
            
            torch.save(save_dict, "{}/model_{:02d}.ckpt".format(cfg.logging.logdir, epoch_idx)) 
            print("Model saved !")
            
        gc.collect()

        ################ validation ################
        if (epoch_idx % cfg.logging.eval_freq == 0) or (epoch_idx == cfg.training.max_epochs - 1):
            model.eval()
            with torch.no_grad():
                avg_test_scalars = DictAverageMeter()
                class_metrics = ClassificationMetrics()
                
                for batch_idx, sample in enumerate(TestImgLoader):
                    
                    # Init
                    sample = tocuda(sample)
                    start_time = time.time()
                    global_step = len(TrainImgLoader) * epoch_idx + batch_idx 
                    do_summary = global_step % cfg.logging.summary_freq == 0                    
                    
                    # FORWARD PASS
                    outputs = model(sample)
                    recons_img, input_img = outputs[0], outputs[1]
                    if cfg.model.name in ['semisup_gmvae', 'unsup_gmvae']:
                        y_hat, y_hat_proba = outputs[3]["y_hat"], outputs[3]["qy"].max(1)[0]

                    # LOSS EVALUATION                                                      
                    loss_dict, scalar_outputs = model.loss_function(*outputs, 
                                                                    sample, 
                                                                    epoch_idx, 
                                                                    len(TestImgLoader.dataset),                                                                    
                                                                    **cfg.architecture) # [B, Ldim]

                    # if cfg.model.name not in ['semisup_gmvae', 'unsup_gmvae']:
                    #     y_hat = (loss_dict["KLD_Loss"].mean(1) > cfg.architecture.KLD_Loss_threshold) * 1
                    #     y_hat_proba = loss_dict["KLD_Loss"].mean(1) # temporaily store KLD_loss until entire set is processed
                    
                    if cfg.architecture.KLD_Loss_dims == "all" or cfg.architecture.KLD_Loss_dims == "all:":
                        KL_dims = []
                    else:
                        KL_dims = [int(n) for n in cfg.architecture.KLD_Loss_dims.split(":")[0].split(",")]
            
                    if cfg.model.name not in ['semisup_gmvae', 'unsup_gmvae']:
                        if KL_dims == []:
                            y_hat = (loss_dict["KLD_Loss"].mean(1) > cfg.architecture.KLD_Loss_threshold) * 1 # average all dims, then evaluate criterion
                        else:
                            y_hat = (loss_dict["KLD_Loss"][:,KL_dims].mean(1) > cfg.architecture.KLD_Loss_threshold) * 1
                        y_hat_proba = loss_dict["KLD_Loss"].mean(1) # TODO: temporaily store KLD_loss for now
                        
                        
                    # Logging
                    avg_test_scalars.update(scalar_outputs)
                    class_metrics.update(sample[1], y_hat, y_hat_proba)                  
    
                    if do_summary:
                        # Logger info  
                        image_outputs = {"recons_img": tensor2numpy(recons_img[0]),
                                        "input_img": tensor2numpy(input_img[0]),
                                        "dummy": tensor2numpy(input_img[0]),
                                        }   
                        
                        save_scalars(tb_logger, 'test', scalar_outputs, global_step)
                        save_images(tb_logger, 'test', image_outputs, global_step)
                        del scalar_outputs, image_outputs  
                        
                        print(
                        "Epoch:{}/{}, Eval iter:{}/{}, lr={:.2E}, loss={:.3f}, KL_loss={:.3f}, Recons_loss={:.3f}, time = {:.3f}".format(
                            epoch_idx+1, cfg.training.max_epochs, batch_idx, len(TestImgLoader),
                            optimizer.param_groups[0]["lr"], 
                            loss_dict["loss"].mean(),
                            loss_dict["KLD_Loss"].mean(),
                            loss_dict["Recons_Loss"].mean(),
                            time.time() - start_time)
                        )   
                        sys.stdout.flush()  
                        
                # Evaluate classification reports 
                class_metrics.report(display=True)
                class_metrics.confus_matrix(display=True) 
                
                # Evaluate classification metrics 
                # average: {‘binary’, ‘micro’, ‘macro’, ‘samples’, ‘weighted’}, default=None         
                A, P, R, F1 = class_metrics.APRF1_scores(average="weighted")
                avg_weighted_metrics = {"Acc_avg_weighted": A, "Prec_avg_weighted": P, "Rec_avg_weighted": R, "F1_avg_weighted": F1}
                A, P, R, F1 = class_metrics.APRF1_scores(average="binary")
                class_metrics_dict= {"Prec_anomaly": P, "Rec_anomaly": R, "F1_anomaly": F1}
                save_scalars(tb_logger, 'classification_metrics', {**class_metrics_dict, **avg_weighted_metrics}, global_step)
                    
                # Save average test results
                save_scalars(tb_logger, 'fulltest', avg_test_scalars.mean(), global_step)
                print("avg_test_scalars:", avg_test_scalars.mean())
                       
                
                # if cfg.model.name not in ['semisup_gmvae', 'unsup_gmvae']:
                #     class_metrics.y_pred_proba_from_KLDloss()
                
            gc.collect()


#################################################################################################################################


@make_nograd_func
def AD_analysis(model, TrainImgLoader, TestImgLoader, current_epoch, tb_logger, cfg):
    
    model.eval() 
    
    #### Read normal set
    ############################
    print("- Processing training set")
    Rec_loss_training_df = pd.DataFrame()  # [Ntrain, 1]
    KL_loss_training_df = pd.DataFrame()   # [Ntrain, Ldim]
    Other_loss_training_df = pd.DataFrame()   # [Ntrain, Ldim]    
    Mu_training_df = pd.DataFrame()        # [Ntrain, Ldim]
    Lvar_training_df = pd.DataFrame()       # [Ntrain, Ldim]
    True_labels_normal_df = pd.DataFrame() 
    for batch_idx, sample in enumerate(TrainImgLoader):   
        _, labels = sample        
        if batch_idx % cfg.logging.summary_freq == 0 :
            print("Evaluate NORMAL set - iter:{}/{}, ".format(batch_idx, len(TrainImgLoader) ))      
        # FORWARD PASS
        outputs = model(tocuda(sample))
        if cfg.model.name in ['semisup_gmvae', 'unsup_gmvae']:
            # outputs: [x_hat, x, out_train, out_infer] with out_train: dict("z", "zm","zv", "zm_prior", "zv_prior","qy_logit","qy","px")
            recons_img, input_img, mu, log_var = outputs[0], outputs[1], outputs[2]["zm"][cfg.postpro.cluster], outputs[2]["zv"][cfg.postpro.cluster]
        else:
            recons_img, input_img, mu, log_var = outputs[0], outputs[1], outputs[2], outputs[3]
        # LOSS EVALUATION                            
        loss_dict, scalar_outputs = model.loss_function(*outputs, 
                                                        tocuda(sample), 
                                                        current_epoch, 
                                                        len(TrainImgLoader.dataset),
                                                        **cfg.architecture) 
        # print (f"B_id: {batch_idx} - {torch.isnan(loss_dict['KLD_Loss']).any().item()}")
        # Append to DataFrames
        True_labels_normal_df = pd.concat([True_labels_normal_df, pd.DataFrame(labels)])
        Rec_loss_training_df = pd.concat([Rec_loss_training_df, pd.DataFrame(loss_dict["Recons_Loss"].cpu())])
        KL_loss_training_df = pd.concat([KL_loss_training_df, pd.DataFrame(loss_dict["KLD_Loss"].cpu())])
        Other_loss_training_df = pd.concat([Other_loss_training_df, pd.DataFrame(loss_dict["Other_Loss"].cpu())])
        # Other_loss_training_df = pd.concat([Other_loss_training_df, pd.DataFrame(loss_dict["MI_Loss"].cpu())])
        Mu_training_df = pd.concat([Mu_training_df, pd.DataFrame(mu.cpu())])
        Lvar_training_df = pd.concat([Lvar_training_df, pd.DataFrame(log_var.cpu())])

    # Check for NaNs
    if KL_loss_training_df.isnull().values.any():
        print("NaN detected")

    # Get quantile
    Rec_loss_quantile = Rec_loss_training_df.quantile(0.999, axis=0) # [float]
    KL_loss_quantile = KL_loss_training_df.quantile(0.999, axis=0) # [Ldim]
    Other_loss_quantile = Other_loss_training_df.quantile(0.999, axis=0) # [Ldim]  
    
    #### Read anomaly set
    ############################
    print("- Processing ANOMALY set")
    Rec_loss_anomaly_df = pd.DataFrame()  # [Nanomaly, 1]
    KL_loss_anomaly_df = pd.DataFrame()   # [Nanomaly, Ldim]
    Other_loss_anomaly_df = pd.DataFrame()   # [Nanomaly, Ldim]
    Mu_anomaly_df = pd.DataFrame()        # [Nanomaly, Ldim]
    Lvar_anomaly_df = pd.DataFrame()       # [Nanomaly, Ldim]    
    True_labels_anomaly_df = pd.DataFrame() 
    
    for batch_idx, sample in enumerate(TestImgLoader):   
        _, labels = sample    
        if batch_idx % cfg.logging.summary_freq == 0 :
            print("Evaluate Test set - iter:{}/{}, ".format(batch_idx, len(TestImgLoader) ))      
        # FORWARD PASS
        outputs = model(tocuda(sample))
        if cfg.model.name in ['semisup_gmvae', 'unsup_gmvae']:
            recons_img, input_img, mu, log_var = outputs[0], outputs[1], outputs[2]["zm"][cfg.postpro.cluster], outputs[2]["zv"][cfg.postpro.cluster]
        else:
            recons_img, input_img, mu, log_var = outputs[0], outputs[1], outputs[2], outputs[3]
        # LOSS EVALUATION        
        loss_dict, scalar_outputs = model.loss_function(*outputs, 
                                                        tocuda(sample), 
                                                        current_epoch, 
                                                        len(TestImgLoader.dataset),
                                                        **cfg.architecture) 
        # print (f"B_id: {batch_idx} - {torch.isnan(loss_dict['KLD_Loss']).any().item()}")
        # Append to DataFrames
        True_labels_anomaly_df = pd.concat([True_labels_anomaly_df, pd.DataFrame(labels)])
        Rec_loss_anomaly_df = pd.concat([Rec_loss_anomaly_df, pd.DataFrame(loss_dict["Recons_Loss"].cpu())])
        KL_loss_anomaly_df = pd.concat([KL_loss_anomaly_df, pd.DataFrame(loss_dict["KLD_Loss"].cpu())])
        Other_loss_anomaly_df = pd.concat([Other_loss_anomaly_df, pd.DataFrame(loss_dict["Other_Loss"].cpu())])
        # Other_loss_anomaly_df = pd.concat([Other_loss_anomaly_df, pd.DataFrame(loss_dict["MI_Loss"].cpu())])        
        Mu_anomaly_df = pd.concat([Mu_anomaly_df, pd.DataFrame(mu.cpu())])
        Lvar_anomaly_df = pd.concat([Lvar_anomaly_df, pd.DataFrame(log_var.cpu())])
        
    # Check for NaNs
    if KL_loss_anomaly_df.isnull().values.any():
        print("NaN detected")

    # Check negative values 
    # Replace zero value to avoid -inf when computing log loss (but loss should not be zero)
    if (KL_loss_training_df<=0.0).any(axis=None) or (KL_loss_anomaly_df<=0.0).any(axis=None):
        KL_loss_training_df[KL_loss_training_df <= 0.0] = 0.0001
        KL_loss_anomaly_df[KL_loss_anomaly_df <= 0.0] = 0.0001
    if (Other_loss_training_df<=0.0).any(axis=None) or (Other_loss_anomaly_df<=0.0).any(axis=None):
        Other_loss_training_df[Other_loss_training_df <= 0.0] = 0.0001
        Other_loss_anomaly_df[Other_loss_anomaly_df <= 0.0] = 0.0001
    
    
    #### Evaluate latent space variables statistics
    ###################################################
    print("- Evaluating latent space variables statistics")
    # Mu 
    Mu_training_mean = Mu_training_df.mean()
    Mu_anomaly_mean = Mu_anomaly_df.mean()
    Mu_diff = Mu_training_mean - Mu_anomaly_mean
    # Mu dimension selection
    Mu_dims_threshold = Mu_diff.mean() + 1 * Mu_diff.std() 
    Mu_dims_selection = Mu_diff[Mu_diff.abs() > Mu_dims_threshold].index 
    with open(os.path.join(os.path.dirname(cfg.model.loadckpt), "Mu_dims_selection.pkl"),'wb') as f: pickle.dump(Mu_dims_selection, f)
    # Mu threshold
    Mu_threshold = Mu_anomaly_mean.abs() + Mu_diff.abs()
    Mu_threshold.to_pickle(os.path.join(os.path.dirname(cfg.model.loadckpt), "Mu_threshold.pkl"))
    # label selection
    # Pred_labels_Mu = (Mu_anomaly_df[Mu_dims_selection].abs() > Mu_threshold[Mu_dims_selection].abs()).all(axis=1) * 1   
    #
    # debug section    
    # plt.figure()
    # Mu_training_mean.abs().plot(label="Mu_training_mean abs")
    # Mu_anomaly_mean.abs().plot(label="Mu_anomaly_mean abs")
    # Mu_threshold.plot(label="Mu treshold")
    # Mu_anomaly_df.abs().iloc[0].plot(label="Mu Anomaly 0")
    # plt.grid(visible=True, which='both', axis='both', mouseover=True)
    # plt.legend()

    # LogVar
    Lvar_training_mean = Lvar_training_df.mean()    
    Lvar_anomaly_mean = Lvar_anomaly_df.mean() 
    Lvar_diff = Lvar_training_mean - Lvar_anomaly_mean
    # Mu dimension selection
    Lvar_dims_threshold = Lvar_diff.mean() + 1 * Lvar_diff.std() 
    Lvar_dims_selection = Lvar_diff[Lvar_diff.abs() > Lvar_dims_threshold].index 
    with open(os.path.join(os.path.dirname(cfg.model.loadckpt), "Lvar_dims_selection.pkl"),'wb') as f: pickle.dump(Lvar_dims_selection, f)
    # Mu threshold
    Lvar_threshold = Lvar_anomaly_mean - Lvar_diff.abs()
    Lvar_threshold.to_pickle(os.path.join(os.path.dirname(cfg.model.loadckpt), "Lvar_threshold.pkl"))
        
    # debug section
    # plt.figure()
    # Lvar_training_mean.plot(label="Lvar_training_mean")
    # Lvar_anomaly_mean.plot(label="Lvar_anomaly_mean")
    # Lvar_threshold.plot(label="Lvar treshold")
    # Lvar_anomaly_df.iloc[0].plot(label="Lvar Anomaly 0")
    # plt.grid(visible=True, which='both', axis='both', mouseover=True)
    # plt.legend()
        

    # Display Latent space variables
    ####################################
    if cfg.debug.latent_boxplots:
        print("- Plotting box plots of latent variables")        

        # Mean
        Max = math.ceil(max(Mu_training_df.values.max(), Mu_anomaly_df.values.max()))
        Min = math.floor(min(Mu_training_df.values.min(), Mu_anomaly_df.values.min()))
        
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
        fig.tight_layout()
        Mu_training_df.boxplot(ax=axes[0])
        axes[0].title.set_text('latent variables Mu - Training')
        axes[0].set_ylim(Min, Max)
        Mu_anomaly_df.boxplot(ax=axes[1])
        axes[1].title.set_text('latent variables Mu - Anomaly')
        axes[1].set_ylim(Min, Max)
        plt.xlabel("Latent space dimension")
        for ax in axes:
            # ax.minorticks_on()
            ax.grid(visible=True, which='both', axis='both', mouseover=True)

        # LogVar
        Max = math.ceil(max(Lvar_training_df.values.max(), Lvar_anomaly_df.values.max()))
        Min = math.floor(min(Lvar_training_df.values.min(), Lvar_anomaly_df.values.min()))
        
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
        fig.tight_layout()
        Lvar_training_df.boxplot(ax=axes[0])
        axes[0].title.set_text('latent variables logVar - Training')
        axes[0].set_ylim(Min, Max)
        Lvar_anomaly_df.boxplot(ax=axes[1])
        axes[1].title.set_text('latent variables logVar - Anomaly')
        axes[1].set_ylim(Min, Max)
        plt.xlabel("Latent space dimension")
        for ax in axes:
            # ax.minorticks_on()
            ax.grid(visible=True, which='both', axis='both', mouseover=True)
  
        # Average difference normal/anomaly        
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
        fig.tight_layout()
        axes[0].title.set_text('latent variables Mu - Average difference normal/anomaly')
        axes[0].plot(Mu_diff)
        axes[1].title.set_text('latent variables LogVar - Average difference normal/anomaly')
        axes[1].plot(Lvar_diff)
        plt.xlabel("Latent space dimension")        
        for ax in axes:
            # ax.minorticks_on()
            ax.grid(visible=True, which='both', axis='both', mouseover=True)
            
              
    # Display loss histograms 
    ##########################
    if cfg.debug.hist_plots:        
        print("- Plotting histograms of losses")
           
        #### Recons + KLD + Other loss (no logscale)        
        # plt.figure("reconstruction loss ")
        # sns.histplot(data=Rec_loss_training_df,  bins=64, color="blue", stat = "probability", label="Normal set", kde=True, log_scale=False)
        # sns.histplot(data=Rec_loss_anomaly_df,  bins=64, palette='husl', color="orange", stat = "probability", label="Anomaly set", kde=True, log_scale=False)
        # plt.grid()
        # plt.legend(title="Recons loss") 
        
        # plt.figure("KL div loss ")
        # sns.histplot(data=KL_loss_training_df.mean(axis=1),  bins=64, color="blue", stat = "probability", label="Normal set", kde=True, log_scale=False)
        # sns.histplot(data=KL_loss_anomaly_df.mean(axis=1),  bins=64, color="orange", stat = "probability", label="Anomaly set", kde=True, log_scale=False)
        # plt.grid()
        # plt.legend(title="KLD loss (logscale and avgd over latent dims)") 
        
        #### Recons + KLD + Other loss with logscale  
            
        plt.figure("reconstruction loss  (logScale)")
        sns.histplot(data=Rec_loss_training_df,  bins=64, color="blue", stat = "probability", label="Normal set", kde=True, log_scale=True)
        sns.histplot(data=Rec_loss_anomaly_df,  bins=64, palette='husl', color="orange", stat = "probability", label="Anomaly set", kde=True, log_scale=True)
        plt.grid()
        plt.legend(title="Recons loss (logscale)") 
        
        plt.figure("KL div loss (logScale)")
        sns.histplot(data=KL_loss_training_df.mean(axis=1),  bins=64, color="blue", stat = "probability", label="Normal set", kde=True, log_scale=True)
        sns.histplot(data=KL_loss_anomaly_df.mean(axis=1),  bins=64, color="orange", stat = "probability", label="Anomaly set", kde=True, log_scale=True)
        plt.grid()
        plt.legend(title="KLD loss (logscale and avgd over latent dims)")   

        N, A = Other_loss_training_df.mean(axis=1), Other_loss_anomaly_df.mean(axis=1)
        plt.figure("TC loss (logScale)")
        sns.histplot(data=N,  bins=64, color="blue", stat = "probability", label="Normal set", alpha=0.4)
        sns.histplot(data=A,  bins=64, color="orange", stat = "probability", label="Anomaly set", alpha=0.4)
        # May need to set limits manually
        # sns.histplot(data=N,  bins=np.linspace(200,400, 64), color="blue", stat = "probability", label="Normal set", alpha=0.4)  
        # sns.histplot(data=A,  bins=np.linspace(200,400, 64), color="orange", stat = "probability", label="Anomaly set", alpha=0.4)
        plt.xscale("log")
        plt.grid()
        plt.legend(title="TC loss (logscale and avgd over latent dims)") 
        
        
        #### SVDD loss (optional)        
        if cfg.model.name == "svdd_vae" and cfg.architecture.svdd_flag == True:
            plt.figure("svdd div loss (logScale)")
            sns.histplot(data=Other_loss_training_df.mean(axis=1),  bins=64, color="blue", stat = "probability", label="Normal set", kde=True, log_scale=True)
            sns.histplot(data=Other_loss_anomaly_df.mean(axis=1),  bins=64, color="orange", stat = "probability", label="Anomaly set", kde=True, log_scale=True)
            plt.grid()
            
            plt.legend(title="SVDD loss (logscale and avgd over latent dims)") 
        
        plt.show()
        
        
    #### Helper functions
    ##########################
        
    # Define dimension-wise loss histogram plots (KLD loss)
    def plot_KLD_distrib(dim, equal_mean=False, show=True):
        eps = -6.0
        T = np.log(KL_loss_training_df[dim])
        A = np.log(KL_loss_anomaly_df[dim])
        T, A = T[T >eps], A[A >eps]
        if equal_mean: A -= A.mean() - T.mean() # experimental 
        plt.figure(f"KL div loss (LogScale) - Latent space dim: {dim}")
        sns.histplot(data=T, bins=64, color="blue", stat = "probability", label="Normal set", kde=True, log_scale=False)
        sns.histplot(data=A, bins=64,  color="orange", stat = "probability", label="Anomaly set", kde=True, log_scale=False)
        plt.grid(visible=True, which='both', axis='both', mouseover=True)
        plt.legend(title=f"KLD loss (dim={dim})") 
        if show: plt.show()

    # Define dimension-wise loss histogram plots (SVDD loss)
    def plot_Other_loss_distrib(dim, equal_mean=False, show=True):
        T = np.log(Other_loss_training_df[dim])
        A = np.log(Other_loss_anomaly_df[dim])
        if equal_mean: A -= A.mean() - T.mean() # experimental 
        plt.figure(f"SVDD loss (LogScale) - Latent space dim: {dim}")
        sns.histplot(data=T, bins=64, color="blue", stat = "probability", label="Normal set", kde=True, log_scale=False)
        sns.histplot(data=A, bins=64,  color="orange", stat = "probability", label="Anomaly set", kde=True, log_scale=False)
        plt.grid(visible=True, which='both', axis='both', mouseover=True)
        plt.legend(title=f"SVDD loss (dim={dim})") 
        if show: plt.show()
         
    # Define dimension-wise loss histogram plots (CDF of KLD loss)
    def plot_CDFs(dim, equal_mean=False, show=True):
        T = np.log(KL_loss_training_df[dim])
        A = np.log(KL_loss_anomaly_df[dim])
        plt.figure(f"KLD loss CDF (LogScale) - Latent space dim: {dim}")
        if equal_mean: A -= A.mean() - T.mean() # experimental 
        sns.ecdfplot(T, color="blue", label="Normal set") #, log_scale=True)
        sns.ecdfplot(A, color="orange", label="Anomaly set") #, log_scale=True) 
        plt.legend(title=f"KLD loss (dim={dim})") 
        plt.grid(visible=True, which='both', axis='both', mouseover=True)
        if show: plt.show()
    
    
    #### Evaluate dimension-wise Loss statistics
    ###############################################
    if cfg.debug.metrics_plots:       
        loss_fcn_training = KL_loss_training_df # Other_loss_training_df # 
        loss_fcn_anomaly = KL_loss_anomaly_df # Other_loss_anomaly_df # KL_loss_anomaly_df
        loss_fcn_name = "KL loss" # "SVDD loss" # 
        
        KS = []
        ttest = []
        chi2 = []
        nrg = []
        kld = []
        for dim in range(loss_fcn_training.shape[1]):
            
            # get log of loss distribution
            T = np.log(loss_fcn_training[dim])
            A = np.log(loss_fcn_anomaly[dim])
            delta_mean = A.mean() - T.mean()
            A_zero_mean = A - delta_mean

            # Evaluate min/max of loss distribution 
            minrange = math.floor( np.min([T.min(), A.min()]) *100)/100
            maxrange = math.floor( np.max([T.max(), A.max()]) *100)/100

            # Get KLD loss distribution as a probability
            f_exp, bin_edges = np.histogram(T, bins=256, range=(minrange, maxrange), density=True, weights=None)
            f_obs, bin_edges = np.histogram(A, bins=256, range=(minrange, maxrange), density=True, weights=None)
            f_obs_zero_mean, bin_edges = np.histogram(A_zero_mean, bins=256, range=(minrange, maxrange), density=True, weights=None)
            
            # mask zero expected values to evaluate chi² 
            expMasked = np.ma.masked_where(f_exp == 0.0, f_exp)
            f_obs_masked = np.array(f_obs[~expMasked.mask])
            f_exp_masked = np.array(f_exp[~expMasked.mask])
            f_obs_zero_mean_masked = np.array(f_obs_zero_mean[~expMasked.mask])
                            
            # Correct for mismatch between the obs & exp distribs (should both integrate to 1)
            f_obs_masked *= f_exp_masked.mean() / f_obs_masked.mean()
            f_obs_zero_mean_masked*= f_exp_masked.mean() / f_obs_zero_mean_masked.mean()
                
            # register KS statistic
            KS.append(scipy.stats.kstest(T, A_zero_mean, alternative='two-sided')[0])       

            # Eval chi square statistic
            # chi2.append(scipy.stats.chisquare(f_obs_zero_mean_masked, f_exp_masked, ddof=0, axis=0)[0])      
            chi2.append(scipy.stats.chisquare(f_obs_masked, f_exp_masked, ddof=0, axis=0)[0])        
            # Entropy fcn
            nrg.append(scipy.stats.entropy(f_obs_zero_mean_masked, f_exp_masked))
            # Sklearn
            kld.append(sklearn.metrics.mutual_info_score(f_obs_zero_mean, f_exp))
            # Eval t-test statistic (with substracted mean difference)
            ttest.append(scipy.stats.ttest_ind(f_obs_zero_mean_masked, f_exp, equal_var=False, alternative='two-sided')[0])

        # Detect "abnormal" latent dimensions using Chi square
        chi2_df = pd.DataFrame(chi2)
        chi2_threshold = chi2_df.mean().item() + 1.5 * chi2_df.std().item()
        abnormal_chi2_list = chi2_df.index[chi2_df[0] > chi2_threshold].to_list()
        print(f"Abnormal dimensions detected (threshold={chi2_threshold:.1f}): {abnormal_chi2_list}")
        
        # plot all statistics
        if cfg.model.name == "svdd_vae" and cfg.architecture.svdd_flag == True: 
            Nrows = 7
        else:
            Nrows = 6
        fig, axes = plt.subplots(nrows=Nrows, ncols=1, sharex=True)
        fig.tight_layout()
        fig.suptitle(f"{loss_fcn_name} statistics ({cfg.model.loadckpt.split('/')[-2]})")
        axes[0].plot(Mu_diff, label="Mu avg diff. normal/anomaly")
        axes[1].plot(Lvar_diff, label="LogVar avg diff. normal/anomaly")
        axes[2].plot(chi2, label="chi^2")  
        axes[2].axhline(y = chi2_threshold, color = 'r', linestyle = '--') 
        axes[3].plot(ttest, label="2 samples t-test")
        axes[4].plot(KS, label="K-S")
        axes[5].plot(nrg, label="Entropy")
        if cfg.model.name == "svdd_vae" and cfg.architecture.svdd_flag == True:
            axes[6].plot(model.get_svdd_R().detach().cpu().numpy(), label="SVDD R")

        for ax in axes:
            ax.minorticks_on()
            ax.grid(visible=True, which='both', axis='both', mouseover=True)
            ax.legend()
        plt.xlabel("Latent space dimension")
        plt.subplots_adjust(wspace=0, hspace=0)

        # Plot distributions for abnormal dimentions 
        for d in abnormal_chi2_list:
            if "KL" in loss_fcn_name:
                plot_KLD_distrib(d, equal_mean=False, show=False)
            else:
                plot_Other_loss_distrib(d, equal_mean=False, show=False)
                
        plt.show()


        # Reconstruct distributions using most important latent dimensions
        chi2_sort_order = sorted(range(len(chi2)), key=lambda i: chi2[i], reverse=False)
        L = cfg.architecture.latent_dim

        N = len(abnormal_chi2_list)
        top_dims = chi2_sort_order[:N]
        plt.figure(f"{loss_fcn_name} LogScale - Latent space dim: {top_dims}")
        sns.histplot(data=loss_fcn_training.mean(axis=1), bins=64, color="green", stat = "probability", label=f"Training set (all dims)", kde=True, log_scale=True)
        sns.histplot(data=loss_fcn_training[top_dims].mean(axis=1), bins=64, color="blue", stat = "probability", label="Normal set", kde=True, log_scale=True)
        sns.histplot(data=loss_fcn_anomaly[top_dims].mean(axis=1), bins=64, color="orange", stat = "probability", label="Anomaly set", kde=True, log_scale=True)
        plt.grid(visible=True, which='both', axis='both', mouseover=True)
        plt.legend(title=f"{loss_fcn_name} - {N} lowest chi2 contribution dims") 

        top_dims = chi2_sort_order[(L-N)//2:(L+N)//2]
        plt.figure(f"{loss_fcn_name} loss LogScale - Latent space dim: {top_dims}")
        sns.histplot(data=loss_fcn_training.mean(axis=1), bins=64, color="green", stat = "probability", label=f"Training set (all dims)", kde=True, log_scale=True)
        sns.histplot(data=loss_fcn_training[top_dims].mean(axis=1), bins=64, color="blue", stat = "probability", label="Normal set", kde=True, log_scale=True)
        sns.histplot(data=loss_fcn_anomaly[top_dims].mean(axis=1), bins=64, color="orange", stat = "probability", label="Anomaly set", kde=True, log_scale=True)
        plt.grid(visible=True, which='both', axis='both', mouseover=True)
        plt.legend(title=f"{loss_fcn_name} - {N} centered chi2 contribution dims") 

        top_dims = chi2_sort_order[-N:]
        plt.figure(f"{loss_fcn_name} LogScale - Latent space dim: {top_dims}")
        sns.histplot(data=loss_fcn_training.mean(axis=1), bins=64, color="green", stat = "probability", label=f"Training set (all dims)", kde=True, log_scale=True)
        sns.histplot(data=loss_fcn_training[top_dims].mean(axis=1), bins=64, color="blue", stat = "probability", label="Normal set", kde=True, log_scale=True)
        sns.histplot(data=loss_fcn_anomaly[top_dims].mean(axis=1), bins=64, color="orange", stat = "probability", label="Anomaly set", kde=True, log_scale=True)
        plt.grid(visible=True, which='both', axis='both', mouseover=True)
        plt.legend(title=f"{loss_fcn_name} - {N} highest chi2 contribution dims") 

        top_dims = chi2_sort_order[:-N]
        plt.figure(f"{loss_fcn_name} LogScale - Latent space dim: {top_dims}")
        sns.histplot(data=loss_fcn_training.mean(axis=1), bins=64, color="green", stat = "probability", label=f"Training set (all dims)", kde=True, log_scale=True)
        sns.histplot(data=loss_fcn_training[top_dims].mean(axis=1), bins=64, color="blue", stat = "probability", label="Normal set", kde=True, log_scale=True)
        sns.histplot(data=loss_fcn_anomaly[top_dims].mean(axis=1), bins=64, color="orange", stat = "probability", label="Anomaly set", kde=True, log_scale=True)
        plt.grid(visible=True, which='both', axis='both', mouseover=True)
        plt.legend(title=f"{loss_fcn_name} - All dims except {N} highest chi2 contribution dims") 

        plt.show()
    
    #### Plot tSNE
    #################
    if cfg.debug.tSNE_plots:
        # Display t-SNE of mu   
        print("- Plotting t-SNE of latent space Mu (this may take a while)")
        tsne = TSNE(n_components=2, verbose=1, random_state=123)
        tsne_mu_train = tsne.fit_transform(Mu_training_df.to_numpy()[np.random.choice(len(Mu_training_df), len(Mu_anomaly_df), replace=False)])
        tsne_mu_anomaly = tsne.fit_transform(Mu_anomaly_df.to_numpy())

        tsne_mu_train_df = pd.DataFrame(tsne_mu_train)
        tsne_mu_train_df.columns = ["comp-1", "comp-2"] 
        tsne_mu_anomaly_df = pd.DataFrame(tsne_mu_anomaly)
        tsne_mu_anomaly_df.columns = ["comp-1", "comp-2"] 

        plt.figure("t-SNE Mean KDE")
        # sns.scatterplot(data=tsne_mu_train_df, x="comp-1", y="comp-2", color="blue", label="Normal set")
        # sns.scatterplot(data=tsne_mu_anomaly_df, x="comp-1", y="comp-2", color="red", label="Anomaly set")
        sns.kdeplot(data=tsne_mu_train_df, x="comp-1", y="comp-2", levels=10, fill=True, color="blue", label="Normal set")
        sns.kdeplot(data=tsne_mu_anomaly_df, x="comp-1", y="comp-2", levels=10, fill=False, color="red", label="Anomaly set")
        plt.title("t-SNE of latent space variable: Mu")
        plt.legend()
        plt.grid(visible=True, which='both', axis='both', mouseover=True)
        # tb_logger.add_figure("anomaly/t-SNE_Mu", plt.gcf())

        # Display t-SNE of LogVAR
        print("- Plotting t-SNE of latent space logVar")
        tsne = TSNE(n_components=2, verbose=1, random_state=123)
        tsne_Lvar_train = tsne.fit_transform(Lvar_training_df.to_numpy()[np.random.choice(len(Lvar_training_df), len(Lvar_anomaly_df), replace=False)])
        tsne_Lvar_anomaly = tsne.fit_transform(Lvar_anomaly_df.to_numpy())

        tsne_Lvar_train_df = pd.DataFrame(tsne_Lvar_train)
        tsne_Lvar_train_df.columns = ["comp-1", "comp-2"] 
        tsne_Lvar_anomaly_df = pd.DataFrame(tsne_Lvar_anomaly)
        tsne_Lvar_anomaly_df.columns = ["comp-1", "comp-2"] 

        plt.figure("t-SNE logVar KDE")
        # sns.scatterplot(data=tsne_mu_train_df, x="comp-1", y="comp-2", color="blue", label="Normal set")
        # sns.scatterplot(data=tsne_mu_anomaly_df, x="comp-1", y="comp-2", color="red", label="Anomaly set")
        sns.kdeplot(data=tsne_Lvar_train_df, x="comp-1", y="comp-2", levels=10, fill=True, color="blue", label="Normal set")
        sns.kdeplot(data=tsne_Lvar_anomaly_df, x="comp-1", y="comp-2", levels=10, fill=False, color="red", label="Anomaly set")
        plt.title("t-SNE of latent space variable: log_Var")
        plt.legend()
        plt.grid(visible=True, which='both', axis='both', mouseover=True)
        plt.show()
        # tb_logger.add_figure("anomaly/t-SNE_LogVar", plt.gcf())
        
    print("ready to close?")


###############################################################################################################################


@make_nograd_func
def anomaly_classification(model, TestImgLoader, current_epoch, tb_logger, cfg):

    model.eval()
    with torch.no_grad():
        class_metrics = ClassificationMetrics()
        epoch_idx = 0        
        True_labels_df = pd.DataFrame() 
        Mu_anomaly_df = pd.DataFrame() 
        Lvar_anomaly_df = pd.DataFrame() 
        
        for batch_idx, sample in enumerate(TestImgLoader):            
            # Init
            _, labels = sample
            True_labels_df = pd.concat([True_labels_df, pd.DataFrame(labels)])
            sample = tocuda(sample)
            start_time = time.time()
            do_summary = batch_idx % cfg.logging.summary_freq == 0                    
            
            # FORWARD PASS
            outputs = model(sample)
            recons_img, input_img, mu, Lvar = outputs[0], outputs[1], outputs[2], outputs[3]
            
            Mu_anomaly_df = pd.concat([Mu_anomaly_df, pd.DataFrame(mu.cpu())])
            Lvar_anomaly_df = pd.concat([Lvar_anomaly_df, pd.DataFrame(Lvar.cpu())])
            
            if cfg.model.name in ['semisup_gmvae', 'unsup_gmvae']:
                y_hat, y_hat_proba = outputs[3]["y_hat"], outputs[3]["qy"].max(1)[0]

            # LOSS EVALUATION                                                      
            loss_dict, scalar_outputs = model.loss_function(*outputs, 
                                                            sample, 
                                                            epoch_idx, 
                                                            len(TestImgLoader.dataset),                                                                    
                                                            **cfg.architecture) # [B, Ldim]

            # Label prediction based on KLD loss threshold
            if cfg.architecture.KLD_Loss_dims == "all" or cfg.architecture.KLD_Loss_dims == "all:":
                KL_dims = []
            else:
                KL_dims = [int(n) for n in cfg.architecture.KLD_Loss_dims.split(":")[0].split(",")]
            
            if cfg.model.name not in ['semisup_gmvae', 'unsup_gmvae']:
                if KL_dims == []:
                    y_hat = (loss_dict["KLD_Loss"].mean(1) > cfg.architecture.KLD_Loss_threshold) * 1 # average all dims, then evaluate criterion
                else:
                    y_hat = (loss_dict["KLD_Loss"][:,KL_dims].mean(1) > cfg.architecture.KLD_Loss_threshold) * 1
                y_hat_proba = loss_dict["KLD_Loss"].mean(1) # TODO: temporaily store KLD_loss for now
            class_metrics.update(true_labels=sample[1], pred_labels=y_hat, pred_proba=y_hat_proba) 

            if do_summary:
                print(
                "Eval iter:{}/{}, loss={:.3f}, KL_loss={:.3f}, Recons_loss={:.3f}, time = {:.3f}".format(
                    batch_idx, len(TestImgLoader),
                    loss_dict["loss"].mean(),
                    loss_dict["KLD_Loss"].mean(),
                    loss_dict["Recons_Loss"].mean(),
                    time.time() - start_time)
                )   
                sys.stdout.flush()  
        
        # Evaluate classification 
        print("=== CLASSIFICATION from KLDloss threshold")
        A, P, R, F1 = class_metrics.APRF1_scores()
        class_report_KLDloss = class_metrics.report(display=True)
        conf_matrix_KLDloss = class_metrics.confus_matrix(display=True) 
        
        print("=== CLASSIFICATION from latent variables")
        # True_labels_df
        
        # Read latent variable thresholds and dimensions
        Mu_dims_selection = pd.read_pickle(os.path.join(os.path.dirname(cfg.model.loadckpt), "Mu_dims_selection.pkl"))
        Mu_threshold = pd.read_pickle(os.path.join(os.path.dirname(cfg.model.loadckpt), "Mu_threshold.pkl"))
        Lvar_dims_selection = pd.read_pickle(os.path.join(os.path.dirname(cfg.model.loadckpt), "Lvar_dims_selection.pkl"))        
        Lvar_threshold = pd.read_pickle(os.path.join(os.path.dirname(cfg.model.loadckpt), "Lvar_threshold.pkl"))
        
        # Evaluate predictios for each latent space variable
        Pred_labels_Mu = (Mu_anomaly_df[Mu_dims_selection].abs() > Mu_threshold[Mu_dims_selection].abs()).all(axis=1) * 1   
        Pred_labels_Lvar = (Lvar_anomaly_df[Lvar_dims_selection] < Lvar_threshold[Lvar_dims_selection]).all(axis=1) * 1   
        
        # classification reports 
        class_report_Mu = classification_report(True_labels_df, Pred_labels_Mu)
        conf_matrix_Mu = confusion_matrix(True_labels_df, Pred_labels_Mu)
        print("==> Classification report for Mu:\n",class_report_Mu)
        print("==> confusion matrix for Mu:\n",conf_matrix_Mu)
                        
        class_report_Lvar = classification_report(True_labels_df, Pred_labels_Lvar)
        conf_matrix_Lvar = confusion_matrix(True_labels_df, Pred_labels_Lvar)
        print("==> Classification report for Lvar:\n",class_report_Lvar)
        print("==> confusion matrix for Mu:\n",conf_matrix_Lvar)
        
        # Write to file
        with open(os.path.join(os.path.dirname(cfg.model.loadckpt), "classification.txt"), 'w') as f:            
            f.write("=== CLASSIFICATION from KLDloss threshold ===\n\n")
            f.write("==> Classification report for KLDloss:\n")
            f.write("KLD loss threshold ="+str(cfg.architecture.KLD_Loss_threshold)+"\n")
            f.write("KLD loss dimensions ="+str(KL_dims)+"\n") 
            f.write(class_report_KLDloss)  
            f.write("\n")   
            f.write(str(conf_matrix_KLDloss)) 
            f.write("\n")   
            
            f.write("\n")   
            f.write("=== CLASSIFICATION from latent variables ===\n\n")
            f.write("==> Classification report for Mu:\n")  
            f.write("Selected dims: "+str(Mu_dims_selection)+"\n")
            f.write(class_report_Mu)          
            f.write(str(conf_matrix_Mu)) 
            f.write("\n")   
            
            f.write("\n")   
            f.write("==> Classification report for Lvar:\n")   
            f.write("Selected dims: "+str(Lvar_dims_selection)+"\n")
            f.write(class_report_Lvar)  
            f.write(str(conf_matrix_Lvar))
    
    gc.collect()
    
    
    
    
    
    
    
    
    
    
###############################################################################################################################
###############################################################################################################################

if __name__ == '__main__':

    ###### INFO: Check and print code version
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    branch = repo.active_branch
    print("Code version: ")
    print(f"  git hash: {sha}")
    print(f"  git hash: {branch.name}")

    ###### parse arguments and check    
    args = parser.parse_args()
    print(args.opts)
    with open(args.filename, "r") as file:
        cfg = load_cfg(file)   
    cfg.merge_from_list(args.opts) 
    print(cfg)    
    cfg.freeze()
           
    ###### CUDA init
    if cfg.model.device == None:    
        cfg.model.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ###### SEED
    if cfg.model.seed == 0:
        set_seed(random.randint(1,99999999))
    else:
        set_seed(cfg.model.seed)


    ##### BUILD DATASETs
    if cfg.dataset.dataset_name == "HSC-SSP_DR4_james":
        list_path = os.path.join("lists", cfg.dataset.dataset_name, "testtrain_split"+str(cfg.dataset.testtrain_split))
        create_candidate_training_lists(list_path, cfg)
        
        if cfg.model.mode == "train":        
            list_name_train = "train.txt"
            list_name_val = "val.txt"            
        else:
            list_name_train = "normal_set.txt"
            list_name_val = "anomaly_set.txt"     
            # list_name_val = "anomaly_set_132.txt"  
        list_name_test = "test.txt"
        # list_name_test = "val.txt"
        
        if cfg.model.mode == "train" and cfg.dataset.fixinbalance:             
            normal_label_qty, anomaly_label_qty = 90469, 2067 # taken from the train.txt file
            sample_weights = [1/normal_label_qty] * normal_label_qty + [1/anomaly_label_qty] * anomaly_label_qty # OBS! assumes normal data fist in train.txt file
            
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            shuffle_data = None
        else:
            sampler = None
            shuffle_data = True
            
        train_dataset = HSCSSP_Dataset(list_path=list_path, list_names = list_name_train,**cfg.dataset)
        val_dataset = HSCSSP_Dataset(list_path=list_path, list_names = list_name_val, **cfg.dataset)
        test_dataset = HSCSSP_Dataset(list_path=list_path, list_names = list_name_test, **cfg.dataset)
        
    elif cfg.dataset.dataset_name == "CIFAR10":
        # labels: 32x32, 10labels: 0-airplane 1-automobile 2-bird 3-cat 4-deer 5-dog	6-frog 7-horse 8-ship 9-truck
        transfo = Compose([T.Grayscale(), T.ToTensor(),  T.Resize((cfg.dataset.img_resize,cfg.dataset.img_resize), antialias=True)])        
        train_dataset = CIFAR10(root="./data/CIFAR10", train=True, transform=transfo, download=True)
        val_dataset = CIFAR10(root="./data/CIFAR10", train=False, transform=transfo, download=True)
        test_dataset = val_dataset
        
    elif cfg.dataset.dataset_name == "CelebA":
        # labels: 203k img, targets=["attr", "identity", "landmark", "bbox"] with 10,177 identities, 40 binary attributes 
        # https://drive.google.com/drive/folders/0B7EVK8r0v71pOC0wOVZlQnFfaGs?resourcekey=0-pEjrQoTrlbjZJO2UL8K_WQ
        # Ex: 5_o_Clock_Shadow(0) Arched_Eyebrows(1) Attractive(2) ... Bald(4) ... Big_Nose(7) ... Eyeglasses(15) ...  Male(20) ...  Smiling(31) 
        transfo = Compose([T.Grayscale(), T.ToTensor(),  T.Resize((cfg.dataset.img_resize,cfg.dataset.img_resize), antialias=True)])          
        tgt_transfo = Compose([T.Lambda(lambda x: x[cfg.dataset.celeba_attr])])  
        train_dataset = CelebA(root="./data/CelebA", split="train", target_type="attr", transform=transfo, target_transform=tgt_transfo, download=True)
        val_dataset = CelebA(root="./data/CelebA", split="valid", target_type="attr", transform=transfo, target_transform=tgt_transfo, download=True)
        test_dataset = CelebA(root="./data/CelebA", split="test", target_type="attr", transform=transfo, target_transform=tgt_transfo, download=True)
        
    elif cfg.dataset.dataset_name == "STL10":
        # labels: 96x96pxls 0-airplane, 1-bird, 2-car, 3-cat, 4-deer, 5-dog, 6-horse, 7-monkey, 8-ship, 9-truck.
        # 
        transfo = Compose([T.Grayscale(), T.ToTensor(),  T.Resize((cfg.dataset.img_resize,cfg.dataset.img_resize), antialias=True)])  
        train_dataset = STL10(root="./data/STL10", split="train", transform=transfo, download=True)
        val_dataset = STL10(root="./data/STL10", split="test", transform=transfo, download=True)
        test_dataset = val_dataset
        
    elif cfg.dataset.dataset_name == "FashionMNIST":
        # labels: 28x28 grayscale image, 10 classes: 0-T-shirt/top, 1-Trouser, 2-Pullover, 3-Dress, 4-Coat, 5-Sandal, 6-Shirt, 7-Sneaker, 8-Bag, 9-Ankle boot
        transfo = Compose([T.ToTensor(),  T.Resize((cfg.dataset.img_resize,cfg.dataset.img_resize), antialias=True)])  
        train_dataset = FashionMNIST(root="./data/FashionMNIST", train=True, transform=transfo, download=True)
        val_dataset = FashionMNIST(root="./data/FashionMNIST", train=False, transform=transfo, download=True)
        test_dataset = val_dataset       
        
    ##### BUILD DATALOADERs
    TrainImgLoader = DataLoader(train_dataset, cfg.training.batch_size, shuffle=shuffle_data,  sampler=sampler, 
                                num_workers=cfg.dataset.workers, drop_last=True, pin_memory=cfg.dataset.pin_m)
    
    ValImgLoader = DataLoader(val_dataset, cfg.training.batch_size, shuffle=False, 
                              num_workers=cfg.dataset.workers, drop_last=False, pin_memory=cfg.dataset.pin_m)
    
    TestImgLoader = DataLoader(test_dataset, cfg.training.batch_size, shuffle=False, 
                              num_workers=cfg.dataset.workers, drop_last=False, pin_memory=cfg.dataset.pin_m)

    ###### Get dataloader input size 
    img_sample = next(iter(TrainImgLoader))[0]
    _, Cin, Hin, Win = img_sample.shape 
    print(f"Input size: Cin={Cin} H={Hin} W={Hin}")
    
    ##### BUILD MODEL
    model = models_classnames[cfg.model.name](img_res=(Hin,Win), device=cfg.model.device, debug=cfg.debug.debug, **cfg.architecture)
    model.to(cfg.model.device)
    
    # Print model summary based on input shape
    # print(Msummary(model, input_size=(1, Cin, Hin, Win))) # slow - comment out if not required
    
    ##### BUILD LOGGER 
    # TODO: replace with wandb
    if not os.path.isdir(cfg.logging.logdir):
        os.makedirs(cfg.logging.logdir)
    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    print("current time", current_time_str)
    print(f"creating new summary file: {cfg.logging.logdir}")
    tb_logger = SummaryWriter(cfg.logging.logdir, flush_secs = 2)
    # tb_logger.add_graph(model, img_sample.repeat(1,3,1,1))
    tb_logger.flush()
    print("argv:", sys.argv[1:])
    print(cfg)    
    
    ###### Build optimizer
    print("Building Optimizer.")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=cfg.training.lr, 
                           betas=(0.9, 0.999), 
                           weight_decay=cfg.training.wd)

    ###### load model parameters for resume or checkpoint start (optional)
    start_epoch = 0
    if cfg.model.resume:
        print("Resuming run.",end="")
        saved_models = [fn for fn in os.listdir(cfg.logging.logdir) if fn.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        print(f" Models found: {saved_models}")
        # use the latest checkpoint file
        loadckpt = os.path.join(cfg.logging.logdir, saved_models[-1])
        print("Resuming run using model:", loadckpt)
        state_dict = torch.load(loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'])
        if cfg.model.name == "svdd_vae": model.set_svdd_R(state_dict['SVDD_R'])        
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1
        print ("Initial LR: {}\n Last epoch LR: {}".format(optimizer.param_groups[0]["initial_lr"],optimizer.param_groups[0]["lr"]) )
        # Overwriting the initial LR 
        if cfg.model.resume_force_LR:
            print("WARNING: Setting last epoch LR as new initial LR !  --> ",end="")
            # args.lr = optimizer.param_groups[0]["lr"]
            optimizer.param_groups[0]["initial_lr"] = cfg.training.lr
            print ("New initial LR: {}".format(cfg.training.lr) )  
              
    elif cfg.model.loadckpt not in ["None", None] :
        # load checkpoint file specified by cfg.model.loadckpt
        print(f"Loading model: {cfg.model.loadckpt} ")
        state_dict = torch.load(cfg.model.loadckpt, map_location=torch.device("cpu"))
        print ("Last iteration LR: {}".format(state_dict["optimizer"]["param_groups"][0]["lr"]) )
        model.load_state_dict(state_dict['model'])
        current_epoch = state_dict['epoch'] 

    
    ###### Start training, analysis or anomaly detection
    if cfg.model.mode == "train":
        train(model, optimizer, TrainImgLoader, ValImgLoader, start_epoch, tb_logger, cfg)
    elif cfg.model.mode == "AD_analysis":
        AD_analysis(model, TrainImgLoader, ValImgLoader, current_epoch, tb_logger, cfg)
    elif cfg.model.mode == "anomaly_classification":
        anomaly_classification(model, TestImgLoader, current_epoch, tb_logger, cfg)
    else:
        raise NotImplementedError
    