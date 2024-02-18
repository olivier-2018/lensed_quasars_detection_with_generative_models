import numpy as np
from math import floor, ceil, sqrt
import random, os
import torch
import torchvision.utils as vutils
import torch.nn.functional as F
from bisect import bisect_right
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import classification_report, precision_recall_curve,  confusion_matrix, roc_curve, auc, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import PrecisionRecallDisplay, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
from typing import List, Callable, Union, Any, TypeVar, Tuple


def cluster_accuracy(Y_pred, Y):
    Y_pred, Y = np.array(Y_pred), np.array(Y)
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    row, col = linear_sum_assignment(w.max()-w)
    return sum([w[row[i],col[i]] for i in range(row.shape[0])]) * 1.0/Y_pred.size

def nmi_score(Y_pred, Y):
    Y_pred, Y = np.array(Y_pred), np.array(Y)
    assert Y_pred.size == Y.size
    return normalized_mutual_info_score(Y_pred, Y, average_method='arithmetic')
    
def get_output_resolution(in_res: tuple, kernels_dims:list, stride:list, padding:list, verbose:bool=False) -> tuple:
    H, W = in_res    
    output = []
    if verbose: print(f"Evaluate filter resoluation at each layer:")
    for i, (K, S, P) in enumerate(zip(kernels_dims, stride, padding)):
        newH, newW = (H + 2*P - K) / S + 1, (W + 2*P - K) / S + 1
        output.append( (int(newH), int(newW)) )
        H,W = newH, newW
        if verbose: print(f" layer-{i+1} (H,W)={output[-1]}")
    return output[-1]

def create_candidate_training_lists(list_path, cfg):
    # check if dataset/split folder exist
    if not os.path.exists(list_path):
    # if True:
        print (f"Creating training/Testing list from {cfg.dataset.cand_list_all} & {cfg.dataset.cand_list_anomalies}  \
             test-train split ({cfg.dataset.testtrain_split}).")

        # Get candidates filename and create lists
        with open(os.path.join(cfg.dataset.datapath, cfg.dataset.dataset_name, cfg.dataset.cand_list_all)) as f:
             cand_list_all = f.read().splitlines()             
        
        with open(os.path.join(cfg.dataset.datapath, cfg.dataset.dataset_name, cfg.dataset.cand_list_anomalies)) as f:
             cand_list_anomalies = f.read().splitlines()

        cand_list_normal = list(set(cand_list_all)-set(cand_list_anomalies)) 
        N_cand_list_normal = len(cand_list_normal)
        
        # Generate train/val/tes list for anomaly candidates
        ltmp = chunk_into_n(cand_list_anomalies, 3)
        cand_list_anomaly_train = ltmp[0]
        cand_list_anomaly_val = ltmp[1] 
        cand_list_anomaly_test = ltmp[2]
        N_cand_list_anomaly_train = len(cand_list_anomaly_train)
        del ltmp
        print(f"[INFO] Anomaly set split into train/val/test sets with {N_cand_list_anomaly_train}/{len(cand_list_anomaly_val)}/{len(cand_list_anomaly_test)} candidates.") 

        # Split normal candidate set into train/val to meet specified ratio
        N_cand_list_normal_train = int((1 - cfg.dataset.testtrain_split) * N_cand_list_normal) - N_cand_list_anomaly_train
        assert N_cand_list_normal_train < N_cand_list_normal, "List of normal candidates for training higher than total number of normal candidates"
        print(f"[INFO] Normal set split into train/val with {N_cand_list_normal_train}/{N_cand_list_normal-N_cand_list_normal_train} candidates.")
        
        # Generate train/val list for normal candidates
        cand_list_normal_train = random.sample(cand_list_normal, N_cand_list_normal_train)
        cand_list_normal_val = list(set(cand_list_normal)-set(cand_list_normal_train)) 
        N_cand_list_normal_val = len(cand_list_normal_val)
        
        # Generate train/val lists including candidates and label (0=normal, 1=anomaly)
        cand_list_normal_train_tuple = list(zip(cand_list_normal_train, [0 for i in range(N_cand_list_normal_train)]))
        cand_list_normal_val_tuple = list(zip(cand_list_normal_val, [0 for i in range(N_cand_list_normal_val)]))
        cand_list_anomaly_train_tuple = list(zip(cand_list_normal_train, [1 for i in range(N_cand_list_anomaly_train)]))
        cand_list_anomaly_val_tuple = list(zip(cand_list_anomaly_val, [1 for i in range(len(cand_list_anomaly_val))]))        
        cand_list_test = list(zip(cand_list_anomaly_test, [1 for i in range(len(cand_list_anomaly_test))]))
        
        cand_list_train = cand_list_normal_train_tuple + cand_list_anomaly_train_tuple
        cand_list_val = cand_list_normal_val_tuple + cand_list_anomaly_val_tuple
        print(f"[INFO] Training & Validation set generated using {len(cand_list_train)}/{len(cand_list_val)} candidates.")

        # save training & val lists to file        
        os.makedirs(list_path, exist_ok=True)
        with open(os.path.join(list_path, "train.txt"), 'w') as fp:
            for item, label in cand_list_train:
                fp.write(f"{item}, {label}\n")
                
        with open(os.path.join(list_path, "val.txt"), 'w') as fp:
            for item, label in cand_list_val:
                fp.write(f"{item}, {label}\n")
                
        with open(os.path.join(list_path, "test.txt"), 'w') as fp:
            for item, label in cand_list_test:
                fp.write(f"{item}, {label}\n")
        
        with open(os.path.join(list_path, "normal_set.txt"), 'w') as fp:
            for item, label in list(zip(cand_list_normal, [0 for i in range(len(cand_list_normal))])):
                fp.write(f"{item}, {label}\n")
                      
        with open(os.path.join(list_path, "anomaly_set.txt"), 'w') as fp:
            for item, label in list(zip(cand_list_anomalies, [0 for i in range(len(cand_list_anomalies))])):
                fp.write(f"{item}, {label}\n")
    else:
        print (f"Using {list_path}.")

    return list_path


def image_checkerer(images:list):
    import cv2
    N = len(images)
    H,W = images[0].shape
    n = ceil(sqrt(N))
    m = floor(sqrt(N))
    img = np.ones((n*H, n*W))
    # print(f"n,m={(n,m)} img shape:{img.shape}")
    for k in range(N):
        i, j = k // n, k % n
        # print(f"i,j:{i,j} -->  {i*H}:{(i+1)*H} {j*W}:{(j+1)*W}")
        img[i*H:(i+1)*H,j*W:(j+1)*W] = images[k]

    return img

def chunk_into_n(lst:list, n:int)->list:
  random.shuffle(lst)
  chunk_list = []
  for i in range(n):
      chunk_list.append(lst[i::n])
  return chunk_list

def NormalizeNumpy(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# print arguments
def print_args(args):
    print("################################  args  ################################")
    for k, v in args.__dict__.items():
        print("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    print("########################################################################")


# Set random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# multi-debug function
def get_powers(n):
    return [str(p) for p,v in enumerate(bin(n)[:1:-1]) if int(v)]

# Tensorboard related functions
def save_scalars(logger, mode, scalar_dict, global_step):
    scalar_dict = tensor2float(scalar_dict)
    for key, value in scalar_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_scalar(name, value, global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_scalar(name, value[idx], global_step)
    logger.flush()


def save_images(logger, mode, images_dict, global_step):
    images_dict = tensor2numpy(images_dict)

    def preprocess(name, img):
        if not (len(img.shape) == 3 or len(img.shape) == 4):
            raise NotImplementedError("invalid img shape {}:{} in save_images".format(name, img.shape))
        if len(img.shape) == 3:
            img = img[:, np.newaxis, :, :]
        img = torch.from_numpy(img[:1])
        return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True)

    for key, value in images_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_image(name, preprocess(name, value), global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_image(name, preprocess(name, value[idx]), global_step)
    logger.flush()


# Custom LR scheduler
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class DictAverageMeter(object):
    def __init__(self):
        self.data = {}
        self.count = 0

    def update(self, new_input):
        self.count += 1
        if len(self.data) == 0:
            for k, v in new_input.items():
                if not (isinstance(v, float) or isinstance(v, int)):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] = v
        else:
            for k, v in new_input.items():
                if not (isinstance(v, float) or isinstance(v, int)):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] += v

    def mean(self):
        return {k: v / self.count for k, v in self.data.items()}

def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)


class ClassificationMetrics(object):    
    def __init__(self):
        self.data = torch.tensor([], dtype=torch.half)
        self.count = 0

    def update(self, true_labels, pred_labels, pred_proba):
        # assert isinstance(true_labels, torch.tensor), "invalid data type."
        self.count += 1            
        if len(self.data) == 0:
            self.data = torch.stack((true_labels, pred_labels, pred_proba), dim=1)
        else:
            tmp_tensor = torch.stack((true_labels, pred_labels, pred_proba), dim=1)
            self.data = torch.cat([self.data, tmp_tensor], dim=0)

    def get_data(self):
        return self.data[:,0].cpu().numpy(), self.data[:,1].cpu().numpy()
    
    def mean(self):
        return torch.mean(self.data,dim=0, dtype=torch.half)
    
    def nmi(self):
        y_true, y_pred = self.get_data()
        return normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    
    def report(self, display=False):
        y_true, y_pred = self.get_data()
        class_report = classification_report(y_true, y_pred)
        if display: print("==> Classification report:\n",class_report)
        return class_report
    
    def confus_matrix(self, display=False):
        y_true, y_pred = self.get_data()
        conf_matrix = confusion_matrix(y_true, y_pred)
        if display: print("==> Confusion_matrix:\n",conf_matrix)
        return conf_matrix

    def APRF1_scores(self, average="weighted"):
        # average: {‘binary’, ‘micro’, ‘macro’, ‘samples’, ‘weighted’}, default=None
        y_true, y_pred = self.get_data()
        A = accuracy_score(y_true, y_pred)
        P, R, F1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division = 0)
        return A, P, R, F1
    
    def roc_curve(self, display=False):        
        y_true, y_pred_proba = self.data[:,0].cpu().numpy(), self.data[:,2].cpu().numpy()
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)    
        auc_score = auc(fpr, tpr)
        if display:
            fig, ax = plt.figure()
            RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax = ax, name="(AUC="+str(round(auc_score,3))+")")
        return auc_score


########### Wrappers section  ###########

# torch.no_grad wrapper for functions
def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper

# wrapper to compute metrics for each image individually
def compute_metrics_for_each_image(metric_func):
    def wrapper(depth_est, depth_gt, mask, *args):
        batch_size = depth_gt.shape[0]
        results = []
        # compute result one by one
        for idx in range(batch_size):
            ret = metric_func(depth_est[idx], depth_gt[idx], mask[idx], *args)
            results.append(ret)
        return torch.stack(results).mean()

    return wrapper

# wrapper to convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)
    return wrapper


# Helper functions
@make_nograd_func
@compute_metrics_for_each_image
def Thres_metrics(depth_est, depth_gt, mask, thres):
    assert isinstance(thres, (int, float))
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    errors = torch.abs(depth_est - depth_gt)
    err_mask = errors > thres
    # import cv2
    # cv2.imshow("depth", depth_est.cpu().detach().numpy()-depth_gt.cpu().detach().numpy())
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return torch.mean(err_mask.float())


@make_recursive_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    if isinstance(vars, int):
        return float(vars)
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))

@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))

@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.to(torch.device("cuda"))
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))

