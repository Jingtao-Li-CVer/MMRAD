from gc import get_threshold
import logging
import matplotlib.pyplot as plt
from numpy import ndarray as NDArray
from sklearn.metrics import roc_auc_score, roc_curve
import os
import random
import numpy as np
from scipy import integrate
from tqdm import tqdm
import torch.nn as nn
import torch
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve



def compute_auroc(epoch: int, ep_reconst, ep_gt, working_dir: str, image_level=False, control_fpr=-1, save_image=False, select_num=-1, compute_bce=False, compute_iou=False, log_info=None,) -> float:
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)

    Args:
        epoch (int): Current epoch
        ep_reconst (NDArray): Reconstructed images in a current epoch
        ep_gt (NDArray): Ground truth masks in a current epoch

    Returns:
        float: AUROC score
    """
    save_dir = os.path.join(working_dir, "epochs-" + str(epoch))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_data = len(ep_reconst)

    if image_level:
        y_score = ep_reconst.reshape(num_data, -1).max(axis=1)
        ep_gt[ep_gt > 1] = 0  # y_score.shape -> (num_data,)
        y_true = ep_gt.reshape(num_data, -1).max(axis=1)  # y_true.shape -> (num_data,)
        scoreDF = roc_auc_score(y_true, y_score)
        logging.info("Image-level max: " + str(scoreDF))
        print("Image-level max: " + str(scoreDF))

        y_score = ep_reconst.reshape(num_data, -1).mean(axis=1)
        y_true = ep_gt.reshape(num_data, -1).max(axis=1)
        scoreDF = roc_auc_score(y_true, y_score)
        logging.info("Image-level mean: " + str(scoreDF))
        print("Image-level mean: " + str(scoreDF))
    else:
        y_score, y_true = [], []

        masks, scores = [], []
        for i, (amap, gt) in enumerate(tqdm(zip(ep_reconst, ep_gt))):

            a = amap[np.where(gt == 0)]
            b = amap[np.where(gt == 1)]
            y_score += a.tolist()
            y_true += np.zeros(len(a)).tolist()
            y_score += b.tolist()
            y_true += np.ones(len(b)).tolist() 
            
            masks.append(gt)
            scores.append(amap)

    if select_num != -1:
        total_selected_num = select_num * num_data
        cc = list(zip(y_score, y_true))
        random.shuffle(cc)
        y_score[:], y_true[:] = zip(*cc)
        y_score = y_score[0:total_selected_num]
        y_true = y_true[0:total_selected_num]
        print("total_selected_num: " + str(total_selected_num))

    scoreDF = roc_auc_score(y_true, y_score)

    log_info('{}: {}\n'.format('scoreDF', str(scoreDF)))

    if compute_iou:
        pass

    if compute_bce:
        criterion = nn.BCELoss()
        y_true = torch.from_numpy(np.array(y_true)).to(torch.float)
        y_score = torch.from_numpy(np.array(y_score)).to(torch.float)
        loss = criterion(y_score, y_true)
        print("bce loss: " + str(loss))

    if save_image:
        
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        
        if control_fpr > 0:
            print("FPR")
            print("\n" + str(fpr))
            print("TPR")
            print("\n" + str(tpr))

        plt.plot(fpr, tpr, marker="o", color="k", label=f"AUROC Score: {round(score, 3)}")
        plt.xlabel("FPR: FP / (TN + FP)", fontsize=14)
        plt.ylabel("TPR: TP / (TP + FN)", fontsize=14)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,"roc_curve.png"))
        plt.close()

    return scoreDF