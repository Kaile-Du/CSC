import numpy as np
import torch


def prf_cal(preds, targets, outputs_test):
    thre = 0.8
    prediction = preds.gt(thre).long()
    tp_c = (prediction + targets).eq(2).sum(dim=0)
    fp_c = (prediction - targets).eq(1).sum(dim=0)
    fn_c = (prediction - targets).eq(-1).sum(dim=0)
    tn_c = (prediction + targets).eq(0).sum(dim=0)
    
    #  cf1
    precision_c = [float(tp_c[i].float() / (tp_c[i] + fp_c[i]).float()) * 100.0 if tp_c[i] > 0 else 0.0 for i in range(len(tp_c))]
    recall_c = [float(tp_c[i].float() / (tp_c[i] + fn_c[i]).float()) * 100.0 if tp_c[i] > 0 else 0.0 for i in range(len(tp_c))]
    f1_c = [2 * precision_c[i] * recall_c[i] / (precision_c[i] + recall_c[i]) if tp_c[i] > 0 else 0.0 for i in range(len(tp_c))]

    mean_p_c = sum(precision_c) / len(precision_c)
    mean_r_c = sum(recall_c) / len(recall_c)
    mean_f_c = sum(f1_c) / len(f1_c)

    #  of1
    precision_o = tp_c.sum().float() / (tp_c + fp_c).sum().float() * 100.0
    recall_o = tp_c.sum().float() / (tp_c + fn_c).sum().float() * 100.0
    f1_o = 2 * precision_o * recall_o / (precision_o + recall_o)

    recall_o = tp_c.sum().float() / (tp_c + fn_c).sum().float() * 100.0


    return mean_p_c, mean_r_c, mean_f_c, precision_o.item(), recall_o.item(), f1_o.item()

def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i

def mAP(targs, preds):
    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean(), ap