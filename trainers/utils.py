import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, accuracy_score
import torch

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def five_scores(bag_labels, bag_predictions, sub_typing=False):
    fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    # threshold_optimal=0.5

    if sub_typing:
        auc_value = roc_auc_score(bag_labels, bag_predictions, multi_class='ovr')
    else:
        auc_value = accuracy_score(bag_labels, bag_predictions)
    this_class_label = np.array(bag_predictions)
    this_class_label[this_class_label>=threshold_optimal] = 1
    this_class_label[this_class_label<threshold_optimal] = 0
    bag_predictions = this_class_label
    avg = 'macro' if sub_typing else 'binary'
    precision, recall, F1, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average=avg)
    accuracy = accuracy_score(bag_labels, bag_predictions)
    return accuracy, auc_value, precision, recall, F1

def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps


def roc_threshold(label, prediction):
    fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    c_auc = roc_auc_score(label, prediction)
    return c_auc, threshold_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def eval_metric(oprob, label):

    auc, threshold = roc_threshold(label.cpu().numpy(), oprob.detach().cpu().numpy())
    prob = oprob > threshold
    label = label > threshold

    TP = (prob & label).sum(0).float()
    TN = ((~prob) & (~label)).sum(0).float()
    FP = (prob & (~label)).sum(0).float()
    FN = ((~prob) & label).sum(0).float()

    accuracy = torch.mean(( TP + TN ) / ( TP + TN + FP + FN + 1e-12))
    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    specificity = torch.mean( TN / (TN + FP + 1e-12))
    F1 = 2*(precision * recall) / (precision + recall+1e-12)

    return accuracy, auc, precision, recall, F1