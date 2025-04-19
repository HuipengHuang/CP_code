import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, accuracy_score
import torch
from sklearn.preprocessing import label_binarize


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def five_scores(bag_labels, bag_predictions, sub_typing=False):
    print(bag_labels)
    if sub_typing:
        # --- Multi-Class Case ---
        n_classes = len(np.unique(bag_labels))

        # Binarize labels for OvR AUC (if not already one-hot)
        if bag_predictions.ndim == 1 or bag_predictions.shape[1] != n_classes:
            bag_predictions = label_binarize(bag_predictions, classes=np.arange(n_classes))

        # AUC (OvR)
        auc_value = roc_auc_score(bag_labels, bag_predictions, multi_class='ovr')

        # Predict class labels (argmax of probabilities)
        pred_labels = np.argmax(bag_predictions, axis=1)

        # Metrics
        precision, recall, F1, _ = precision_recall_fscore_support(
            bag_labels, pred_labels, average='macro'
        )
        accuracy = accuracy_score(bag_labels, pred_labels)

    else:
        # --- Binary Case ---
        fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)

        # Threshold predictions
        pred_labels = np.where(bag_predictions >= threshold_optimal, 1, 0)

        # AUC and metrics
        auc_value = roc_auc_score(bag_labels, bag_predictions)
        precision, recall, F1, _ = precision_recall_fscore_support(
            bag_labels, pred_labels, average='binary'
        )
        accuracy = accuracy_score(bag_labels, pred_labels)

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