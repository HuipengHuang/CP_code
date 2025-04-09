import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, accuracy_score


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def five_scores(bag_labels, bag_predictions,sub_typing=False):
    fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    # threshold_optimal=0.5
    auc_value = roc_auc_score(bag_labels, bag_predictions)
    this_class_label = np.array(bag_predictions)
    this_class_label[this_class_label>=threshold_optimal] = 1
    this_class_label[this_class_label<threshold_optimal] = 0
    bag_predictions = this_class_label
    avg = 'macro' if sub_typing else 'binary'
    precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average=avg)
    accuracy = accuracy_score(bag_labels, bag_predictions)
    return accuracy, auc_value, precision, recall, fscore

