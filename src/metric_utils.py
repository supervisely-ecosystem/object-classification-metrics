import supervisely as sly
from dotenv import load_dotenv
import os
import pathlib
import requests
import time
from itertools import chain
import numpy as np
import pandas as pd


def get_confusion_matrix_multilabel(img2classes_gt: dict, img2classes_pred: dict, classes: list):

    cols = list(classes) + ["None"]
    confusion_matrix = pd.DataFrame(
        np.zeros((len(cols), len(cols)), dtype=int), columns=cols, index=cols
    )

    for img_name, classes_gt in img2classes_gt.items():
        classes_pred = img2classes_pred.get(img_name)
        # TODO: checking for existence is redundant here.
        if classes_pred is None:
            FN = classes_gt
            FP = []
            TP = []
        else:
            gt, pred = set(classes_gt), set(classes_pred)
            FN = list(gt - pred)
            FP = list(pred - gt)
            TP = list(gt & pred)

        for cls in FN:
            confusion_matrix.loc[cls, "None"] += 1
        for cls in FP:
            confusion_matrix.loc["None", cls] += 1
        for cls in TP:
            confusion_matrix.loc[cls, cls] += 1

    return confusion_matrix


def get_confusion_matrix(img2classes_gt: dict, img2classes_pred: dict, classes: list):

    cols = list(classes) + ["None"]
    confusion_matrix = pd.DataFrame(
        np.zeros((len(cols), len(cols)), dtype=int), columns=cols, index=cols
    )

    for img_name, classes_gt in img2classes_gt.items():
        classes_pred = img2classes_pred.get(img_name)
        # We only have one class in a list for a single-label classification
        class_gt = classes_gt[0] if len(classes_gt) else "None"
        class_pred = classes_pred[0] if len(classes_pred) else "None"
        confusion_matrix[class_gt][class_pred] += 1

    return confusion_matrix


def get_dataframes(
    img2classes_gt: dict, img2classes_pred: dict, classes: list, is_single_label=True
):

    assert img2classes_gt.keys() == img2classes_pred.keys()

    gt = pd.DataFrame(np.zeros((len(img2classes_gt), len(classes)), dtype=int), columns=classes)
    pred = pd.DataFrame(np.zeros((len(img2classes_gt), len(classes)), dtype=int), columns=classes)

    for i, (img_name, classes_gt) in enumerate(img2classes_gt.items()):
        classes_pred = img2classes_pred[img_name]
        if is_single_label:
            # taking the first item as we already has sorted by conf earlier
            classes_gt = classes_gt[0] if len(classes_gt) else "None"
            classes_pred = classes_pred[0] if len(classes_pred) else "None"
        gt.loc[i, classes_gt] = 1
        pred.loc[i, classes_pred] = 1

    return gt, pred


def get_metrics(gt: pd.DataFrame, pred: pd.DataFrame):
    gt, pred = gt.values, pred.values

    acc = (gt == pred).mean(0)

    P = gt == 1
    N = ~P

    PP = pred == 1
    PN = ~PP

    T = gt == pred
    F = ~T

    TP = T & PP
    TN = T & PN
    FP = F & PP
    FN = F & PN

    precision = TP.sum(0) / (0 + TP + FP).sum(0)
    recall = TP.sum(0) / (0 + TP + FN).sum(0)

    ### Filling NaNs
    # p_exists = P.sum(0) > 0
    # precision[np.isnan(precision) & p_exists] = 0.0
    # precision[np.isnan(precision) & ~p_exists] = 1.0
    # recall[np.isnan(recall) & p_exists] = 0.0
    # recall[np.isnan(recall) & ~p_exists] = 1.0

    return acc, precision, recall


def filter_by_class(img2classes: dict, cls: str, not_in=False):
    img_names = []
    for img_name, img_classes in img2classes.items():
        if not_in is False and cls in img_classes:
            img_names.append(img_name)
        if not_in is True and cls not in img_classes:
            img_names.append(img_name)
    return img_names


# def check_is_task_multilabel(img2classes: dict):
#     for img_name, classes in img2classes.items():
#         if len(classes) > 1:
#             return True
#     return False


# def bce(y_true, y_pred, eps=1e-7):
#     y_pred = np.clip(y_pred, eps, 1 - eps)
#     term_0 = (1 - y_true) * np.log(1 - y_pred + eps)
#     term_1 = y_true * np.log(y_pred + eps)
#     return -np.mean(term_0 + term_1, axis=0)


def img_metrics(gt_tags, pred_tags, is_multilabel, suffix=None):
    if is_multilabel:
        if suffix is not None:
            pred_tags = [
                tag.name[: -len(suffix)] if tag.name.endswith(suffix) else tag.name
                for tag in pred_tags
            ]
        else:
            pred_tags = [tag.name for tag in pred_tags]

        gt_tags = [tag.name for tag in gt_tags]
        gt_tags = set(gt_tags)
        pred_tags = set(pred_tags)
        tp = len(gt_tags & pred_tags)
        fp = len(pred_tags - gt_tags)
        fn = len(gt_tags - pred_tags)
        return [tp, fp, fn]
    else:  # single-label
        gt_tags = gt_tags[0].name
        if suffix is not None and pred_tags[0].name.endswith(suffix):
            pred_tags = pred_tags[0].name[: -len(suffix)]
        else:
            pred_tags = pred_tags[0].name
        correct = gt_tags == pred_tags
        return [correct]
