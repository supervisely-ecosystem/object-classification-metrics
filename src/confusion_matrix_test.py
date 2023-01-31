import supervisely as sly
from dotenv import load_dotenv
import os
import pathlib
import requests
import time
from itertools import chain
import numpy as np
import pandas as pd


def get_image_matching(img_names_gt, img_names_pred):
    return len(set(img_names_gt) & set(img_names_pred)) / len(set(img_names_gt) | set(img_names_pred))


def collect_img2classes(api: sly.Api, project_id: int, tagid2cls: dict):
    # 1. collecting img_infos
    # 2. making img2classes
    image_infos = []
    datasets = api.dataset.get_list(project_id)
    for dataset in datasets:
        image_infos += api.image.get_list(dataset.id)

    img2cls = {}
    for img in image_infos:
        img2cls[img.name] = [tagid2cls[tag["tagId"]] for tag in img.tags]
    # img2tags = {img.name: tagid2name[[tag["tagId"] for tag in img.tags]].values for img in image_infos}
    return img2cls


def get_confusion_matrix_multilabel(img2classes_gt: dict, img2classes_pred: dict, classes: list):

    cols = list(classes) + ["None"]
    confusion_matrix = pd.DataFrame(np.zeros((len(cols), len(cols)), dtype=int), columns=cols, index=cols)

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

        confusion_matrix.loc[FN, "None"] += 1
        confusion_matrix.loc["None", FP] += 1
        confusion_matrix.loc[TP, TP] += 1

    return confusion_matrix


def get_confusion_matrix(img2classes_gt: dict, img2classes_pred: dict, classes: list):

    cols = list(classes) + ["None"]
    confusion_matrix = pd.DataFrame(np.zeros((len(cols), len(cols)), dtype=int), columns=cols, index=cols)

    for img_name, classes_gt in img2classes_gt.items():
        classes_pred = img2classes_pred.get(img_name)
        # We only have one class in a list for a single-label classification
        class_gt = classes_gt[0]
        class_pred = classes_pred[0]
        confusion_matrix[class_gt][class_pred] += 1

    return confusion_matrix


def get_dataframes(img2classes_gt: dict, img2classes_pred: dict, classes: list):
    # добавить поддержку single-label
    # обрабатывать ли conf здесь? (обрезка по conf)

    assert img2classes_gt.keys() == img2classes_pred.keys()

    gt = pd.DataFrame(np.zeros((len(img2classes_gt), len(classes)), dtype=int), columns=classes)
    pred = pd.DataFrame(np.zeros((len(img2classes_gt), len(classes)), dtype=int), columns=classes)

    for i, (img_name, classes_gt) in enumerate(img2classes_gt.items()):
        classes_pred = img2classes_pred[img_name]
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


def check_is_task_multilabel(img2classes: dict):
    for img_name, classes in img2classes.items():
        if len(classes) > 1:
            return True
    return False


if __name__ == "__main__":

    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

    api = sly.Api()

    project_id_gt = 16744
    project_id_pred = 16778

    ### 1. MATCH IMAGES

    ### 2. MATCH TAGS
    meta_gt = api.project.get_meta(project_id_gt)
    meta_pred = api.project.get_meta(project_id_pred)
    selected_tags = list(chain(meta_gt["tags"], meta_pred["tags"]))

    ### 3. Settings

    ### 4. confusion_matrix and metrics
    tagid2cls = {tag["id"]: tag["name"] for tag in selected_tags}
    classes = sorted(list(set([tag["name"] for tag in selected_tags])))
    n_classes = len(classes)

    img2classes_gt = collect_img2classes(api, project_id_gt, tagid2cls)
    img2classes_pred = collect_img2classes(api, project_id_pred, tagid2cls)

    print("Image matching:", get_image_matching(img2classes_gt, img2classes_pred))

    confusion_matrix = get_confusion_matrix_multilabel(img2classes_gt, img2classes_pred, classes)

    print(confusion_matrix)

    gt, pred = get_dataframes(img2classes_gt, img2classes_pred, classes)

    print(get_metrics(gt, pred))

    imgs = filter_by_class(img2classes_gt, "cat", not_in=True)
    print(imgs)

    import sklearn.metrics

    res = sklearn.metrics.multilabel_confusion_matrix(gt.values, pred.values, labels=gt.columns)
    print(res)

    report = sklearn.metrics.classification_report(gt.values, pred.values, target_names=classes, output_dict=True)

    LRAP = sklearn.metrics.label_ranking_average_precision_score(gt.values, pred.values)

    def bce(y_true, y_pred, eps=1e-7):
        y_pred = np.clip(y_pred, eps, 1 - eps)
        term_0 = (1 - y_true) * np.log(1 - y_pred + eps)
        term_1 = y_true * np.log(y_pred + eps)
        return -np.mean(term_0 + term_1, axis=0)
