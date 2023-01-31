import os
import json
from collections import defaultdict
import numpy as np
from itertools import chain
import numpy as np
import pandas as pd
import sklearn.metrics

from dotenv import load_dotenv
import supervisely as sly
from supervisely.app.widgets import (
    ConfusionMatrix,
    Container,
    Card,
    SelectDataset,
    Button,
    MatchDatasets,
    Field,
    MatchTagMetas,
    Input,
)

from src import metric_utils
from src import utils
from src import globals as g


load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api()

project_id_gt = os.environ.get("project_id_gt")
project_id_pred = os.environ.get("project_id_pred")
if project_id_gt is not None:
    ds_gt = api.dataset.get_list(project_id_gt)
    ds_pred = api.dataset.get_list(project_id_pred)


### 1. Select Datasets
select_dataset_gt = SelectDataset(project_id=project_id_gt, multiselect=True)
select_dataset_pred = SelectDataset(project_id=project_id_pred, multiselect=True)
card_gt = Card(
    "Ground Truth Project", "Select project with ground truth labels", content=select_dataset_gt
)
card_pred = Card(
    "Prediction Project", "Select project with predicted labels", content=select_dataset_pred
)
select_dataset_btn = Button("Select Datasets")
select_datasets_container = Container([card_gt, card_pred], "horizontal", gap=15)


### 2. Match Datasets
match_datasets = MatchDatasets()
match_datasets_btn = Button("Match")
match_datasets_container = Container([match_datasets, match_datasets_btn])
match_datasets_card = Card(
    "Match datasets",
    "Datasets and their images are compared by name. Only matched pairs of images are used in metrics.",
    True,
    match_datasets_container,
)


@match_datasets_btn.click
def on_match_datasets():
    match_datasets.set(ds_gt, ds_pred, "GT datasets", "Pred datasets")


### 3. Match Tags
# tags_gt = sly.ProjectMeta.from_json(api.project.get_meta(project_id_gt)).tag_metas
# tags_pred = sly.ProjectMeta.from_json(api.project.get_meta(project_id_pred)).tag_metas
match_tags_input = Input("_nn")
match_tags_input_f = Field(
    match_tags_input,
    "Input suffix for Pred tags",
    "If there is no matching you want due to suffix, you can input it manually.",
)
match_tags = MatchTagMetas(selectable=True)
match_tags_btn = Button("Match")
match_tags_container = Container([match_tags_input_f, match_tags, match_tags_btn])
match_tags_card = Card(
    "Match tags",
    "Choose tags/classes that will be used for metrics.",
    True,
    match_tags_container,
)


@match_tags_btn.click
def on_match_tags():
    tags_gt = sly.ProjectMeta.from_json(api.project.get_meta(project_id_gt)).tag_metas
    tags_pred = sly.ProjectMeta.from_json(api.project.get_meta(project_id_pred)).tag_metas
    suffix = match_tags_input.get_value()
    tags_pred_filtered = utils.filter_tags_by_suffix(tags_pred, suffix)
    match_tags.set(tags_gt, tags_pred_filtered, "GT tags", "Pred tags", suffix=suffix)
    g.tags_gt = tags_gt
    g.tags_pred = tags_pred
    g.tags_pred_filtered = tags_pred_filtered


### 4. Confusion Matrix & Metrics
confusion_matrix_widget = ConfusionMatrix()
metrics_btn = Button("Calculate metrics")
metrics_container = Container([confusion_matrix_widget, metrics_btn])
metrics_card = Card(
    "Confusion Matrix & Metrics",
    "",
    True,
    metrics_container,
)


@metrics_btn.click
def on_metrics_click():
    ds_matching = match_datasets.get_stat()
    selected_tags = match_tags.get_selected()
    img2classes_gt, img2classes_pred, classes = utils.collect_matching(
        ds_matching, g.tags_gt, g.tags_pred_filtered, selected_tags
    )
    utils.filter_imgs_without_tags_(img2classes_gt, img2classes_pred)
    is_multilabel = utils.is_task_multilabel(img2classes_gt, img2classes_pred)
    print(f"is_task_multilabel: {is_multilabel}")

    if is_multilabel:
        confusion_matrix = metric_utils.get_confusion_matrix_multilabel(
            img2classes_gt, img2classes_pred, classes
        )
    else:
        confusion_matrix = metric_utils.get_confusion_matrix(
            img2classes_gt, img2classes_pred, classes
        )
    print(confusion_matrix)

    confusion_matrix_widget._update_matrix_data(confusion_matrix)
    from supervisely.app import DataJson

    confusion_matrix_widget.update_data()
    DataJson().send_changes()

    gt, pred = metric_utils.get_dataframes(img2classes_gt, img2classes_pred, classes)
    confusion_matrix = metric_utils.get_confusion_matrix(img2classes_gt, img2classes_pred, classes)
    print(confusion_matrix)
    print(metric_utils.get_metrics(gt, pred))


### FINAL APP
final_container = Container(
    [select_datasets_container, match_datasets_card, match_tags_card, metrics_card], gap=15
)
# final_container = Container([select_datasets_container, match_datasets_card], gap=15)
app = sly.Application(final_container)


### 3. Match Tags


# ### 1. MATCH IMAGES

# ### 2. MATCH TAGS
# meta_gt = api.project.get_meta(project_id_gt)
# meta_pred = api.project.get_meta(project_id_pred)
# selected_tags = list(chain(meta_gt["tags"], meta_pred["tags"]))

# ### 3. Settings

# ### 4. confusion_matrix and metrics
# tagid2cls = {tag["id"]: tag["name"] for tag in selected_tags}
# classes = sorted(list(set([tag["name"] for tag in selected_tags])))
# n_classes = len(classes)

# img2classes_gt = metric_utils.collect_img2classes(api, project_id_gt, tagid2cls)
# img2classes_pred = metric_utils.collect_img2classes(api, project_id_pred, tagid2cls)

# confusion_matrix = metric_utils.get_confusion_matrix_multilabel(img2classes_gt, img2classes_pred, classes)

# print(confusion_matrix)

# gt, pred = metric_utils.get_dataframes(img2classes_gt, img2classes_pred, classes)

# imgs = metric_utils.filter_by_class(img2classes_gt, "cat", not_in=True)
# print(imgs)


# res = sklearn.metrics.multilabel_confusion_matrix(gt.values, pred.values, labels=gt.columns)
# print(res)

# report = sklearn.metrics.classification_report(gt.values, pred.values, target_names=classes, output_dict=True)
