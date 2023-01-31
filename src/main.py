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
from supervisely.app import DataJson
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
    Table,
    GridGallery,
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
select_dataset_gt._all_datasets_checkbox.check()
select_dataset_pred = SelectDataset(project_id=project_id_pred, multiselect=True)
select_dataset_pred._all_datasets_checkbox.check()
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
    project_id_gt = select_dataset_gt._project_id
    project_id_pred = select_dataset_pred._project_id
    ds_gt = api.dataset.get_list(project_id_gt)
    ds_pred = api.dataset.get_list(project_id_pred)

    match_datasets.set(ds_gt, ds_pred, "GT datasets", "Pred datasets")

    tags_gt = sly.ProjectMeta.from_json(api.project.get_meta(project_id_gt)).tag_metas
    tags_pred = sly.ProjectMeta.from_json(api.project.get_meta(project_id_pred)).tag_metas
    suffix = match_tags_input.get_value()
    tags_pred_filtered = utils.filter_tags_by_suffix(tags_pred, suffix)
    match_tags.set(tags_gt, tags_pred_filtered, "GT tags", "Pred tags", suffix=suffix)

    g.project_id_gt = project_id_gt
    g.project_id_pred = project_id_pred
    g.tags_gt = tags_gt
    g.tags_pred = tags_pred
    g.tags_pred_filtered = tags_pred_filtered
    match_tags_card.uncollapse()


### 3. Match Tags
match_tags_input = Input("_nn")
match_tags_input_f = Field(
    match_tags_input,
    "Input suffix for Pred tags",
    "If there is no matching you want due to suffix, you can input it manually.",
)
match_tags = MatchTagMetas(selectable=True)
match_tags_btn = Button("Select")
match_tags_container = Container([match_tags_input_f, match_tags, match_tags_btn])
match_tags_card = Card(
    "Match tags",
    "Choose tags/classes that will be used for metrics.",
    True,
    match_tags_container,
)
match_tags_card.collapse()


@match_tags_btn.click
def on_match_tags():
    metrics_card.uncollapse()


### 4. Confusion Matrix & Metrics
confusion_matrix_widget = ConfusionMatrix()
metrics_btn = Button("Calculate metrics")
metrics_overall_table = Table()
metrics_per_class_table = Table()
metrics_per_image = Table()
images_gallery = GridGallery(2, show_opacity_slider=False)
metrics_container = Container([confusion_matrix_widget, metrics_overall_table, metrics_btn])
per_image_container = Container(
    [metrics_per_image, images_gallery], direction="horizontal", fractions=[5, 5]
)
metrics_card = Card(
    "Confusion Matrix & Metrics",
    "",
    collapsable=True,
    content=Container([metrics_container, per_image_container]),
)
confusion_matrix_widget.hide()
metrics_overall_table.hide()
metrics_per_class_table.hide()
metrics_per_image.hide()
images_gallery.hide()
metrics_card.collapse()


@metrics_btn.click
def on_metrics_click():
    ds_matching = match_datasets.get_stat()
    selected_tags = match_tags.get_selected()
    (
        img2classes_gt,
        img2classes_pred,
        classes,
        img_name_2_img_info_gt,
        img_name_2_img_info_pred,
        ds_name_2_img_names,
    ) = utils.collect_matching(ds_matching, g.tags_gt, g.tags_pred_filtered, selected_tags)
    utils.filter_imgs_without_tags_(img2classes_gt, img2classes_pred)

    g.img2classes_gt = img2classes_gt
    g.img2classes_pred = img2classes_pred
    g.classes = classes
    g.img_name_2_img_info_gt = img_name_2_img_info_gt
    g.img_name_2_img_info_pred = img_name_2_img_info_pred
    g.ds_name_2_img_names = ds_name_2_img_names

    g.is_multilabel = utils.is_task_multilabel(img2classes_gt, img2classes_pred)
    print(f"is_task_multilabel: {g.is_multilabel}")

    if g.is_multilabel:
        confusion_matrix = metric_utils.get_confusion_matrix_multilabel(
            img2classes_gt, img2classes_pred, classes
        )
    else:
        confusion_matrix = metric_utils.get_confusion_matrix(
            img2classes_gt, img2classes_pred, classes
        )
    print(confusion_matrix)

    confusion_matrix_widget._update_matrix_data(confusion_matrix)
    confusion_matrix_widget.update_data()
    DataJson().send_changes()

    gt, pred = metric_utils.get_dataframes(img2classes_gt, img2classes_pred, classes)
    report = sklearn.metrics.classification_report(
        gt.values, pred.values, target_names=classes, output_dict=True
    )
    mlcm = sklearn.metrics.multilabel_confusion_matrix(gt.values, pred.values)
    # [[TN, FP]
    #  [FN, TP]]

    df = pd.DataFrame(report)[["micro avg"]].T
    mlcm_sum = mlcm.sum(0)
    df["TP"] = mlcm_sum[1, 1]
    df["FN"] = mlcm_sum[1, 0]
    df["FP"] = mlcm_sum[0, 1]
    df.index = ["total"]
    metrics_overall_table._update_table_data(input_data=df)
    metrics_overall_table.update_data()
    DataJson().send_changes()

    df = pd.DataFrame(report).iloc[:, : len(classes)]

    confusion_matrix_widget.show()
    metrics_overall_table.show()
    metrics_per_class_table.show()

    ### For debug:
    confusion_matrix = metric_utils.get_confusion_matrix(img2classes_gt, img2classes_pred, classes)
    print(confusion_matrix)
    print(metric_utils.get_metrics(gt, pred))
    print(report)


@confusion_matrix_widget.click
def on_confusion_matrix_click(cell: ConfusionMatrix.ClickedDataPoint):
    cls_gt = cell.row_name
    cls_pred = cell.column_name
    if cls_gt != "None":
        img_names1 = metric_utils.filter_by_class(g.img2classes_gt, cls_gt)
    else:
        img_names1 = metric_utils.filter_by_class(g.img2classes_gt, cls_pred, True)
    if cls_pred != "None":
        img_names2 = metric_utils.filter_by_class(g.img2classes_pred, cls_pred)
    else:
        img_names2 = metric_utils.filter_by_class(g.img2classes_pred, cls_gt, True)
    img_names = list(set(img_names1) & set(img_names2))

    columns = ["GT_IMG_ID", "PRED_IMG_ID", "NAME", "TP", "FP", "FN"]  # ["link"]
    # tags from img names
    # img2classes_gt {img_nmae: [tag]}
    # tag_meta from tags_gt, pred
    # img_name_2_img_id_gt - ID
    # img_name to image_info
    rows = [
        [
            g.img_name_2_img_info_gt[image_name].id,
            g.img_name_2_img_info_pred[image_name].id,
            image_name,
            *metric_utils.img_metrics(g.img2classes_gt[image_name], g.img2classes_pred[image_name]),
        ]
        for image_name in img_names
    ]

    df = pd.DataFrame(rows, columns=columns)
    metrics_per_image.read_pandas(df)
    metrics_per_image.show()
    set_img_to_gallery(img_names[0])
    images_gallery.show()


@metrics_per_image.click
def select_image_row(cell: Table.ClickedDataPoint):
    image_name = cell.row["NAME"]
    set_img_to_gallery(image_name)


def set_img_to_gallery(image_name):
    images_gallery.loading = True
    img_tags_gt = []
    for tag_json in g.img_name_2_img_info_gt[image_name].tags:
        if "name" not in tag_json.keys():
            sly_id = tag_json["tagId"]
            for tag_meta in g.tags_gt:
                tag_meta: sly.TagMeta
                if tag_meta.sly_id == sly_id:
                    tag_json["name"] = tag_meta.name
                    break
        tag = sly.Tag.from_json(tag_json, g.tags_gt)
        img_tags_gt.append(tag)

    img_tags_pred = []
    for tag_json in g.img_name_2_img_info_pred[image_name].tags:
        if "name" not in tag_json.keys():
            sly_id = tag_json["tagId"]
            for tag_meta in g.tags_pred:
                tag_meta: sly.TagMeta
                if tag_meta.sly_id == sly_id:
                    tag_json["name"] = tag_meta.name
                    break
        tag = sly.Tag.from_json(tag_json, g.tags_pred)
        img_tags_pred.append(tag)

    img_info_gt = g.img_name_2_img_info_gt[image_name]
    img_info_pred = g.img_name_2_img_info_pred[image_name]
    images_for_preview = utils.get_preview_image_pair(
        img_info_gt,
        img_info_pred,
        img_tags_gt,
        img_tags_pred,
        g.is_multilabel,
    )

    images_gallery.clean_up()
    for current_image in images_for_preview:
        images_gallery.append(image_url=current_image["url"], title=current_image["title"])

    images_gallery.loading = False


### FINAL APP
final_container = Container(
    [select_datasets_container, match_datasets_card, match_tags_card, metrics_card], gap=15
)
app = sly.Application(final_container)
