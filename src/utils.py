import pandas as pd
from itertools import chain
from supervisely import TagMetaCollection


# def collect_ds_matching(ds_matching):
#     img2tags_gt = {}
#     img2tags_pred = {}
#     for ds_name, ds_values in ds_matching.items():
#         if ds_values["dataset_matched"] != "both":
#             continue
#         for img_pair in ds_values["matched"]:
#             img_gt, img_pred = img_pair["left"], img_pair["right"]
#             img2tags_gt[img_gt.name] = img_gt.tags
#             img2tags_pred[img_pred.name] = img_pred.tags
#     return img2tags_gt, img2tags_pred


def collect_matching(ds_matching, tags_gt, tags_pred, selected_tags):
    selected_tags = list(filter(lambda x: bool(x[0]) and bool(x[1]), selected_tags))
    cls2clsGT = dict(map(reversed, selected_tags))
    cls2clsGT.update(dict(map(lambda x: (x[0], x[0]), selected_tags)))
    id2tag_gt = tags_gt.get_id_mapping()
    id2tag_pred = tags_pred.get_id_mapping()
    tagId2classGT_gt = lambda tagId: cls2clsGT[id2tag_gt[tagId].name]
    tagId2classGT_pred = lambda tagId: cls2clsGT[id2tag_pred[tagId].name]
    names_keep = set(chain(*selected_tags))
    ids_keep = set(
        [id for id, tag in chain(id2tag_gt.items(), id2tag_pred.items()) if tag.name in names_keep]
    )

    img2classes_gt = {}
    img2classes_pred = {}
    img_name_2_img_info_gt = {}
    img_name_2_img_info_pred = {}
    ds_name_2_img_names = {}
    for ds_name, ds_values in ds_matching.items():
        if ds_values["dataset_matched"] != "both":
            continue
        ds_name_2_img_names[ds_name] = []
        for img_pair in ds_values["matched"]:
            img_gt, img_pred = img_pair["left"], img_pair["right"]
            filtered_classes_gt = [
                tagId2classGT_gt(tag["tagId"]) for tag in img_gt.tags if tag["tagId"] in ids_keep
            ]
            filtered_classes_pred = [
                tagId2classGT_pred(tag["tagId"])
                for tag in img_pred.tags
                if tag["tagId"] in ids_keep
            ]
            img2classes_gt[img_gt.name] = filtered_classes_gt
            img2classes_pred[img_pred.name] = filtered_classes_pred
            img_name_2_img_info_gt[img_gt.name] = img_gt
            img_name_2_img_info_pred[img_pred.name] = img_pred
            ds_name_2_img_names[ds_name].append(img_gt.name)

    classes = list(zip(*selected_tags))[0]  # classes == left selected tag_names
    return (
        img2classes_gt,
        img2classes_pred,
        classes,
        img_name_2_img_info_gt,
        img_name_2_img_info_pred,
        ds_name_2_img_names,
    )


def tagId2name(img2tags: dict, tags: TagMetaCollection):
    id2tag = tags.get_id_mapping()
    return dict(map(lambda img, tag: (img, id2tag[tag["tagId"]].name), *img2tags.items()))


def collect_tag_matching(tag_matching):
    selected_tags = []
    selected_tags = tag_matching["match"]
    return selected_tags


def filter_imgs_without_tags_(img2tags_gt: dict, img2tags_pred: dict):
    for k, tags in list(img2tags_gt.items()):
        if len(tags) == 0:
            img2tags_gt.pop(k)
            img2tags_pred.pop(k, None)


def filter_imgs_(img2tags: dict, remove_keys):
    filtered = img2tags.copy()
    for k in remove_keys:
        del filtered[k]


def is_task_multilabel(img2tags_gt: dict, img2tags_pred: dict):
    for k, tags in chain(img2tags_gt.items(), img2tags_pred.items()):
        if len(tags) != 1:
            return True
    return False


def filter_tags_by_suffix(tags, suffix):
    # filtering "duplicated with suffix" (cat, cat_nn, dog) -> (cat_nn, dog)
    names = set([tag.name for tag in tags])
    filtered_tags = []
    for tag in tags:
        if tag.name + suffix in names:
            continue
        filtered_tags.append(tag)
    return TagMetaCollection(filtered_tags)


def get_overall_metrics(report, mlcm):
    df = pd.DataFrame(report)[["micro avg"]].T
    mlcm_sum = mlcm.sum(0)
    df["TP"] = mlcm_sum[1, 1]
    df["FN"] = mlcm_sum[1, 0]
    df["FP"] = mlcm_sum[0, 1]
    df.index = ["total"]
    df = df.rename(columns={"support": "count"})
    return df


def get_per_class_metrics(report, mlcm, classes):
    df = pd.DataFrame(report).iloc[:, : len(classes)].T
    df["TP"] = mlcm[:, 1, 1]
    df["FN"] = mlcm[:, 1, 0]
    df["FP"] = mlcm[:, 0, 1]
    df["Class"] = classes
    cols = list(df.columns)
    cols = [cols[-1]] + cols[:-1]
    df = df[cols]
    df = df.rename(columns={"support": "count"})
    return df
