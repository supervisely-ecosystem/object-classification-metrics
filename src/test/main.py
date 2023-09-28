import supervisely as sly
import numpy as np

try:
    import pyximport
    pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=False)
    from .compute_overlap import compute_overlap
except:
    print("Couldn't import compute_overlap. Check cython installation.")
    from compute_overlap_slow import compute_overlap


def match_bboxes(pairwise_iou: np.ndarray, min_iou_threshold=0.15):
    # pairwise_iou shape: [pred, gt]
    n_pred, n_gt = pairwise_iou.shape

    matched_idxs = []
    ious_matched = []
    unmatched_idxs_gt, unmatched_idxs_pred = set(range(n_gt)), set(range(n_pred))

    m = pairwise_iou.flatten()
    for i in m.argsort()[::-1]:
        if m[i] < min_iou_threshold or m[i] == 0:
            break
        pred_i = i // n_gt  # row
        gt_i = i % n_gt  # col
        if gt_i in unmatched_idxs_gt and pred_i in unmatched_idxs_pred:
            matched_idxs.append([gt_i, pred_i])
            ious_matched.append(m[i])
            unmatched_idxs_gt.remove(gt_i)
            unmatched_idxs_pred.remove(pred_i)

    return matched_idxs, list(unmatched_idxs_gt), list(unmatched_idxs_pred), ious_matched


def collect_labels_with_tags(labels: list, selected_tags: list, unsuffix: dict = None):
    selected_tags = set(selected_tags)
    bboxes, classes = [], []
    for label in labels:
        label : sly.Label
        if not isinstance(label.geometry, sly.Rectangle):
            continue
        
        tags = []
        scores = []
        for tag in label.tags:
            tag_name = unsuffix[tag.name] if unsuffix else tag.name
            if tag_name in selected_tags:
                tags.append(tag_name)
                # if the tag has a confidence value
                if isinstance(tag.get("value"), float):
                    scores.append(tag["value"])

        if len(tags) == 0:
            continue

        # sort tags by confidence score
        if len(tags) == len(scores):
            tags, _ = list(zip(*sorted(zip(tags, scores), key=lambda pair: pair[1], reverse=True)))
            tags = list(tags)
        else:
            # If we don't have scores for every tag, the mode is single-label and the label has multiple tags,
            # it can lead to incorrect metric calculation.
            pass

        rect = label.geometry
        bbox = [rect.left, rect.top, rect.right, rect.bottom]
        bboxes.append(bbox)
        classes.append(tags)
    return bboxes, classes


def get_image_match(bboxes_gt, bboxes_pred, classes_gt, classes_pred, iou_threshold=0.5):

    if len(bboxes_gt) == 0 and len(bboxes_pred) == 0:
        return [], []

    if len(bboxes_gt) != 0 and len(bboxes_pred) != 0:
        # [Pred x GT]
        pairwise_iou = compute_overlap(np.array(bboxes_pred, dtype=np.float64), np.array(bboxes_gt, dtype=np.float64))
        matched_idxs, unmatched_idxs_gt, unmatched_idxs_pred, box_ious_matched = match_bboxes(pairwise_iou, iou_threshold)
    else:
        matched_idxs = []
        unmatched_idxs_gt = list(range(len(bboxes_gt)))
        unmatched_idxs_pred = list(range(len(bboxes_pred)))

    matched_classes_gt = []
    matched_classes_pred = []
    for i_gt, i_pred in matched_idxs:
        matched_classes_gt.append(classes_gt[i_gt])
        matched_classes_pred.append(classes_pred[i_pred])

    for i_gt in unmatched_idxs_gt:
        matched_classes_gt.append(classes_gt[i_gt])
        matched_classes_pred.append([])

    for i_pred in unmatched_idxs_pred:
        matched_classes_pred.append(classes_pred[i_pred])
        matched_classes_gt.append([])

    # [["car", "cat"], ["person"], [], []] vs. [["cat"], [], ["car"], ["car"]]
    return matched_classes_gt, matched_classes_pred


def download_annotations(api: sly.Api, project_id):
    id2ann = {}
    
    project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
    datasets = api.dataset.get_list(project_id)
    for dataset in datasets:
        ann_infos = api.annotation.get_list(dataset.id)
        for ann_info in ann_infos:
            ann = sly.Annotation.from_json(ann_info.annotation, project_meta)
            id2ann[ann_info.image_id] = ann

    return id2ann

def download_annotations_gt_and_pred(api: sly.Api, project_id_gt, project_id_pred):
    id2ann = download_annotations(api, project_id_gt)
    id2ann.update(download_annotations(api, project_id_pred))
    return id2ann


def main(ds_match, id2ann, selected_tags, unsuffix, is_multilabel, iou_threshold):

    img2classes_gt, img2classes_pred = {}, {}

    for i, (dataset_name, item) in enumerate(ds_match.items()):
        if item["dataset_matched"] != "both":
            continue
        for img_item in item["matched"]:
            img_gt, img_pred = img_item["left"], img_item["right"]
            img_gt: sly.ImageInfo
            img_pred: sly.ImageInfo
            ann_gt: sly.Annotation = id2ann[img_gt.id]
            ann_pred: sly.Annotation = id2ann[img_pred.id]

            bboxes_gt, classes_gt = collect_labels_with_tags(ann_gt.labels, selected_tags)
            bboxes_pred, classes_pred = collect_labels_with_tags(ann_pred.labels, selected_tags, unsuffix)

            matched_classes_gt, matched_classes_pred = get_image_match(bboxes_gt, bboxes_pred, classes_gt, classes_pred, iou_threshold)

            img2classes_gt[img_gt.name] = matched_classes_gt
            img2classes_pred[img_gt.name] = matched_classes_pred

