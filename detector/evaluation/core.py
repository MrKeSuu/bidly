import abc
import dataclasses
import json
import typing

import cv2
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from . import metrics
from . import util


DEFAULT_MIN_IOU = 0.7
TEXT_Y_OFFEST = 50 / 1200  # 50 was based on img with H1200
GROUND_TRUTH_EC = (0.5, 1, 0.5)
GROUND_TRUTH_FC = (0.8, 1, 0.8)
PREDICTION_FC = (1, 0.5, 0.5)
PREDICTION_EC = (1, 0.8, 0.8)


@dataclasses.dataclass(frozen=True)
class YoloObject:
    """Class for YOLO detected or ground truth object."""
    name: str
    x: float
    y: float
    w: float
    h: float
    confid: typing.Optional[float] = None


class ILabelReader(abc.ABC):
    @abc.abstractmethod
    def read(self, path) -> typing.List[YoloObject]:
        pass


class Yolo4Reader(ILabelReader):
    @staticmethod
    def _transform_to_objs(info):
        objs = []
        for d in info:
            objs.append(
                YoloObject(
                    name=d['name'],
                    x=d['relative_coordinates']['center_x'],
                    y=d['relative_coordinates']['center_y'],
                    w=d['relative_coordinates']['width'],
                    h=d['relative_coordinates']['height'],
                    confid=d['confidence'],
                )
            )
        return objs


class GroudTruthReader(Yolo4Reader):
    def read(self, path):
        with open(path, 'r') as fi:
            gt_info = json.load(fi)

        objs = self._transform_to_objs(gt_info)
        return objs


class Yolo4PredReader(Yolo4Reader):
    def read(self, path):
        with open(path, 'r') as fi:
            pred_info = json.load(fi)[0]['objects']

        objs = self._transform_to_objs(pred_info)
        return objs


class Yolo5PredReader(ILabelReader):
    def read(self, path):
        pred_info = pd.read_csv(path, sep=' ', header=None)
        pred_info.columns = [
            'cls_id', 'center_x', 'center_y', 'width', 'height', 'confid'
        ]

        with open('yolo-cfg/obj.names', 'r') as f:  # TODO use constant for abs paths
            card_names = [l.strip() for l in f.readlines() if l]
        objs = (
            pred_info
                .assign(card_name=pred_info.cls_id.map(card_names.__getitem__))
                .assign(yolo_obj=lambda df: df.apply(self._make_yolo_obj, axis=1))
                .yolo_obj.tolist()
        )
        return objs

    @staticmethod
    def _make_yolo_obj(row: pd.Series):
        return YoloObject(
            name=row.card_name,
            x=row.center_x,
            y=row.center_y,
            w=row.width,
            h=row.height,
            confid=row.confid,
        )


def _calc_iou(obj1: YoloObject, obj2: YoloObject):  # tested with another impl.
    # intersection first (correctness for 6 cases verified)
    min_w, max_w = 0, min(obj1.w, obj2.w)
    min_h, max_h = 0, min(obj1.h, obj2.h)
    possible_w = obj1.w/2 + obj2.w/2 - abs(obj1.x - obj2.x)
    possible_h = obj1.h/2 + obj2.h/2 - abs(obj1.y - obj2.y)
    inter_w = min(max(min_w, possible_w), max_w)
    inter_h = min(max(min_h, possible_h), max_h)
    area_inter = inter_w * inter_h

    # union
    area1 = obj1.w * obj1.h
    area2 = obj2.w * obj2.h
    area_union = area1 + area2 - area_inter

    iou = area_inter / area_union
    return iou


class Evaluator:
    IOU_LEVELS = [DEFAULT_MIN_IOU, 0.9]
    DIFFICULT_CLASSES = {'As', '4s', 'Ah', '4h', 'Ad', '4d', 'Ac', '4c'}

    def __init__(self, gt_path, pred_path, pred_reader: ILabelReader) -> None:
        self.gt_objs = GroudTruthReader().read(gt_path)
        self.pred_objs = pred_reader.read(pred_path)

    def report_precision_metrics(self):
        results = {}
        for iou in self.IOU_LEVELS:
            iou_ = int(iou*100)
            results[f'mAP{iou_}'] = self.report_mean_ap(iou)
            results[f'subset_mAP{iou_}'] = self.report_mean_ap(iou, self.DIFFICULT_CLASSES)

        return results

    def report_clf_metrics(self, min_iou=DEFAULT_MIN_IOU, thresh=0.5):
        pairs = list(self.paired_objs(min_iou))
        gt_proba_info = self._convert_to_gt_proba_info(pairs)
        return metrics.classification_metrics(gt_proba_info, self.gt_objs, thresh)

    def report_mean_ap(self, min_iou=DEFAULT_MIN_IOU, classes=None):
        pairs = list(self.paired_objs(min_iou))
        gt_proba_info = self._convert_to_gt_proba_info(pairs)
        return metrics.mean_average_precision(gt_proba_info, classes)

    def paired_objs(self, min_iou=DEFAULT_MIN_IOU):
        """Pair GT with Pred based on IOU."""
        paired_gts, paired_preds = set(), set()
        for gt in self.gt_objs:
            for pred in self.pred_objs:
                if gt.name == pred.name:
                    iou = _calc_iou(gt, pred)
                    if iou > min_iou:
                        paired_gts.add(gt)
                        paired_preds.add(pred)
                        yield (gt, pred, iou)

        for gt in self.gt_objs:
            if gt not in paired_gts:
                yield (gt, None, None)

        for pred in self.pred_objs:
            if pred not in paired_preds:
                yield (None, pred, None)

    def _convert_to_gt_proba_info(self, pairs):
        gt_n_probas = []
        for gt_obj, pred_obj, __ in pairs:
            if gt_obj is None:
                # FP potentially
                y_true, y_pred, name = 0, pred_obj.confid, pred_obj.name
            elif pred_obj is None:
                # FN
                y_true, y_pred, name = 1, 0, gt_obj.name
            else:  # iou >= min_iou
                y_true, y_pred, name = 1, pred_obj.confid, gt_obj.name

            gt_n_probas.append((y_true, y_pred, name))
        return gt_n_probas


def plot_paired_boxes(obj1: YoloObject, obj2: YoloObject, ax=None):
    print(obj1, obj2, _calc_iou(obj1, obj2), sep='\n')
    ax = _plot_bbox(obj1, (1, 1), ec='b', ax=ax)
    ax = _plot_bbox(obj2, (1, 1), ec='r', ax=ax)
    return


def plot_misclf(pairs, img_filepath, thresh=0.5, classes=None):
    img = _load_img(img_filepath)

    __, ax = plt.subplots(figsize=(12, 12))
    for gt, pred, __ in pairs:
        if not _is_misclf(gt, pred, thresh):
            continue
        if _not_in_classes(gt, pred, classes):
            continue

        if gt is not None:
            ax = _plot_bbox(gt, img_shape=img.shape, ax=ax, ec='g')
            ax = _plot_label(gt, 'top', img_shape=img.shape, ax=ax, ec=GROUND_TRUTH_EC, fc=GROUND_TRUTH_FC)
        if pred is not None:
            ax = _plot_bbox(pred, img_shape=img.shape, ax=ax, ec='r')
            pred_fc = PREDICTION_FC if pred.confid >= thresh else (.5, .5, .5)  # FN ref, rather than FP
            ax = _plot_label(pred, 'bottom', img_shape=img.shape, ax=ax, ec=PREDICTION_EC, fc=pred_fc)

    ax.imshow(img)


def _load_img(path):
    image = cv2.imread(str(path))
    height, width = image.shape[:2]
    resized_image = cv2.resize(
        image, (width, height),
        interpolation=cv2.INTER_CUBIC)
    converted_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    return converted_image


def _is_misclf(gt: YoloObject, pred: YoloObject, thresh=0.5):
    # Note: similar logics in `_convert_to_gt_proba_info` and `metrics.classification_metrics`
    if gt is None and pred.confid >= thresh:
        return True  # FP
    if pred is None:
        return True  # FN

    if gt is not None and pred.confid < thresh:
        return True  # FN

    return False

def _not_in_classes(gt, pred, classes):
    gt_class = None if gt is None else gt.name
    pd_class = None if pred is None else pred.name
    return not util.in_default(gt_class, classes) and not util.in_default(pd_class, classes)


def _plot_bbox(obj: YoloObject, img_shape, ax=None, **kwargs):
    if ax is None:
        __, ax = plt.subplots(figsize=(12, 12))

    x_scaler, y_scaler = img_shape[:2]

    x = x_scaler * (obj.x - obj.w/2)
    y = y_scaler * (obj.y - obj.h/2)
    w = x_scaler * obj.w
    h = y_scaler * obj.h
    rect = matplotlib.patches.Rectangle(
        (x, y),
        w, h,
        linewidth=.5, facecolor='none', alpha=0.7, **kwargs
    )
    ax.add_patch(rect)
    return ax


def _plot_label(obj: YoloObject, pos, img_shape, ax=None, **kwargs):
    if ax is None:
        __, ax = plt.subplots(figsize=(12, 12))

    x_scaler, y_scaler = img_shape[:2]
    text_y_offset = TEXT_Y_OFFEST if pos == 'bottom' else -TEXT_Y_OFFEST

    text_x = x_scaler * obj.x
    text_y = y_scaler * (obj.y + text_y_offset)
    text = f"{obj.name}"
    if pos == 'bottom':
        text += f", {round(obj.confid, 3)}"

    ax.text(
        text_x, text_y, text,
        ha="center", va="center",
        bbox=dict(boxstyle="round", alpha=0.3, **kwargs),
    )
    return ax
