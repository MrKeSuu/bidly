import dataclasses
import json
import typing

from . import metrics


DEFAULT_MIN_IOU = 0.5


@dataclasses.dataclass(frozen=True)
class YoloObject:
    """Class for YOLO detected or ground truth object."""
    name: str
    x: float
    y: float
    w: float
    h: float
    confid: typing.Optional[float]


def _transform_res_to_obj(info):
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

    def __init__(self, gt_path, pred_path) -> None:
        # load
        with open(gt_path, 'r') as fi:
            self.gt_info = json.load(fi)
        with open(pred_path, 'r') as fi:
            self.pred_info = json.load(fi)[0]['objects']

        # transform
        self.gt_objs = _transform_res_to_obj(self.gt_info)
        self.pred_objs = _transform_res_to_obj(self.pred_info)

        self.pairs = list(self._paired_objs(self.gt_objs, self.pred_objs))

        self.gt_proba_info = self._convert_to_gt_proba_info(self.pairs)

    def report_precision_metrics(self):
        pass

    def report_clf_metrics(self, thresh=0.5):
        return metrics.classification_metrics(self.gt_proba_info, self.gt_objs, thresh)

    def report_mean_ap(self, min_iou=DEFAULT_MIN_IOU, classes=None):
        pass

    def _paired_objs(self, gt_objs, pred_objs):
        """Pair GT with Pred based on IOU."""
        paired_gts, paired_preds = set(), set()
        for gt in gt_objs:
            for pred in pred_objs:
                if gt.name == pred.name:
                    iou = _calc_iou(gt, pred)
                    if iou > 0:
                        paired_gts.add(gt)
                        paired_preds.add(pred)
                        yield (gt, pred, iou)

        for gt in gt_objs:
            if gt not in paired_gts:
                yield (gt, None, None)

        for pred in pred_objs:
            if pred not in paired_preds:
                yield (None, pred, None)

    def _convert_to_gt_proba_info(self, pairs, min_iou=0.5):
        gt_n_probas = []
        for gt_obj, pred_obj, iou in pairs:
            if gt_obj is None:
                # non overlapping FP potentially
                y_true, y_pred, name = 0, pred_obj.confid, pred_obj.name
            elif pred_obj is None:
                # FN
                y_true, y_pred, name = 1, 0, gt_obj.name

            elif iou < min_iou:
                # overlapping FP potentially
                y_true, y_pred, name = 0, pred_obj.confid, gt_obj.name
            else:  # iou >= min_iou
                y_true, y_pred, name = 1, pred_obj.confid, gt_obj.name

            gt_n_probas.append((y_true, y_pred, name))
        return gt_n_probas

