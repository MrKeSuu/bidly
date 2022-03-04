"""Converting .json from yolo into .pbn for pythondds."""
import json

import pandas as pd


CARD_CLASSES = [
    '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s', '10s', 'Js', 'Qs', 'Ks', 'As',
    '2c', '3c', '4c', '5c', '6c', '7c', '8c', '9c', '10c', 'Jc', 'Qc', 'Kc', 'Ac',
    '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d', '10d', 'Jd', 'Qd', 'Kd', 'Ad',
    '2h', '3h', '4h', '5h', '6h', '7h', '8h', '9h', '10h', 'Jh', 'Qh', 'Kh', 'Ah'
]


class DealConverter:
    card: pd.DataFrame
    card_: pd.DataFrame

    def __init__(self):
        self.card = None
        self.card_ = None

    def read_yolo(self, path):
        with open(path, 'r') as f:
            yolo_json = json.load(f)
        self.card = pd.json_normalize(yolo_json[0]['objects'])  # image has one frame only

    def report_missing_and_fp(self):
        # report missing
        detected_classes = set(self.card.name)
        missing_classes = [name for name in CARD_CLASSES if name not in detected_classes]
        print("Missing cards:", missing_classes)

        # report FP
        fp_classes = self.card.name.value_counts()[lambda s: s > 2].index.tolist()
        print("FP cards:", fp_classes)

    def dedup(self):
        self.card_ = self._dedup_simple()

    def _dedup_simple(self):
        """Dedup in a simple way, only keeping the one with highest confidence."""
        return (self.card
                    .sort_values('confidence')
                    .drop_duplicates(subset='name', keep='last'))

    def _dedup_smart(self):
        """Dedup in a smart way.

        steps:  // YL: got a feeling this is to complex
        1. find out dup pairs whose dist between a range: not too far nor too close
        2. remove dup objs by keeping one with highest conf; could lead to removal of valid objs
        3. append back good dups found in 1. and leave them for assign to decide
        """
        # TODO
        dist = self._calc_pair_dist()

        X_dist = (
            dist.query('0.1 <= dist_ <= 0.3')  # reasonable range of two symbol dist on same card
                .dist_.values.reshape(-1, 1))
        densest_dist = self._find_densest(X_dist)
        good_dup = self._get_good_dup(self.card, dist, densest_dist)

        return (self._dedup_simple()
                    .append(good_dup)
                    .drop_duplicates())

    # two cases after dedup
    def assign(self):
        """Case 1: everything is perfect -> work on assigning cards to four hands"""
        pass

    def infer_missing(self):
        """Case 2: missing cards -> attempt to infer"""
        pass  # *lower priority

    def write_pbn(self, path):
        pass

    def _calc_pair_dist(self):
        pass

    @staticmethod
    def _find_densest(dists, min_size=3):
        pass

    @staticmethod
    def _get_good_dup(card, dist, densest_dist):
        pass
