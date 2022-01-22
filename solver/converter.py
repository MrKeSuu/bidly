"""Converting .json from yolo into .pbn for pythondds."""
import json

import pandas as pd


class DealConverter:
    card: pd.DataFrame
    card_: pd.DataFrame

    def __init__(self):
        self.card = None
        self.card_ = None

    def read_yolo(self, path):
        with open(path, 'r') as f:
            yolo_json = json.load(f)
        self.card = pd.json_normalize(yolo_json[0]['objects'])

    def dedup(self):
        self.card_ = self._dedup_simple()

    def _dedup_simple(self):
        """Dedup in a simple way, only keeping the one with highest confidence."""
        return (self.card
                    .sort_values('confidence')
                    .drop_duplicates(subset='name', keep='last'))

    def _dedup_smart(self):
        # TODO
        # rule:
        # 1. find out dup pairs whose dist between a range: not too far nor too close
        # 2. reomve from each pair:
        #     1) the one with lower conf, if conf diff to much
        #     2) the onw closer to image center, otherwise
        pass

    # three cases after dedup
    def assign(self):
        """Case 1: everything is perfect -> work on assign cards to four hands"""
        pass

    def report_missing_and_fp(self):
        """Case 2: missing cards or FP cards -> report them"""
        pass

    def infer_missing(self):
        """Case 3: missing cards -> attempt to infer"""
        pass  # * lower priority

    def write_pbn(self, path):
        pass
