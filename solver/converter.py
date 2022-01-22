"""Converting .json from yolo into .pbn for pythondds."""
import json

import pandas as pd


class DealConverter:
    def __init__(self):
        self.card = None

    def read_yolo(self, path):
        with open(path, 'r') as f:
            yolo_json = json.load(f)
        self.card = pd.json_normalize(yolo_json[0]['objects'])

    def dedup(self):
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
